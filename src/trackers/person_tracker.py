import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
import utils
from core.base_tracker import BaseTracker
from models.person_data import PersonTrackingData
from models.scene_data import SceneTrackingData
from networks.osnet import OSNetEmbedder
from utils import *

class PersonTracker(BaseTracker):

    def __init__(self, model_dir: str, num_persons: int = 1, frame_skip: int = 1, freeze_frames: int = 0):
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        if freeze_frames < 0:
            raise ValueError("freeze_frames must not be negative")

        # DeepSort deletes a track after max_age missed frames, so a freeze cannot outlast that.
        max_freeze = max(0, config.PERSON_MAX_AGE_FRAMES - frame_skip)
        if freeze_frames > max_freeze:
            print(f"PERSON_FREEZE_FRAMES={freeze_frames} exceeds what PERSON_MAX_AGE_FRAMES allows; using {max_freeze}")
            freeze_frames = max_freeze

        self.num_persons = num_persons
        self.frame_skip = frame_skip
        self.freeze_frames = freeze_frames
        self.last_boxes = {}

        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.player_galleries = {}
        self.player_last_positions = {}
        self.track_to_player = {}

        self.frame_count = 0
        self.last_tracks = []

        super().__init__(model_dir)

    def _initialize_model(self):
        print(f"Using {self.device}")

        self.detector = YOLO(os.path.join(self.model_dir, "yolov8m.pt"))
        self.detector.to(self.device)

        self.embedder = OSNetEmbedder(model_dir=self.model_dir, device=self.device)

        self.tracker = DeepSort(
            max_age=config.PERSON_MAX_AGE_FRAMES,
            n_init=2,
            nn_budget=300,
            max_cosine_distance=config.PERSON_MAX_COSINE_DIST,
            embedder=None,
            embedder_gpu=(self.device=="cuda")
        )

        return None

    def detect(self, frame: np.ndarray) -> tuple[list, list]:
        if (self.frame_count - 1) % self.frame_skip == 0:
            results = self.detector(
                frame,
                classes=[0],
                device=self.device,
                imgsz=config.PERSON_DETECTOR_IMAGE_SIZE,
                verbose=False
            )[0]
    
            detections = []
            crops = []
    
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                
                h_img, w_img, _ = frame.shape
    
                for box, conf in zip(boxes, confs):
                    if conf > config.PERSON_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        
                        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                        ix2, iy2 = min(w_img, int(x2)), min(h_img, int(y2))
                        
                        crop = frame[iy1:iy2, ix1:ix2]
                        if crop.size > 0:
                            detections.append(([x1, y1, w, h], conf, 'person'))
                            crops.append(crop)
    
            embeddings = self.embedder(crops) if len(crops) > 0 else []
            return detections, embeddings
        else:
            return [], []
    
    def process_frame(self, frame: np.ndarray) -> SceneTrackingData:
        self.frame_count += 1

        scene_data = SceneTrackingData(
            capture_time=get_milliseconds_since_epoch()
        )

        detections, embeddings = self.detect(frame)

        if (self.frame_count - 1) % self.frame_skip == 0:
            tracks = self.tracker.update_tracks(detections, embeds=embeddings, frame=frame)
            self.last_tracks = tracks
        else:
            for track in self.tracker.tracker.tracks:
                if track.is_confirmed():
                    track.predict(self.tracker.tracker.kf)
            tracks = self.last_tracks

        # time_since_update counts frames since the track was last matched to a detection.
        # - healthy: matched recently enough; the box is the tracker's current estimate.
        # - frozen:  detection was lost, but the person is held for `freeze_frames` more frames at the
        #            last known box. Only players that already own a slot are held. Healthy tracks are
        #            listed first so they win a slot if a new track took over for a frozen one.
        healthy_tracks = []
        frozen_tracks = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            if t.time_since_update < self.frame_skip:
                healthy_tracks.append(t)
            elif (
                t.time_since_update < self.frame_skip + self.freeze_frames
                and t.track_id in self.track_to_player
                and t.track_id in self.last_boxes
            ):
                frozen_tracks.append(t)

        track_boxes = {}
        for t in healthy_tracks:
            track_boxes[t.track_id] = np.array(t.to_ltrb())
            self.last_boxes[t.track_id] = track_boxes[t.track_id]
        for t in frozen_tracks:
            track_boxes[t.track_id] = self.last_boxes[t.track_id]

        frozen_track_ids = {t.track_id for t in frozen_tracks}
        confirmed_tracks = healthy_tracks + frozen_tracks

        known_ids = {t.track_id for t in tracks}
        self.last_boxes = {k: v for k, v in self.last_boxes.items() if k in known_ids}

        occluded_track_ids = set()
        for i, t1 in enumerate(confirmed_tracks):
            for t2 in confirmed_tracks[i+1:]:
                iou = compute_iou(track_boxes[t1.track_id], track_boxes[t2.track_id])
                if iou > 0.15:
                    occluded_track_ids.add(t1.track_id)
                    occluded_track_ids.add(t2.track_id)

        assigned_slots_this_frame = set()
        unassigned_tracks = []

        for track in confirmed_tracks:
            track_id = track.track_id
            feature = track.features[-1] if hasattr(track, 'features') and len(track.features) > 0 else None
            x1, y1, x2, y2 = map(int, track_boxes[track_id])
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            if track_id in self.track_to_player:
                slot = self.track_to_player[track_id]
                if slot in assigned_slots_this_frame:
                    del self.track_to_player[track_id]
                else:
                    assigned_slots_this_frame.add(slot)
                    self.player_last_positions[slot] = (center_x, center_y)
                    
                    if (
                        feature is not None
                        and track.time_since_update == 0
                        and track_id not in occluded_track_ids
                    ):
                        if slot not in self.player_galleries:
                            self.player_galleries[slot] = []
                        if len(self.player_galleries[slot]) >= 15:
                            self.player_galleries[slot].pop(0)
                        self.player_galleries[slot].append(feature)

            elif feature is not None and len(self.player_galleries) > 0:
                best_match_slot = None
                highest_sim = config.PERSON_REID_SIMILARITY_THRESHOLD

                for slot, gallery in self.player_galleries.items():
                    if slot in assigned_slots_this_frame:
                        continue 
                    
                    app_sim = get_gallery_similarity(feature, gallery)
                    spatial_score = 1.0
                    if slot in self.player_last_positions:
                        last_x, last_y = self.player_last_positions[slot]
                        dist = np.hypot(center_x - last_x, center_y - last_y)
                        if dist > 250:
                            spatial_score = 0.70 

                    total_score = app_sim * spatial_score

                    if total_score > highest_sim:
                        highest_sim = total_score
                        best_match_slot = slot

                if best_match_slot is not None:
                    self.track_to_player[track_id] = best_match_slot
                    assigned_slots_this_frame.add(best_match_slot)
                    self.player_last_positions[best_match_slot] = (center_x, center_y)
                else:
                    unassigned_tracks.append((center_x, center_y, track, feature))
            else:
                unassigned_tracks.append((center_x, center_y, track, feature))

        if unassigned_tracks and len(self.player_galleries) < self.num_persons:
            unassigned_tracks.sort(key=lambda item: item[0])
            for center_x, center_y, track, feature in unassigned_tracks:
                if len(self.player_galleries) >= self.num_persons:
                    break
                
                track_id = track.track_id
                assigned_slots = set(self.player_galleries.keys())
                new_slot = next(s for s in range(1, self.num_persons + 1) if s not in assigned_slots)

                self.player_galleries[new_slot] = [feature] if feature is not None else []
                self.player_last_positions[new_slot] = (center_x, center_y)
                self.track_to_player[track_id] = new_slot
                assigned_slots_this_frame.add(new_slot)

        person_data = []

        for track in confirmed_tracks:
            track_id = track.track_id
            if track_id in self.track_to_player:
                x1, y1, x2, y2 = map(int, track_boxes[track_id])
                person_data.append(PersonTrackingData(
                    id=self.track_to_player[track_id],
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    occluded=(track_id in occluded_track_ids),
                    frozen=(track_id in frozen_track_ids),
                    tracking_active=True
                ))

        self.last_data = person_data

        scene_data.persons = person_data
        scene_data.processing_time = get_milliseconds_since_epoch() - scene_data.capture_time

        return scene_data

    def draw_debug(self, frame: np.ndarray, notes: dict[int, str]) -> None:
        if self.last_data is None:
            pass

        for player in self.last_data:
            color = utils.player_color_from_id(player.id)

            note = notes[player.id] if player.id in notes else ""
            
            status_str = f"Player {player.id} " + ("[LOCKED] " if player.is_occluded else "") + ("[FROZEN] " if player.frozen else "") + note
            
            cv2.rectangle(frame, (player.x1, player.y1), (player.x2, player.y2), color, 2)
            cv2.putText(frame, status_str, (player.x1, player.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def close(self) -> None:
        pass
