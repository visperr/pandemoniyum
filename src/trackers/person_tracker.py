import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
from core.base_tracker import BaseTracker
from models.person_data import PersonTrackingData
from models.scene_data import SceneTrackingData
from networks.osnet import OSNetEmbedder
from utils import *

class PersonTracker(BaseTracker):

    def __init__(self, model_dir: str, num_persons: int = 1, frame_skip: int = 1):
        self.num_persons = num_persons
        self.frame_skip = frame_skip

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.player_galleries = {}
        self.player_last_positions = {}
        self.track_to_player = {}

        self.frame_count = 0
        self.last_detections = []
        self.last_embeddings = []

        super().__init__(model_dir)

    def _initialize_model(self):
        print(f"Using {self.device}")

        self.detector = YOLO(os.path.join(self.model_dir, "yolov8m.pt")) #Sett to yolov8n.pt for faster computing, may at wrong detection when extra people are close to the players.
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
        if self.frame_count % self.frame_skip == 1 or self.frame_skip == 1:
            results = self.detector(frame, classes=[0], device=self.device, verbose=False)[0]
    
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
            self.last_detections, self.last_embeddings = detections, embeddings
        else:
            detections, embeddings = self.last_detections, self.last_embeddings

        return detections, embeddings
    
    def process_frame(self, frame: np.ndarray) -> SceneTrackingData:
        self.frame_count += 1

        scene_data = SceneTrackingData(
            capture_time=get_milliseconds_since_epoch()
        )

        detections, embeddings = self.detect(frame)

        tracks = self.tracker.update_tracks(detections, embeds=embeddings, frame=frame)

        confirmed_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]
        track_boxes = {t.track_id: t.to_ltrb() for t in confirmed_tracks}

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
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            if track_id in self.track_to_player:
                slot = self.track_to_player[track_id]
                if slot in assigned_slots_this_frame:
                    del self.track_to_player[track_id]
                else:
                    assigned_slots_this_frame.add(slot)
                    self.player_last_positions[slot] = (center_x, center_y)
                    
                    if feature is not None and track_id not in occluded_track_ids:
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
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                person_data.append(PersonTrackingData(
                    id=self.track_to_player[track_id],
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    occluded=(track_id in occluded_track_ids),
                    tracking_active=True
                ))

            else:
                print("offscreen")

        self.last_data = person_data

        scene_data.persons = person_data
        scene_data.processing_time = get_milliseconds_since_epoch() - scene_data.capture_time

        return scene_data

    def draw_debug(self, frame: np.ndarray) -> None:
        if self.last_data is None:
            pass

        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

        for player in self.last_data:
            color = colors[(player.id - 1) % len(colors)]
            
            cv2.rectangle(frame, (player.x1, player.y1), (player.x2, player.y2), color, 2)
            status_str = f"Player {player.id}" + (" [LOCKED]" if player.is_occluded else "")
            cv2.putText(frame, status_str, (player.x1, player.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def close(self) -> None:
        pass
