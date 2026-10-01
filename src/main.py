import cv2
import sys
import os
import time
import json
import numpy as np
from pathlib import Path

import config
import utils
from core.data_packer import DataPacker
from core.network_streamer import UDPStreamer
from trackers.face_tracker import FaceTracker
from trackers.person_tracker import PersonTracker
from trackers.pose_tracker import PoseTracker


def get_asset_path(relative_path: str) -> str:
    """Resolve correct asset path when running normally or from a PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def open_video_capture() -> cv2.VideoCapture:
    if config.USE_CAMERA:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(config.VIDEO_PATH)
    return cap

def open_video_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    if not output_path.parent.exists():
        raise Exception("Save path folder does not exist")
    
    print(f"Saving output video to: {output_path}")

    # Initialize video writer for test_output.mp4 using mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback FPS for webcams or unread video headers

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return out


def main():
    streamer = UDPStreamer(ip=config.UDP_IP, port=config.UDP_PORT)
    packer = DataPacker()

    model_dir = get_asset_path(config.MODEL_DIR)
    face_tracker = FaceTracker(
        model_dir=model_dir,
        num_faces=config.FACE_NUM_FACES,
        min_tracking_confidence=config.FACE_MIN_CONFIDENCE,
        blendshapes=config.FACE_BLENDSHAPES
    )

    person_tracker = PersonTracker(
        model_dir=model_dir,
        num_persons=config.PERSON_NUM_PERSONS,
    )

    pose_tracker = PoseTracker(
        model_dir=model_dir,
        num_poses=config.POSE_NUM_POSES,
        min_tracking_confidence=config.POSE_MIN_CONFIDENCE
    )

    cap = open_video_capture()

    if config.SAVE_PREVIEW:
        out = open_video_writer(cap, Path(config.SAVE_PATH))

    print("Tracking pipeline started. Press 'q' to exit.")

    prev_time = 0.0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            # 1. Process tracking components
            scene_data = person_tracker.process_frame(frame)
            notes = {}

            for person in scene_data.persons:
                if not person.is_tracking or (person.x2 - person.x1) < 20 or (person.y2 - person.y1) < 20:
                    continue

                # TODO determine better/smaller subframe for face detection based on datapoints 0-10 from the pose detection output!

                subframe = frame[person.y1:person.y2, person.x1:person.x2]

                if subframe.shape[0] < 20 or subframe.shape[1] < 20:
                    print(f"sf {subframe.shape[0]}, {subframe.shape[1]}")
                    continue

                person.pose = pose_tracker.process_frame(subframe)
                person.face = face_tracker.process_frame(subframe)

                # TODO for demonstration only
                if "jawOpen" in person.face.blendshapes and person.face.blendshapes["jawOpen"] > 0.05:
                    notes[person.id] = "(JAW OPEN)"

            # 2. Add tracking models incrementally
            # packer.add(face_data)
            packer.add(scene_data)
            # packer.add(gesture_data) # Ready for other modules

            # 3. Serialise and send accumulated payload
            packet = packer.pack_frame()
            if packet:
                streamer.send(packet)

                # 4. Debug network packages to terminal
                if config.DEBUG_MODE:
                    print(f"[NETWORK DEBUG] {packet.decode('utf-8')}")

            # 5. Preview and Debug overlay
            if config.SHOW_PREVIEW:
                if config.DEBUG_MODE:
                    person_tracker.draw_debug(frame, notes)

                    for person in scene_data.persons:
                        subframe = frame[person.y1:person.y2, person.x1:person.x2]
                        color = utils.player_color_from_id(person.id)
                        FaceTracker.draw_landmarks(subframe, person.face.landmarks, color)
                        PoseTracker.draw_landmarks(subframe, person.pose.landmarks, color)

                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
                    prev_time = curr_time
                    cv2.putText(
                        frame, f"FPS: {int(fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

                if config.SAVE_PREVIEW:
                    out.write(frame)

                cv2.imshow("Tracking Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        face_tracker.close()
        streamer.close()
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()