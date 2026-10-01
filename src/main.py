import cv2
import sys
import os
import time

import config
from core.data_packer import DataPacker
from core.network_streamer import UDPStreamer
from trackers.face_tracker import FaceTracker
from trackers.person_tracker import PersonTracker


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

    cap = open_video_capture()

    print("Tracking pipeline started. Press 'q' to exit.")

    prev_time = 0.0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            # 1. Process tracking components
            # face_data = face_tracker.process_frame(frame)
            scene_data = person_tracker.process_frame(frame)

            # 2. Add tracking models incrementally
            #packer.add(face_data)
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
                    # face_tracker.draw_debug(frame)
                    person_tracker.draw_debug(frame)

                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
                    prev_time = curr_time
                    cv2.putText(
                        frame, f"FPS: {int(fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

                cv2.imshow("Tracking Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        face_tracker.close()
        streamer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()