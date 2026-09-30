import cv2
import json
import argparse
import mediapipe as mp
import config
from network.udp_sender import UDPSender
from processors.data_packer import DataPacker
from trackers.face_tracker import FaceTracker
from trackers.pose_tracker import PoseTracker
from trackers.gesture_tracker import GestureTracker


def parse_args():
    parser = argparse.ArgumentParser(description="PANdeminiYUM Tracking Engine")
    parser.add_argument('-p', '--preview', action='store_true',
                        help="Show the OpenCV camera preview window with visualizer")
    return parser.parse_args()


def main(preview: bool = None):
    # If not explicitly passed in code, read from CLI arguments
    if preview is None:
        args = parse_args()
        preview = args.preview

    # Initialize Modules
    sender = UDPSender(config.UDP_IP, config.UDP_PORT)
    packer = DataPacker()

    face_tracker = FaceTracker(config.MODEL_DIR, config.NUM_FACES)
    pose_tracker = PoseTracker(config.MODEL_DIR, config.NUM_POSES)
    gesture_tracker = GestureTracker(config.MODEL_DIR, config.NUM_HANDS)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    print(f"Tracking Engine Live. Broadcasting to {config.UDP_IP}:{config.UDP_PORT}")
    if preview:
        print("Preview mode enabled. Press ESC in the video window to stop.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 1. Run Inference
        face_res = face_tracker.process(mp_image)
        pose_res = pose_tracker.process(mp_image)
        gesture_res = gesture_tracker.process(mp_image)

        # 2. Package Data
        payload = packer.package_frame(face_res, pose_res, gesture_res)

        # 3. Transmit
        sender.send(json.dumps(payload))

        # 4. Render Preview (Alleen als de flag is meegegeven)
        if preview:
            height, width, _ = image.shape

            # Teken Face Mesh (Groen)
            if face_res and face_res.face_landmarks:
                for face in face_res.face_landmarks:
                    for lm in face:
                        cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 1, (0, 255, 0), -1)

            # Teken Body Pose (Blauw)
            if pose_res and pose_res.pose_landmarks:
                for pose in pose_res.pose_landmarks:
                    for lm in pose:
                        cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 4, (255, 0, 0), -1)

            # Teken Handen/Gestures (Rood)
            if gesture_res and gesture_res.hand_landmarks:
                for hand in gesture_res.hand_landmarks:
                    for lm in hand:
                        cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 3, (0, 0, 255), -1)

            cv2.imshow('PANdemoniYUM Tracker', image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    face_tracker.close()
    pose_tracker.close()
    gesture_tracker.close()
    sender.close()


if __name__ == "__main__":
    main(preview=True)