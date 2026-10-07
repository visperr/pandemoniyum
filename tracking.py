import cv2
import socket
import json
import math
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup UDP Socket
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2. Configure Models
face_detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
    output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=2))

pose_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'), num_poses=2))

hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'), num_hands=4))

# Helper to calculate grab score
def calculate_grab_score(landmarks):
    palm_size = math.hypot(landmarks[0].x - landmarks[9].x, landmarks[0].y - landmarks[9].y)
    finger_dist = math.hypot(landmarks[0].x - landmarks[12].x, landmarks[0].y - landmarks[12].y)
    ratio = finger_dist / (palm_size + 0.0001)
    grab = (2.0 - ratio) / 1.2
    return max(0.0, min(1.0, grab))

# 3. Source selection (webcam or video file)

use_webcam = False

if use_webcam:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
else:
    cap = cv2.VideoCapture("WIN_20260923_15_21_55_Pro.mp4")

print(f"Broadcasting to {UDP_IP}:{UDP_PORT}... Press ESC in the video window to stop.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run Inferences
    face_result = face_detector.detect(mp_image)
    pose_result = pose_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)

    payload = {
        "blendshapes": {}, "matrix": [], "pose": [], 
        "hands": {"Left": 0.0, "Right": 0.0}
    }
    
    debug_hud = [] # Stores text to draw on the video frame

    # --- PROCESS FACE (GREEN) ---
    if face_result.face_landmarks:
        payload["blendshapes"] = {c.category_name: c.score for c in face_result.face_blendshapes[0]}
        payload["matrix"] = face_result.facial_transformation_matrixes[0].flatten().tolist()
        
        for lm in face_result.face_landmarks[0]:
            cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 1, (0, 255, 0), -1)
            
        jaw_val = payload["blendshapes"].get("jawOpen", 0.0)
        debug_hud.append(f"FACE : Active (Jaw: {jaw_val:.2f})")
    else:
        debug_hud.append("FACE : None")

    # --- PROCESS POSE (BLUE) ---
    if pose_result.pose_landmarks:
        for person in pose_result.pose_landmarks:
            #mp.solutions.drawing_utils.draw_landmarks()
            for lm in person:
                payload["pose"].append({"x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility})
                cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 4, (255, 0, 0), -1)
        debug_hud.append(f"POSE : {len(pose_result.pose_landmarks)}")
    else:
        debug_hud.append("POSE : None")

    # --- PROCESS HANDS (RED) ---
    if hand_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            true_hand = "Right" if handedness == "Left" else "Left"
            payload["hands"][true_hand] = calculate_grab_score(hand_landmarks)

            for lm in hand_landmarks:
                cv2.circle(image, (int(lm.x * width), int(lm.y * height)), 3, (0, 0, 255), -1)
                
    l_grab = payload["hands"]["Left"]
    r_grab = payload["hands"]["Right"]
    debug_hud.append(f"HANDS: L:{l_grab:.2f} | R:{r_grab:.2f}")

    # --- RENDER DEBUG OVERLAYS ---
    # 1. Draw HUD on the video frame
    y_offset = 40
    for text in debug_hud:
        # Draw a black outline for readability, then the yellow text
        cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 35

    # 2. Print a clean, updating line in the terminal
    console_out = " || ".join(debug_hud)
    sys.stdout.write(f"\r{console_out: <80}")
    sys.stdout.flush()

    # --- SEND NETWORK DATA ---
    sock.sendto(json.dumps(payload).encode('utf-8'), (UDP_IP, UDP_PORT))

    cv2.imshow('MediaPipe Full Body Tracker', image)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
print("\nTracker stopped.")