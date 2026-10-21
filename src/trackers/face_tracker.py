import base64
import math
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.base_tracker import BaseTracker
from models.face_data import FaceTrackingData

class FaceTracker(BaseTracker):
    def __init__(self, model_dir: str, num_faces: int = 1, min_tracking_confidence: float = 0.5, blendshapes=None):
        if blendshapes is None:
            self.blendshapes = []
        else:
            self.blendshapes = blendshapes
        self.num_faces = num_faces
        self.min_tracking_confidence = min_tracking_confidence
        self.last_result = None
        super().__init__(model_dir)

    def _initialize_model(self):
        model_path = os.path.join(self.model_dir, "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=self.num_faces,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE
        )
        return vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, cv2_frame: np.ndarray) -> FaceTrackingData:
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.last_result = self.model.detect(mp_image)

        if not self.last_result.face_landmarks:
            return FaceTrackingData(tracking_active=False)

        blendshapes = {
            category.category_name: category.score
            for category in self.last_result.face_blendshapes[0]
            if category.category_name in self.blendshapes
        }

        transform = self.last_result.facial_transformation_matrixes[0]
        matrix = transform.flatten().tolist()

        return FaceTrackingData(
            landmarks=self.last_result.face_landmarks,
            blendshapes=blendshapes,
            transformation_matrix=matrix,
            rotation=FaceTracker.unity_euler(transform),
            tracking_active=True
        )

    @staticmethod
    def aligned_face_image(face_frame: np.ndarray, landmarks, size: int = 128, max_bytes: int = 5000):
        """
        Cut the face out of `face_frame` (the crop the face model ran on), rotated so the eyes are level and
        scaled so the whole face fills a size x size image. The chef's head already copies the player's head
        tilt, so the picture must be upright. Returns (base64 JPEG, [mustache x, y, width] as image fractions),
        or ("", []) if there is no face.
        """
        if not landmarks:
            return "", []
        h, w = face_frame.shape[:2]
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks[0]], dtype=np.float32)

        # MediaPipe face mesh indices: eye outer corners, forehead top, chin, cheeks, nose base, upper lip, mouth corners
        eye_a, eye_b = pts[33], pts[263]
        top, chin = pts[10], pts[152]
        cheek_a, cheek_b = pts[234], pts[454]
        nose_base, upper_lip = pts[2], pts[0]
        mouth_a, mouth_b = pts[61], pts[291]

        angle = math.degrees(math.atan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))
        center = (top + chin) / 2.0
        extent = max(np.linalg.norm(chin - top), np.linalg.norm(cheek_b - cheek_a)) * 1.2
        scale = size / max(extent, 1.0)

        m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, scale)
        m[:, 2] += np.array([size / 2.0, size / 2.0]) - center
        face = cv2.warpAffine(face_frame, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        def warp(p):
            return m @ np.array([p[0], p[1], 1.0])

        # Mustache: between the nose base and the upper lip, a bit wider than the mouth
        stache = warp(nose_base + (upper_lip - nose_base) * 0.45)
        stache_width = np.linalg.norm(warp(mouth_b) - warp(mouth_a)) * 1.9

        for quality in (75, 60, 45, 30):
            ok, buffer = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and len(buffer) <= max_bytes:
                break
        if not ok:
            return "", []
        mustache = [round(float(stache[0] / size), 4), round(float(stache[1] / size), 4), round(float(stache_width / size), 4)]
        return base64.b64encode(buffer).decode("ascii"), mustache

    @staticmethod
    def unity_euler(transform: np.ndarray) -> dict:
        """
        Head rotation as Unity Euler angles (degrees), ready for Quaternion.Euler(pitch, yaw, roll).
        MediaPipe's face space is right-handed with +Z towards the camera; Unity is left-handed and
        looks along +Z, so flipping Z (S R S) gives the same rotation as seen on screen.
        """
        r = np.asarray(transform)[:3, :3]
        s = np.diag([1.0, 1.0, -1.0])
        r = s @ r @ s
        # Unity applies Z, then X, then Y: R = Ry(yaw) Rx(pitch) Rz(roll)
        pitch = math.degrees(math.asin(float(np.clip(-r[1, 2], -1.0, 1.0))))
        yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
        roll = math.degrees(math.atan2(r[1, 0], r[1, 1]))
        return {"pitch": round(pitch, 2), "yaw": round(yaw, 2), "roll": round(roll, 2)}

    def draw_debug(self, frame: np.ndarray) -> None:
        """Render facial landmark points using pure OpenCV."""
        if not self.last_result or not self.last_result.face_landmarks:
            return

        FaceTracker.draw_landmarks(frame, self.last_result.face_landmarks)

    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks, color=(0, 255, 0)) -> None:
        if landmarks is None:
            return
        
        h, w, _ = frame.shape
        for face_landmarks in landmarks:
            for lm in face_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, color, -1)

    def close(self) -> None:
        if self.model:
            self.model.close()