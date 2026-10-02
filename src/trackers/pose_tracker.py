import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import utils
from core.base_tracker import BaseTracker
from models.pose_data import PoseTrackingData

class PoseTracker(BaseTracker):
    def __init__(self, model_dir: str, num_poses: int = 1, min_tracking_confidence: float = 0.5):
        self.num_poses = num_poses
        self.min_tracking_confidence = min_tracking_confidence
        self.last_result = None
        super().__init__(model_dir)

    def _initialize_model(self):
        model_path = os.path.join(self.model_dir, "pose_landmarker_lite.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_tracking_confidence=self.min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        return vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, cv2_frame: np.ndarray) -> PoseTrackingData:
        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.last_result = self.model.detect(mp_image)

        if not self.last_result.pose_landmarks or not self.last_result.pose_landmarks[0]:
            return PoseTrackingData(tracking_active=False)

        return PoseTrackingData(
            landmarks=self.last_result.pose_landmarks[0],
            tracking_active=True
        )

    def draw_debug(self, frame: np.ndarray) -> None:
        """Render facial landmark points using pure OpenCV."""
        if not self.last_result or not self.last_result.pose_landmarks or not self.last_result.pose_landmarks[0]:
            return

        PoseTracker.draw_landmarks(frame, self.last_result.pose_landmarks[0])

    @staticmethod
    def draw_landmarks(frame: np.ndarray, landmarks, color=(0, 255, 0)) -> None:
        if landmarks is None or len(landmarks) < 33:
            return
        
        h, w, _ = frame.shape
        i = -1
        for lm in landmarks:
            i += 1

            # Do not draw face landmarks because we're already using FaceTracker
            # https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
            if i <= 10:
                continue

            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, color, -1)

        connected = [
            (11, 12), (11, 13), (13, 15), (15, 21), (15, 19), (15, 17), (17, 19), (12, 14),
            (14, 16), (16, 22), (16, 18), (16, 20), (18, 20), (11, 23), (12, 24), (23, 24),
            (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), (29, 31), (28, 30),
            (28, 32), (30, 32)
        ]

        for (l1, l2) in connected:
            cv2.line(frame, utils.landmark_to_point(landmarks[l1], w, h), utils.landmark_to_point(landmarks[l2], w, h), color, 2)

    def close(self) -> None:
        if self.model:
            self.model.close()