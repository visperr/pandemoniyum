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

        matrix = self.last_result.facial_transformation_matrixes[0].flatten().tolist()

        return FaceTrackingData(
            blendshapes=blendshapes,
            transformation_matrix=matrix,
            tracking_active=True
        )

    def draw_debug(self, frame: np.ndarray) -> None:
        """Render facial landmark points using pure OpenCV."""
        if not self.last_result or not self.last_result.face_landmarks:
            return

        h, w, _ = frame.shape
        for face_landmarks in self.last_result.face_landmarks:
            for lm in face_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

    def close(self) -> None:
        if self.model:
            self.model.close()