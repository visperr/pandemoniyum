import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class HandTracker:
    def __init__(self, model_dir, num_hands):
        model_path = os.path.join(model_dir, 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, mp_image):
        return self.detector.detect(mp_image)

    def close(self):
        self.detector.close()