import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class GestureTracker:
    def __init__(self, model_dir, num_hands):
        model_path = os.path.join(model_dir, 'gesture_recognizer.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=num_hands,
            # custom_gesture_classifier_options= FOR ALLOWING ONLY GESTURES WE WANT
        )
        self.detector = vision.GestureRecognizer.create_from_options(options)

    def process(self, mp_image):
        return self.detector.recognize(mp_image)

    def close(self):
        self.detector.close()