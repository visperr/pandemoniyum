import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class FaceTracker:
    def __init__(self, model_dir, num_faces):
        model_path = os.path.join(model_dir, 'face_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, mp_image):
        return self.detector.detect(mp_image)

    def close(self):
        self.detector.close()