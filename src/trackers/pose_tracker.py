import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class PoseTracker:
    def __init__(self, model_dir, num_poses):
        model_path = os.path.join(model_dir, 'pose_landmarker_lite.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=num_poses,
            output_segmentation_masks=True
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, mp_image):
        return self.detector.detect(mp_image)

    def close(self):
        self.detector.close()