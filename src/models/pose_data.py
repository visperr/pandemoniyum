from dataclasses import dataclass, field
from typing import List
from models.base_data import BaseTrackingData


# MediaPipe returns 33 landmarks. Only 0-24 (face, arms, torso, hips) are used by the game;
# everything below the hips (25-32: knees, ankles, heels, foot points) is dropped at the source.
POSE_LANDMARK_COUNT = 25


@dataclass
class PoseLandmark:
    """
    Lightweight landmark with the same attributes as MediaPipe's NormalizedLandmark,
    so the debug drawing code and serialisation work with raw and smoothed landmarks alike.

    x/y are normalised to the person crop, z uses the same scale as x.
    visibility is the confidence that the position is a real measurement
    (0.0 = the joint is off-screen/occluded and its position was inferred).
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


def _visibility(lm) -> float:
    value = getattr(lm, "visibility", None)
    return 1.0 if value is None else float(value)


@dataclass
class PoseTrackingData(BaseTrackingData):
    landmarks: List[tuple] = field(default_factory=list)
    tracking_active: bool = False

    @property
    def system_id(self) -> str:
        return "pose"

    @property
    def is_tracking(self) -> bool:
        return self.tracking_active

    def to_dict(self) -> dict:
        return {
            "landmarks": [
                {
                    "x": round(float(lm.x), 5),
                    "y": round(float(lm.y), 5),
                    "z": round(float(lm.z), 5),
                    "visibility": round(_visibility(lm), 3),
                }
                for lm in self.landmarks
            ],
            "is_tracking": self.tracking_active
        }
