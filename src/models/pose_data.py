from dataclasses import dataclass, field
from typing import List
from models.base_data import BaseTrackingData

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
            "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in self.landmarks],
            "is_tracking": self.tracking_active
        }