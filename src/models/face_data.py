from dataclasses import dataclass, field
from typing import Dict, List
from models.base_data import BaseTrackingData

@dataclass
class FaceTrackingData(BaseTrackingData):
    landmarks: any = None
    blendshapes: Dict[str, float] = field(default_factory=dict)
    transformation_matrix: List[float] = field(default_factory=list)
    tracking_active: bool = False

    @property
    def system_id(self) -> str:
        return "face"

    @property
    def is_tracking(self) -> bool:
        return self.tracking_active

    def to_dict(self) -> dict:
        return {
            "blendshapes": self.blendshapes,
            "matrix": self.transformation_matrix,
            "is_tracking": self.tracking_active
        }