from dataclasses import dataclass, field
from models.base_data import BaseTrackingData
from models.face_data import FaceTrackingData
from models.pose_data import PoseTrackingData

@dataclass
class PersonTrackingData(BaseTrackingData):
    id: int = 0

    x1: int = 0
    x2: int = 0
    y1: int = 0
    y2: int = 0

    face: FaceTrackingData = field(default_factory=FaceTrackingData)
    pose: PoseTrackingData = field(default_factory=PoseTrackingData)

    occluded: bool = False
    tracking_active: bool = False

    @property
    def system_id(self) -> str:
        return "person"

    @property
    def is_occluded(self) -> bool:
        return self.occluded
    
    @property
    def is_tracking(self) -> bool:
        return self.tracking_active

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x1": self.x1,
            "x2": self.x2,
            "y1": self.y1,
            "y2": self.y2,
            "occluded": self.occluded,
            "face": self.face.to_dict(),
            "pose": self.pose.to_dict(),
            "is_tracking": self.tracking_active
        }
