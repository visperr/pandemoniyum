from dataclasses import dataclass, field
from typing import Optional, Tuple
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

    # Region (x1, y1, x2, y2, frame pixels) the pose landmarks are normalised to: the padded, clamped person box
    crop: Optional[Tuple[int, int, int, int]] = None
    # Open/closed state per wrist, keyed "hand15"/"hand16" (see HandGestureTracker.hands_dict)
    hands: dict = field(default_factory=dict)

    occluded: bool = False
    # True while the person was not detected recently and the last known box is being held
    frozen: bool = False
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
            "frozen": self.frozen,
            "face": self.face.to_dict(),
            "pose": self.pose.to_dict(),
            "crop": list(self.crop) if self.crop else [],
            "hands": self.hands,
            "is_tracking": self.tracking_active
        }
