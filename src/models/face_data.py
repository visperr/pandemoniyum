from dataclasses import dataclass, field
from typing import Dict, List
from models.base_data import BaseTrackingData

@dataclass
class FaceTrackingData(BaseTrackingData):
    landmarks: any = None
    blendshapes: Dict[str, float] = field(default_factory=dict)
    transformation_matrix: List[float] = field(default_factory=list)
    # Head rotation as Unity Euler angles in degrees: {"pitch", "yaw", "roll"}
    rotation: Dict[str, float] = field(default_factory=dict)
    # Upright, face-centred JPEG of the player's face (base64) and where the mustache goes on it:
    # [centre x, centre y, width], as fractions of the image (see FaceTracker.aligned_face_image)
    image: str = ""
    mustache: List[float] = field(default_factory=list)
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
            "rotation": self.rotation,
            "image": self.image,
            "mustache": self.mustache,
            "is_tracking": self.tracking_active
        }