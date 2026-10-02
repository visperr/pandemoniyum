from dataclasses import dataclass, field
from typing import Dict, List
from models.base_data import BaseTrackingData
from models.person_data import PersonTrackingData

@dataclass
class SceneTrackingData(BaseTrackingData):
    """The epoch timestamp the frame was captured by the webcam."""
    capture_time: int = 0

    """Time required to process the frame in milliseconds."""
    processing_time: int = 0

    """Persons which are tracked in the scene."""
    persons: List[PersonTrackingData] = field(default_factory=list)

    @property
    def system_id(self) -> str:
        return "scene"

    @property
    def is_tracking(self) -> bool:
        return any(person.is_tracking for person in self.persons)

    def to_dict(self) -> dict:
        return {
            "capture_time": self.capture_time,
            "processing_time": self.processing_time,
            "persons": [person.to_dict() for person in self.persons]
        }
