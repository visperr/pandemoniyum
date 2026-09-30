from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTrackingData(ABC):
    @property
    @abstractmethod
    def system_id(self) -> str:
        """Unique identifier key for this data type (e.g. 'face', 'gesture')."""
        pass

    @property
    @abstractmethod
    def is_tracking(self) -> bool:
        """Whether valid tracking was captured this frame."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialise this model's data into a dictionary payload."""
        pass