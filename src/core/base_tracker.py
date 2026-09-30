from abc import ABC, abstractmethod
from typing import Any
from models.base_data import BaseTrackingData

class BaseTracker(ABC):
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = self._initialize_model()

    @abstractmethod
    def _initialize_model(self) -> Any:
        """Initialise and return the vision/ML model instance."""
        pass

    @abstractmethod
    def process_frame(self, frame: Any) -> BaseTrackingData:
        """Process a frame and return an instance of BaseTrackingData."""
        pass

    @abstractmethod
    def draw_debug(self, frame: Any) -> None:
        """Draw visual tracking markers directly on the frame."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying detector and memory allocations."""
        pass