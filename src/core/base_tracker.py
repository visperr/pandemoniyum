from abc import ABC, abstractmethod
from typing import Any

class BaseTracker(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        """Initialise and return the specific ML model."""
        pass

    @abstractmethod
    def process_frame(self, frame: Any) -> Any:
        """Process a frame and return a specific data model instance."""
        pass