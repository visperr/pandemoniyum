from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class FaceTrackingData:
    landmarks: List[Tuple[float, float, float]]
    confidence: float
    bounding_box: Tuple[float, float, float, float]