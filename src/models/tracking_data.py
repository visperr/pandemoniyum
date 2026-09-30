from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time

@dataclass(slots=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "z": round(self.z, 4)}


@dataclass(slots=True)
class Landmark(Vector3):
    visibility: float = 1.0

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "v": round(self.visibility, 3)
        }


@dataclass(slots=True)
class HandData:
    handedness: str                     # "Left" or "Right"
    gesture: str = "None"
    gesture_score: float = 0.0
    grab_score: float = 0.0             # Continuous grab: 0.0 (open) to 1.0 (closed)
    landmarks: List[Landmark] = field(default_factory=list)

    @property
    def wrist(self) -> Optional[Landmark]:
        return self.landmarks[0] if self.landmarks else None

    def to_dict(self) -> dict:
        return {
            "gesture": self.gesture,
            "gesture_score": round(self.gesture_score, 3),
            "grab_score": round(self.grab_score, 3),
            "landmarks": [lm.to_dict() for lm in self.landmarks]
        }


@dataclass(slots=True)
class FaceData:
    blendshapes: Dict[str, float] = field(default_factory=dict)
    head_matrix: List[float] = field(default_factory=list) # 16-element affine matrix

    @property
    def is_detected(self) -> bool:
        return len(self.blendshapes) > 0

    def to_dict(self) -> dict:
        return {
            "blendshapes": {k: round(v, 4) for k, v in self.blendshapes.items()},
            "head_matrix": [round(val, 5) for val in self.head_matrix]
        }


@dataclass(slots=True)
class PoseData:
    landmarks: List[Landmark] = field(default_factory=list)

    @property
    def is_detected(self) -> bool:
        return len(self.landmarks) > 0

    # Key landmark helpers based on standard MediaPipe Pose indexing
    @property
    def nose(self) -> Optional[Landmark]:
        return self.landmarks[0] if len(self.landmarks) > 0 else None

    @property
    def left_wrist(self) -> Optional[Landmark]:
        return self.landmarks[15] if len(self.landmarks) > 15 else None

    @property
    def right_wrist(self) -> Optional[Landmark]:
        return self.landmarks[16] if len(self.landmarks) > 16 else None

    @property
    def centre_x(self) -> float:
        """Returns approximate horizontal centre based on mid-hip or nose."""
        if len(self.landmarks) > 24:
            return (self.landmarks[23].x + self.landmarks[24].x) * 0.5
        elif self.nose:
            return self.nose.x
        return 0.5

    def to_dict(self) -> list:
        return [lm.to_dict() for lm in self.landmarks]


@dataclass(slots=True)
class PlayerData:
    player_id: int
    is_tracked: bool = False
    face: FaceData = field(default_factory=FaceData)
    pose: PoseData = field(default_factory=PoseData)
    hands: Dict[str, Optional[HandData]] = field(
        default_factory=lambda: {"Left": None, "Right": None}
    )

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "is_tracked": self.is_tracked,
            "face": self.face.to_dict(),
            "pose": self.pose.to_dict(),
            "hands": {
                side: hand.to_dict() if hand else None
                for side, hand in self.hands.items()
            }
        }


@dataclass(slots=True)
class TrackingFrame:
    timestamp: float = field(default_factory=time.time)
    frame_index: int = 0
    players: List[PlayerData] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "players": [p.to_dict() for p in self.players]
        }