"""
Loading recorded sessions (see core/session_recorder.py) into arrays for the metrics.

All positions are in pixels of the mirrored frame the tracker processed (the image the player sees),
so annotations, raw poses and sent poses can be compared directly.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.session_recorder import FRAMES_FILE, META_FILE, VIDEO_FILE

# The evaluated keypoints: name -> MediaPipe pose landmark index. "left"/"right" are the player's own sides.
KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
}
# The wrists place the chef's hands and drive grabbing; they matter most for the game
CRITICAL_KEYPOINTS = ("left_wrist", "right_wrist")
L_SHOULDER, R_SHOULDER = 11, 12

# MediaPipe's own threshold for "this landmark is visible"
VISIBLE = 0.5

KEYPOINT_ANNOTATIONS = "annotations_keypoints.json"
EVENT_ANNOTATIONS = "annotations_events.json"


@dataclass
class Track:
    """One person's landmarks over a session. Arrays have one row per frame of the session."""
    i: np.ndarray          # (F,) frame numbers
    t: np.ndarray          # (F,) frame times (s)
    xy: np.ndarray         # (F, 25, 2) pixels, NaN where there is no pose
    vis: np.ndarray        # (F, 25) visibility, NaN where there is no pose
    present: np.ndarray    # (F,) bool, a pose exists on this frame

    def keypoint(self, name: str):
        j = KEYPOINTS[name]
        return self.xy[:, j], self.vis[:, j]

    def shoulder_width(self) -> float:
        """Median shoulder width in pixels over frames where both shoulders are clearly seen."""
        ok = (self.vis[:, L_SHOULDER] >= 0.8) & (self.vis[:, R_SHOULDER] >= 0.8)
        if not ok.any():
            ok = self.present
        if not ok.any():
            return float("nan")
        return float(np.median(np.linalg.norm(self.xy[ok, L_SHOULDER] - self.xy[ok, R_SHOULDER], axis=1)))


class Session:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path / META_FILE) as f:
            self.meta = json.load(f)
        with open(self.path / FRAMES_FILE) as f:
            self.frames: List[dict] = [json.loads(line) for line in f if line.strip()]
        if not self.frames:
            raise ValueError(f"{self.path} has no frames")
        self.index = {record["i"]: k for k, record in enumerate(self.frames)}

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def condition(self) -> str:
        return self.meta.get("condition", "")

    def factor(self, name: str):
        """A recording factor (subject, distance_m, lux, shoulder_width_mm); a replay inherits its source's."""
        value = self.meta.get("factors", {}).get(name)
        source = self.source_path()
        if value is None and source is not None and (source / META_FILE).exists():
            with open(source / META_FILE) as f:
                value = json.load(f).get("factors", {}).get(name)
        return value

    def config(self, name: str, default=None):
        return self.meta.get("config", {}).get(name, default)

    def column(self, key: str) -> np.ndarray:
        return np.array([record.get(key, np.nan) if record.get(key) is not None else np.nan
                         for record in self.frames], dtype=float)

    @property
    def is_live(self) -> bool:
        """Recorded from the camera, so its timestamps are real-time latencies."""
        return self.meta.get("source") == "camera"

    def video_path(self) -> Optional[Path]:
        """The raw video of this session or, for a replay, of the session it replayed."""
        for folder in (self.path, self.source_path()):
            if folder is not None and (folder / VIDEO_FILE).exists():
                return folder / VIDEO_FILE
        return None

    def source_path(self) -> Optional[Path]:
        source = self.meta.get("source_session")
        return Path(source) if source else None

    def annotations(self, filename: str) -> Optional[dict]:
        """Annotations of this session; a replay uses those of its source session (same frame numbers)."""
        for folder in (self.path, self.source_path()):
            if folder is not None and (folder / filename).exists():
                with open(folder / filename) as f:
                    return json.load(f)
        return None

    def person_ids(self) -> Counter:
        """How many frames each person id has a sent pose."""
        counts = Counter()
        for record in self.frames:
            for person in record["persons"]:
                if person.get("pose"):
                    counts[person["id"]] += 1
        return counts

    def main_person(self) -> Optional[int]:
        counts = self.person_ids()
        return counts.most_common(1)[0][0] if counts else None

    def person(self, record: dict, pid: int) -> Optional[dict]:
        for person in record["persons"]:
            if person["id"] == pid:
                return person
        return None

    def track(self, pid: int, kind: str = "pose") -> Track:
        """
        kind="pose": the pose that was sent to Unity (smoothed, missing joints inferred).
        kind="raw":  the pose model's output that went into the smoother (only on frames the model ran).
        """
        n = len(self.frames)
        xy = np.full((n, 25, 2), np.nan)
        vis = np.full((n, 25), np.nan)
        present = np.zeros(n, dtype=bool)
        for k, record in enumerate(self.frames):
            person = self.person(record, pid)
            if person is None or not person.get(kind) or not person.get("crop"):
                continue
            rows = np.array(person[kind], dtype=float)
            x1, y1, x2, y2 = person["crop"]
            xy[k, :, 0] = x1 + rows[:25, 0] * (x2 - x1)
            xy[k, :, 1] = y1 + rows[:25, 1] * (y2 - y1)
            v = rows[:25, 3]
            if kind == "raw" and rows.shape[1] > 4:
                v = np.minimum(v, rows[:25, 4])  # the smoother trusts min(visibility, presence)
            vis[k] = v
            present[k] = True
        return Track(self.column("i").astype(int), self.column("t"), xy, vis, present)

    def hands(self, pid: int, hand: str) -> Dict[str, np.ndarray]:
        """The sent state of one hand ("hand15" = player's left, "hand16" = right) per frame."""
        closed, seen, prob, gesture = [], [], [], []
        for record in self.frames:
            person = self.person(record, pid)
            state = (person or {}).get("hands", {}).get(hand) or {}
            closed.append(bool(state.get("closed", False)))
            seen.append(bool(state.get("seen", False)))
            prob.append(float(state.get("closed_prob", np.nan)) if state else np.nan)
            gesture.append(state.get("gesture", "None"))
        return {"closed": np.array(closed), "seen": np.array(seen), "closed_prob": np.array(prob),
                "gesture": np.array(gesture)}


HAND_KEYS = {"left": "hand15", "right": "hand16"}
WRIST_OF_HAND = {"left": "left_wrist", "right": "right_wrist"}


def segments(mask: np.ndarray, min_length: int = 1) -> List[tuple]:
    """(start, end) index ranges (end exclusive) of the runs of True in mask."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_length]
