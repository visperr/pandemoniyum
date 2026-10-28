"""
Recording and replaying tracking sessions for offline evaluation (see EVALUATION.md and src/evaluation/).

A session is a folder with:
- meta.json     what was recorded: condition label, notes, video source, lighting simulation, and the config
                values in effect (the metrics replay the smoother with them)
- frames.jsonl  one line per processed frame: timestamps, stage timings, frame brightness and, per person,
                the box, the pose that went into the smoother ("raw"), the pose that was sent ("pose") and
                the hand states
- raw.mp4       the camera frames as the tracker received them (before mirroring), for annotation and for
                replaying the session through the pipeline (main.py --replay)

Files are written on a background thread so recording barely changes the timings it measures.
"""

import json
import queue
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

FRAMES_FILE = "frames.jsonl"
META_FILE = "meta.json"
VIDEO_FILE = "raw.mp4"


def pose_rows(pose, with_presence: bool = False) -> Optional[list]:
    """Crop-normalised landmarks as [x, y, z, visibility(, presence)] rows, or None without a pose."""
    if pose is None or not pose.is_tracking or not pose.landmarks:
        return None
    rows = []
    for lm in pose.landmarks:
        vis = getattr(lm, "visibility", None)
        row = [round(float(lm.x), 5), round(float(lm.y), 5), round(float(lm.z), 5),
               round(1.0 if vis is None else float(vis), 3)]
        if with_presence:
            pres = getattr(lm, "presence", None)
            row.append(round(1.0 if pres is None else float(pres), 3))
        rows.append(row)
    return rows


def frame_luma(frame: np.ndarray) -> float:
    """Mean brightness (0-255) of a subsampled grey version of the frame."""
    return float(cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY).mean())


def simulate_lighting(frame: np.ndarray, brightness: float = 1.0, gamma: float = 1.0,
                      noise: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Darken/brighten a frame to test lighting robustness on a fixed recording.
    brightness scales the intensity, gamma > 1 crushes the shadows, noise adds Gaussian sensor noise
    (standard deviation in 0-255 units), as a dim scene does once the camera raises its gain.
    """
    if brightness == 1.0 and gamma == 1.0 and noise <= 0.0:
        return frame
    out = frame.astype(np.float32) / 255.0
    if gamma != 1.0:
        out = np.power(out, gamma)
    out *= brightness * 255.0
    if noise > 0.0:
        out += (rng or np.random.default_rng()).normal(0.0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "session"


class SessionRecorder:
    def __init__(self, root: str, condition: str, meta: dict, record_video: bool = True, fps: float = 30.0):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(root) / f"{stamp}_{_slug(condition)}"
        self.path.mkdir(parents=True, exist_ok=False)

        self.meta = {"created": datetime.now().isoformat(timespec="seconds"), "condition": condition, **meta,
                     "video": VIDEO_FILE if record_video else None}
        self._write_meta()

        self._record_video = record_video
        self._fps = fps
        self._video = None
        self._frames = 0
        self._lines = open(self.path / FRAMES_FILE, "w")
        # Bounded so a stalled disk cannot eat all memory; ~8 s of frames at 30 fps
        self._queue = queue.Queue(maxsize=256)
        self._thread = threading.Thread(target=self._writer, name="SessionRecorder", daemon=True)
        self._thread.start()
        print(f"Recording session to: {self.path}")

    def add(self, record: dict, raw_frame: Optional[np.ndarray] = None) -> None:
        """Queue one frame's record (and the raw camera frame). `record["video_frame"]` is set here."""
        if self._record_video and raw_frame is not None:
            record["video_frame"] = self._frames
            self._frames += 1
        self._queue.put((record, raw_frame if self._record_video else None))

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            record, frame = item
            if frame is not None:
                if self._video is None:
                    h, w = frame.shape[:2]
                    self._video = cv2.VideoWriter(str(self.path / VIDEO_FILE), cv2.VideoWriter_fourcc(*"mp4v"),
                                                  self._fps, (w, h))
                self._video.write(frame)
            self._lines.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _write_meta(self) -> None:
        with open(self.path / META_FILE, "w") as f:
            json.dump(self.meta, f, indent=2)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._lines.close()
        if self._video is not None:
            self._video.release()
        self.meta["video_frames"] = self._frames
        self._write_meta()
        print(f"Session saved: {self.path}")


class SessionReplayCapture:
    """
    Plays a recorded session's raw video back through the pipeline with the frame times that were logged
    live, so the time-based filters behave as they did during the recording. Drop-in for cv2.VideoCapture.
    """

    def __init__(self, session_dir: str):
        session = Path(session_dir)
        self._cap = cv2.VideoCapture(str(session / VIDEO_FILE))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"No {VIDEO_FILE} in {session}")
        times = {}
        with open(session / FRAMES_FILE) as f:
            for line in f:
                record = json.loads(line)
                if "video_frame" in record:
                    times[record["video_frame"]] = record["t"]
        self._times = [times[k] for k in sorted(times)]
        self._index = -1

    def read(self):
        if self._index + 1 >= len(self._times):
            return False, None
        success, frame = self._cap.read()
        if success:
            self._index += 1
        return success, frame

    def isOpened(self) -> bool:
        return self._cap.isOpened() and self._index + 1 < len(self._times)

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return self._times[max(self._index, 0)] * 1000.0
        return self._cap.get(prop_id)

    def release(self) -> None:
        self._cap.release()
