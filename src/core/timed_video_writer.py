"""
Writes frames to a video that keeps real time, for lining recordings up with other recordings (play_session.py
stitches the tracker's view onto the recorded gameplay).

Frames are handed in with the wall-clock time they show (when the camera captured them). The video runs at a
fixed frame rate starting at the first frame's time: frames are repeated to fill gaps when tracking slows down
and skipped when two land in the same slot, so a moment that happened t seconds after the first frame is t
seconds into the video. The first frame's time is written to a JSON file next to the video.

Encoding runs in an ffmpeg process (macOS hardware encoder) fed from a background thread, so writing barely
slows the tracking loop. Frames are dropped rather than queued without limit if encoding falls behind.
"""

import json
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

import numpy as np


def sidecar_path(video_path: Path) -> Path:
    """Where a recording's timing (first frame time, frame rate, size) is stored."""
    return video_path.with_suffix(video_path.suffix + ".json")


class TimedVideoWriter:
    def __init__(self, path: str, fps: float = 30.0, bitrate: str = "4M"):
        self.path = Path(path)
        self.fps = fps
        self.bitrate = bitrate
        self.first_frame_time: Optional[float] = None  # epoch seconds
        self.dropped = 0
        self._ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        self._process: Optional[subprocess.Popen] = None
        self._queue = queue.Queue(maxsize=32)
        self._thread = threading.Thread(target=self._writer, name="TimedVideoWriter", daemon=True)
        self._thread.start()

    def add(self, frame: np.ndarray, epoch_time: float) -> None:
        """Queue a BGR frame showing the moment epoch_time (seconds since 1970)."""
        try:
            self._queue.put_nowait((frame, epoch_time))
        except queue.Full:
            self.dropped += 1

    def _start(self, width: int, height: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [self._ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
             "-c:v", "h264_videotoolbox", "-b:v", self.bitrate, "-pix_fmt", "yuv420p", str(self.path)],
            stdin=subprocess.PIPE)
        with open(sidecar_path(self.path), "w") as f:
            json.dump({"first_frame_time": self.first_frame_time, "fps": self.fps,
                       "width": width, "height": height}, f, indent=2)

    def _writer(self) -> None:
        written = 0
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame, t = item
            if self._process is None:
                self.first_frame_time = t
                self._start(frame.shape[1], frame.shape[0])
            slot = round((t - self.first_frame_time) * self.fps)
            if slot < written:
                continue  # this slot already shows a frame
            data = np.ascontiguousarray(frame).tobytes()
            try:
                while written <= slot:
                    self._process.stdin.write(data)
                    written += 1
            except BrokenPipeError:
                break

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._process is not None:
            self._process.stdin.close()
            self._process.wait(timeout=30)
            print(f"Tracking video saved: {self.path}" + (f" ({self.dropped} frames dropped)" if self.dropped else ""))
