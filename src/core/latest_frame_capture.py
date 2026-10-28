import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class LatestFrameCapture:
    """
    Wraps a webcam cv2.VideoCapture and reads it on a background thread, keeping only the newest frame.

    OpenCV queues camera frames internally; when tracking runs slower than the camera, cap.read()
    returns ever older frames and the delay grows. Here read() always returns the most recent frame
    and stale ones are dropped.

    capture_time is the time.perf_counter() at which the frame last returned by read() arrived from the
    camera, and dropped counts the frames that were replaced before anyone read them.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._fresh = False
        self._stamp = 0.0
        self.capture_time = 0.0
        self.dropped = 0
        self._running = cap.isOpened()
        self._thread = threading.Thread(target=self._reader, name="CameraReader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            success, frame = self._cap.read()
            stamp = time.perf_counter()
            with self._cond:
                if not success:
                    self._running = False
                else:
                    if self._fresh:
                        self.dropped += 1
                    self._frame = frame
                    self._stamp = stamp
                    self._fresh = True
                self._cond.notify_all()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Block until a frame newer than the last one returned is available."""
        with self._cond:
            while not self._fresh and self._running:
                self._cond.wait(timeout=1.0)
            if not self._fresh:
                return False, None
            self._fresh = False
            self.capture_time = self._stamp
            return True, self._frame

    def isOpened(self) -> bool:
        return self._running

    def get(self, prop_id: int) -> float:
        return self._cap.get(prop_id)

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
