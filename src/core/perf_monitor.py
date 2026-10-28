import time
from collections import deque

import cv2


class PerformanceMonitor:
    """
    Per-stage timings plus a frame time / FPS figure for the debug overlay.

    Some stages do not run on every frame (the person detector runs every PERSON_DETECTION_FRAME_SKIP
    frames, pose/face every LANDMARK_FRAME_SKIP frames, ...). Two things keep the numbers honest:

    - Frame time / FPS is averaged over a rolling window of recent frames, so cheap (skipped) and
      expensive (full) frames even out instead of the FPS jumping up on every cheap frame.
    - A stage's time is the time of its last *full* run. On frames where a stage did not run
      (it never gets a next() call, or it was flagged with keep_previous()) the old value stays.
    """

    def __init__(self, window: int = 30, stale_frames: int = 90):
        self.window = deque(maxlen=window)  # total time of the most recent frames
        self.stale_frames = stale_frames    # forget a stage that has not fully run for this many frames
        self.frame_index = 0
        self.stages = {}                    # stage name -> (seconds of last full run, frame index of that run)
        self.reset()

    def reset(self):
        self.times = {}          # stage -> seconds spent on the current frame
        self.partial = set()     # stages that only ran partially on the current frame
        self.title = None
        self._mark = None
        self._frame_start = None

    def start(self):
        self.reset()
        self._frame_start = self._mark = time.perf_counter()

    def next(self, title):
        """Close the running stage and start timing `title`."""
        self._close_stage()
        self.title = title

    def keep_previous(self, title):
        """The stage `title` did not really run this frame: keep showing the time of its last full run."""
        self.partial.add(title)

    def _close_stage(self):
        now = time.perf_counter()
        if self.title is not None:
            self.times[self.title] = self.times.get(self.title, 0.0) + (now - self._mark)
        self._mark = now
        return now

    def end(self):
        now = self._close_stage()
        self.title = None

        self.window.append(now - self._frame_start)
        self.frame_index += 1

        for title, seconds in self.times.items():
            if title in self.partial and title in self.stages:
                continue  # skipped this frame: retain the previous full-run time
            self.stages[title] = (seconds, self.frame_index)

        self.stages = {name: entry for name, entry in self.stages.items()
                       if self.frame_index - entry[1] <= self.stale_frames}

    def _draw_statistic(self, frame, name, value, pos):
        cv2.putText(
            frame, f"{name}:", (20, 40+40*pos),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame, value, (260, 40+40*pos),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    def draw(self, frame):
        if self.window:
            average = sum(self.window) / len(self.window)
            fps = 1 / average if average > 0 else 0
            self._draw_statistic(frame, "Frame time", f"{average * 1000:.1f} ms ({fps:.0f} fps)", 0)

        for i, (name, (seconds, _)) in enumerate(sorted(self.stages.items(), key=lambda x: x[0].lower())):
            self._draw_statistic(frame, name, f"{seconds * 1000:3.1f} ms", 1+i)
