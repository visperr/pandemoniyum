import time
import cv2

PERSISTED_STATISTICS = {"Pose tracker", "Pose smoother", "Face tracker", "Person tracker"}

class PerformanceMonitor():

    def __init__(self):
        self.reset()

    def reset(self, preserve=False):
        if preserve:
            preserved = {key: self.times[key] for key in self.times.keys() & PERSISTED_STATISTICS}
            self.skipped += 1
        else:
            self.skipped = 1

        self.times = {}

        if preserve:
            self.times.update(preserved)

        self.title = None

    def start(self, preserve=False):
        self.reset(preserve)
        
        if preserve:
            self.total = time.time() - self.total
        else:
            self.total = time.time()

    def next(self, title):
        self.time = self.save()
        self.title = title

    def save(self):
        t = time.time()

        if self.title is not None:
            last_time = self.times[self.title] if self.title in self.times else 0

            self.times[self.title] = last_time + (t - self.time)

        return t

    def end(self):
        self.total = self.save() - self.total
        self.times = dict(sorted(self.times.items(), key=lambda x: x[0].lower()))

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
        self._draw_statistic(frame, "Frame time", f"{self.total/self.skipped * 1000:.1f} ms ({1/(self.total/self.skipped):.0f} fps) [1/{self.skipped}]", 0)

        for i, (name, time) in enumerate(self.times.items()):
            self._draw_statistic(frame, name, f"{time * 1000:3.1f} ms", 1+i)
