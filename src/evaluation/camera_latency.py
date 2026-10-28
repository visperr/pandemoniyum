"""
Camera ("glass-to-frame") latency: how long after something happens in front of the camera its frame reaches
the tracker. The tracker's own timestamps start when a frame arrives, so this part is measured separately
and added to the end-to-end delays (report.py --camera-latency-ms).

    python -m evaluation.camera_latency [--camera 0] [--trials 30]

A window flips between black and white; point the webcam at it (for a laptop's built-in camera: hold a mirror
in front of the screen) so the window fills most of the image. The latency of a flip is the time from
showing it to the first camera frame whose brightness crossed halfway. The result includes the display's
own delay (about one refresh, 8-17 ms), so it is a slight overestimate: report it as an upper bound.
"""

import argparse
import random
import time

import cv2
import numpy as np

import config
from core.latest_frame_capture import LatestFrameCapture

WINDOW = "Camera latency test"


def brightness(frame) -> float:
    h, w = frame.shape[:2]
    centre = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(cv2.cvtColor(centre[::4, ::4], cv2.COLOR_BGR2GRAY).mean())


def show(level: int, size=(720, 1280)) -> float:
    cv2.imshow(WINDOW, np.full((*size, 3), level, np.uint8))
    cv2.waitKey(1)
    return time.perf_counter()


def settle(cap, seconds: float) -> float:
    """Mean brightness over the last frames after waiting (lets auto-exposure settle)."""
    end, values = time.perf_counter() + seconds, []
    while time.perf_counter() < end:
        success, frame = cap.read()
        if success:
            values.append(brightness(frame))
        cv2.waitKey(1)
    return float(np.mean(values[-5:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    raw = cv2.VideoCapture(args.camera)
    raw.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
    raw.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
    raw.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap = LatestFrameCapture(raw)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Point the camera at the window. Calibrating...")
    show(0)
    dark = settle(cap, 2.0)
    show(255)
    bright = settle(cap, 2.0)
    if bright - dark < 30:
        cap.release()
        raise SystemExit(f"The camera barely sees the flips (black {dark:.0f}, white {bright:.0f}): "
                         "fill more of the image with the window, or turn off auto-exposure.")
    middle = (dark + bright) / 2.0
    print(f"Black {dark:.0f}, white {bright:.0f}")

    latencies, level = [], 255
    for trial in range(args.trials):
        level = 0 if level else 255
        shown = show(level)
        result = None
        while time.perf_counter() - shown < 1.0:
            success, frame = cap.read()
            if success and cap.capture_time > shown and (brightness(frame) > middle) == (level == 255):
                result = 1000.0 * (cap.capture_time - shown)
                break
        if result is None:
            print(f"Trial {trial + 1}: no change seen")
        else:
            latencies.append(result)
            print(f"Trial {trial + 1}: {result:.1f} ms")
        # Random pause so the flips do not lock onto the camera's frame timing
        settle(cap, random.uniform(0.4, 0.8))
    cap.release()
    cv2.destroyAllWindows()

    if latencies:
        v = np.array(latencies)
        print(f"\nCamera latency over {v.size} flips: median {np.median(v):.1f} ms, mean {v.mean():.1f} ms, "
              f"p95 {np.percentile(v, 95):.1f} ms, min {v.min():.1f} ms (includes display delay)")
        print(f"Use: python -m evaluation.report SESSION --camera-latency-ms {np.median(v):.0f}")


if __name__ == "__main__":
    main()
