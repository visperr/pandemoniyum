"""
Ground-truth annotation of a recorded session's raw video.

    python -m evaluation.annotate keypoints SESSION [--count 60] [--person ID]
        Click the evaluated keypoints (session.KEYPOINTS) on evenly spaced frames.
        click = place the asked keypoint   n = not visible (occluded / outside the image)
        u = undo   s = skip this frame   q = save and quit (annotation resumes where it stopped)

    python -m evaluation.annotate events SESSION [--person ID]
        Step through the video and mark the frame where a movement physically starts.
        a/d = previous/next frame   A/D = 10 frames   space = play/pause
        c = hand starts closing   o = hand starts opening   g = hand sign (e.g. thumbs up) formed
        m = arm starts moving     h = which hand (either/left/right)   x = delete events on this frame
        v = show/hide the tracker output (keep it hidden while marking, so it doesn't bias you)
        q = save and quit

The video is shown mirrored, exactly as the tracker saw it, and coordinates are stored in that image.
"left"/"right" always mean the player's own left/right hand. Annotations are saved next to the video.
"""

import argparse
import json

import cv2
import numpy as np

from evaluation.session import (EVENT_ANNOTATIONS, HAND_KEYS, KEYPOINT_ANNOTATIONS, KEYPOINTS, Session,
                                WRIST_OF_HAND)

WINDOW = "Annotate"
EVENT_TYPES = {ord("c"): "close", ord("o"): "open", ord("g"): "sign", ord("m"): "move"}
HAND_MODES = ("either", "left", "right")


class VideoFrames:
    """Random access to a video's frames (sequential reads where possible), mirrored like the tracker's input."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = -1
        self.frame = None

    def get(self, k: int) -> np.ndarray:
        if k != self.pos:
            if k != self.pos + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            success, frame = self.cap.read()
            if not success:
                raise IndexError(f"Cannot read video frame {k}")
            self.frame, self.pos = cv2.flip(frame, 1), k
        return self.frame.copy()


def text(img, line: str, row: int, color=(255, 255, 255)) -> None:
    y = 24 + 24 * row
    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def video_session(path) -> Session:
    """The session that owns the raw video (a replay points to its source session)."""
    session = Session(path)
    if not (session.path / "raw.mp4").exists() and session.source_path() is not None:
        session = Session(session.source_path())
    if session.video_path() is None:
        raise SystemExit(f"{path} has no raw video to annotate (it was recorded with --no-record-video)")
    return session


def video_frames_of(session: Session) -> dict:
    """Video frame index -> session frame record."""
    return {record["video_frame"]: record for record in session.frames if "video_frame" in record}


def load(session: Session, filename: str, default: dict) -> dict:
    existing = session.annotations(filename)
    return existing if existing is not None else default


def save(session: Session, filename: str, data: dict) -> None:
    with open(session.path / filename, "w") as f:
        json.dump(data, f, indent=1)


def annotate_keypoints(session: Session, pid: int, count: int, scale: float) -> None:
    data = load(session, KEYPOINT_ANNOTATIONS, {"person": pid, "keypoints": list(KEYPOINTS), "frames": {}})
    pid = data["person"]
    by_video = video_frames_of(session)
    candidates = [k for k, record in sorted(by_video.items()) if session.person(record, pid)]
    if not candidates:
        raise SystemExit(f"Person {pid} never appears in {session.path}")
    picks = [candidates[int(round(x))] for x in np.linspace(0, len(candidates) - 1, min(count, len(candidates)))]
    todo = [k for k in picks if str(by_video[k]["i"]) not in data["frames"]]
    print(f"{len(picks) - len(todo)} of {len(picks)} frames already annotated")

    video = VideoFrames(session.video_path())
    clicks = []
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, *_: clicks.append((x / scale, y / scale))
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    names = list(KEYPOINTS)

    for n, k in enumerate(todo):
        record = by_video[k]
        box = session.person(record, pid)["box"]
        base = video.get(k)
        points, j = {}, 0
        clicks.clear()
        while j < len(names):
            img = base.copy()
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 200, 255), 1)
            for name, p in points.items():
                if p is not None:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
                    cv2.putText(img, name, (int(p[0]) + 4, int(p[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 255, 0), 1)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            text(img, f"Frame {record['i']}  ({n + 1}/{len(todo)} left)  player in the orange box", 0)
            text(img, f"Click the player's {names[j].replace('_', ' ').upper()}", 1, (0, 255, 255))
            text(img, "n = not visible   u = undo   s = skip frame   q = save + quit", 2)
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(20) & 0xFF
            if clicks:
                points[names[j]] = [round(c, 1) for c in clicks.pop(0)]
                j += 1
            elif key == ord("n"):
                points[names[j]] = None
                j += 1
            elif key == ord("u") and j > 0:
                j -= 1
                points.pop(names[j], None)
            elif key == ord("s"):
                break
            elif key in (ord("q"), 27):
                save(session, KEYPOINT_ANNOTATIONS, data)
                print(f"Saved {len(data['frames'])} annotated frames")
                return
        if j == len(names):
            data["frames"][str(record["i"])] = points
            save(session, KEYPOINT_ANNOTATIONS, data)
    save(session, KEYPOINT_ANNOTATIONS, data)
    print(f"Done: {len(data['frames'])} annotated frames in {session.path / KEYPOINT_ANNOTATIONS}")


def draw_output(img, person: dict) -> None:
    """The sent wrists and hand states, for checking an annotation afterwards."""
    if not person or not person.get("pose") or not person.get("crop"):
        return
    x1, y1, x2, y2 = person["crop"]
    for side, hand in HAND_KEYS.items():
        row = person["pose"][KEYPOINTS[WRIST_OF_HAND[side]]]
        p = (int(x1 + row[0] * (x2 - x1)), int(y1 + row[1] * (y2 - y1)))
        state = person.get("hands", {}).get(hand, {})
        closed = state.get("closed", False)
        cv2.circle(img, p, 6, (0, 0, 255) if closed else (0, 255, 0), 2)
        label = ("closed" if closed else "open") + ("" if state.get("gesture", "None") == "None"
                                                    else f" {state['gesture']}")
        cv2.putText(img, label, (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def annotate_events(session: Session, pid: int, scale: float) -> None:
    data = load(session, EVENT_ANNOTATIONS, {"person": pid, "events": []})
    pid = data["person"]
    by_video = video_frames_of(session)
    video = VideoFrames(session.video_path())
    last = min(video.count, len(by_video)) - 1
    k, playing, hand_mode, show_output = 0, False, 0, False
    cv2.namedWindow(WINDOW)
    t0 = by_video[0]["t"]

    while True:
        record = by_video[k]
        img = video.get(k)
        if show_output:
            draw_output(img, session.person(record, pid))
        img = cv2.resize(img, None, fx=scale, fy=scale)
        here = [e for e in data["events"] if e["i"] == record["i"]]
        text(img, f"Frame {record['i']}   t = {record['t'] - t0:6.2f} s   {'PLAYING' if playing else ''}", 0)
        text(img, f"Hand: {HAND_MODES[hand_mode]} (h)   c close  o open  g sign  m move  x delete  "
                  f"v output  q quit", 1)
        if here:
            text(img, "Marked here: " + ", ".join(f"{e['type']} ({e['hand']})" for e in here), 2, (0, 255, 255))
        # Timeline with the marked events
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 12), (w, h), (40, 40, 40), -1)
        for e in data["events"]:
            x = int(w * (session.index.get(e["i"], 0) / max(last, 1)))
            cv2.line(img, (x, h - 12), (x, h), (0, 255, 255), 2)
        cv2.line(img, (int(w * k / max(last, 1)), h - 16), (int(w * k / max(last, 1)), h), (255, 255, 255), 1)
        cv2.imshow(WINDOW, img)

        key = cv2.waitKey(33 if playing else 0) & 0xFF
        if playing and key == 255:
            k = min(k + 1, last)
            playing = k < last
            continue
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            k = min(k + 1, last)
        elif key == ord("a"):
            k = max(k - 1, 0)
        elif key == ord("D"):
            k = min(k + 10, last)
        elif key == ord("A"):
            k = max(k - 10, 0)
        elif key == ord("h"):
            hand_mode = (hand_mode + 1) % len(HAND_MODES)
        elif key == ord("v"):
            show_output = not show_output
        elif key == ord("x"):
            data["events"] = [e for e in data["events"] if e["i"] != record["i"]]
        elif key in EVENT_TYPES:
            data["events"].append({"i": record["i"], "type": EVENT_TYPES[key], "hand": HAND_MODES[hand_mode]})
            data["events"].sort(key=lambda e: e["i"])
            save(session, EVENT_ANNOTATIONS, data)
    save(session, EVENT_ANNOTATIONS, data)
    print(f"Saved {len(data['events'])} events to {session.path / EVENT_ANNOTATIONS}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("keypoints", "events"))
    parser.add_argument("session")
    parser.add_argument("--person", type=int, help="player id to annotate (default: the one tracked longest)")
    parser.add_argument("--count", type=int, default=60, help="keypoints: number of frames to annotate")
    parser.add_argument("--scale", type=float, default=1.5, help="display zoom")
    args = parser.parse_args()

    session = video_session(args.session)
    pid = args.person if args.person is not None else session.main_person()
    if args.mode == "keypoints":
        annotate_keypoints(session, pid, args.count, args.scale)
    else:
        annotate_events(session, pid, args.scale)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
