"""
One command for a recorded play session of PANdemoniYUM.

    .venv/bin/python play_session.py [--game PATH] [--notes "..."]

1. Starts the tracker in evaluation mode (main.py --record): metrics data, the raw camera video and a video of
   the annotated tracking view.
2. Once the tracker is running, starts the built game, which records its own screen.
3. Play. Quit the game (Cmd+Q) to end the session; the tracker is stopped then.
4. Stitches the two recordings into one video, the tracking view small in the top right of the gameplay, lined up in
   time from the first-frame times both recordings store, and writes the tracking metrics report.

Everything goes into recordings/<date>_<time>_play/:
    game.mp4                gameplay as rendered (1280 px wide, 30 fps)
    tracking.mp4            the tracker's annotated camera view (30 fps)
    combined.mp4            gameplay with the tracking view inset
    <date>_<condition>/     the tracker's evaluation session: frames.jsonl, raw.mp4, report.md, metrics.json

Any other arguments go to the tracker, e.g. --video clip.mp4 to run it on a file instead of the camera.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RECORDINGS = ROOT / "recordings"
DEFAULT_GAME = Path(os.environ.get("PANDEMONIYUM_GAME", Path.home() / "PANdemoniYUM-game" / "PANdemoniYUM.app"))
TRACKER_READY = "Tracking pipeline started"


def parse_args():
    parser = argparse.ArgumentParser(description="Play PANdemoniYUM with the tracker, recording both.")
    parser.add_argument("--game", type=Path, default=DEFAULT_GAME, help=f"the built game (default: {DEFAULT_GAME})")
    parser.add_argument("--condition", default="gameplay", help="label for the tracker's evaluation session")
    parser.add_argument("--notes", default="", help="free text stored with the session (players, lighting, ...)")
    parser.add_argument("--inset", choices=["bottom-center", "bottom-right", "bottom-left", "top-right", "top-left"],
                        default="top-right", help="where on the gameplay the tracking view goes")
    parser.add_argument("--inset-size", type=float, default=0.25, help="tracking view width, share of the video")
    parser.add_argument("--no-report", action="store_true", help="skip the tracking metrics report")
    return parser.parse_known_args()


def python_executable() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def start_tracker(session: Path, args, tracker_args) -> tuple[subprocess.Popen, threading.Event]:
    """Starts main.py, echoing its output; the event is set once it's tracking."""
    command = [python_executable(), "-u", "main.py", "--record", "--condition", args.condition, "--notes", args.notes,
               "--recordings-dir", str(session), "--tracking-video", str(session / "tracking.mp4"), "--no-preview",
               *tracker_args]
    tracker = subprocess.Popen(command, cwd=SRC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1)
    ready = threading.Event()

    def echo():
        for line in tracker.stdout:
            print(f"[tracker] {line}", end="")
            if TRACKER_READY in line:
                ready.set()

    threading.Thread(target=echo, daemon=True).start()
    return tracker, ready


def stop_tracker(tracker: subprocess.Popen) -> None:
    if tracker.poll() is not None:
        return
    tracker.send_signal(signal.SIGINT)  # main.py stops cleanly and finishes its recordings
    try:
        tracker.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print("Tracker didn't stop; killing it (its recordings may be incomplete)")
        tracker.kill()


def play_game(game: Path, video: Path, ffmpeg: str) -> None:
    """Runs the game until it's quit; it records itself to video."""
    try:
        subprocess.run(["open", "-W", "-n", str(game), "--args", "-record", str(video), "-ffmpeg", ffmpeg],
                       check=True)
    except KeyboardInterrupt:
        print("\nEnding the session: quitting the game...")
        subprocess.run(["osascript", "-e", f'quit app "{game.stem}"'])
        time.sleep(3)


def timing(video: Path) -> dict:
    with open(f"{video}.json") as f:
        return json.load(f)


def stitch(session: Path, ffmpeg: str, position: str, size: float) -> Path | None:
    """Gameplay with the tracking view inset, lined up by the time each recording's first frame was taken."""
    game, tracking, combined = session / "game.mp4", session / "tracking.mp4", session / "combined.mp4"
    for video in (game, tracking):
        if not video.exists() or not Path(f"{video}.json").exists():
            print(f"Can't stitch: {video.name} (or its timing file) is missing")
            return None

    game_timing, tracking_timing = timing(game), timing(tracking)
    # Seconds into the gameplay at which the tracking video starts (negative: it started earlier)
    offset = tracking_timing["first_frame_time"] - game_timing["first_frame_time"]
    line_up = ["-ss", f"{-offset:.3f}"] if offset < 0 else ["-itsoffset", f"{offset:.3f}"]

    inset_width = int(game_timing["width"] * size) // 2 * 2
    margin = 24
    x = {"left": str(margin), "right": f"W-w-{margin}", "center": "(W-w)/2"}[position.split("-")[1]]
    y = f"H-h-{margin}" if position.startswith("bottom") else str(margin)
    layout = (f"[1:v]scale={inset_width}:-2,pad=iw+6:ih+6:3:3:white[inset];"
              f"[0:v][inset]overlay={x}:{y}:eof_action=pass[out]")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(game), *line_up, "-i", str(tracking),
                    "-filter_complex", layout, "-map", "[out]", "-c:v", "h264_videotoolbox", "-b:v", "10M",
                    "-pix_fmt", "yuv420p", str(combined)], check=True)
    return combined


def tracker_session(session: Path) -> Path | None:
    found = [p.parent for p in session.glob("*/frames.jsonl")]
    return found[0] if found else None


def main():
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up straight away in a log or IDE console too
    args, tracker_args = parse_args()
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).exists():
        sys.exit("ffmpeg is needed for the recordings: brew install ffmpeg")
    if not args.game.exists():
        sys.exit(f"No built game at {args.game}. Build it in Unity (File > Build Profiles > macOS) or pass --game")

    session = RECORDINGS / f"{datetime.now():%Y%m%d_%H%M%S}_play"
    session.mkdir(parents=True)
    print(f"Session folder: {session}")

    tracker, ready = start_tracker(session, args, tracker_args)
    try:
        print("Starting the tracker (loading models)...")
        while not ready.wait(0.5):
            if tracker.poll() is not None:
                sys.exit("The tracker stopped before it was ready; see its output above")
        print("Tracker running. Starting the game: quit it (Cmd+Q) to end the session.")
        play_game(args.game, session / "game.mp4", ffmpeg)
    finally:
        print("Stopping the tracker...")
        stop_tracker(tracker)

    print("Stitching the videos...")
    combined = stitch(session, ffmpeg, args.inset, args.inset_size)

    evaluation = tracker_session(session)
    if evaluation is not None and not args.no_report:
        print("Writing the tracking metrics report...")
        subprocess.run([python_executable(), "-m", "evaluation.report", str(evaluation)], cwd=SRC)

    print("\nDone.")
    if combined is not None:
        print(f"  Combined video:  {combined}")
    print(f"  Session folder:  {session}")
    if evaluation is not None:
        print(f"  Metrics:         {evaluation / 'report.md'}")


if __name__ == "__main__":
    main()
