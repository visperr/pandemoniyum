"""
Influence of lighting: one row per session, recorded live under different lighting or replayed with simulated
lighting (main.py --replay SESSION --brightness 0.3 --record ...).

    python -m evaluation.compare SESSION [SESSION ...] [--reference SESSION] [--out lighting.md]

--reference is the normally lit run of the same video (for simulated lighting: a replay with brightness 1);
every session is then also compared with it frame by frame, which needs no ground truth.
"""

import argparse
import json
from pathlib import Path

from evaluation import metrics
from evaluation.report import fmt, table
from evaluation.session import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--reference", help="normally lit replay of the same video")
    parser.add_argument("--person", type=int)
    parser.add_argument("--out", default="lighting.md", help="Markdown output (a .json is written next to it)")
    args = parser.parse_args()

    reference = Session(args.reference) if args.reference else None
    rows = []
    for path in args.sessions:
        session = Session(path)
        pid = args.person if args.person is not None else (reference.main_person() if reference else None)
        rows.append(metrics.lighting_summary(session, pid, reference))
    rows.sort(key=lambda r: -r["luma_mean"])

    header = ["Session", "Condition", "Brightness", "Tracked", "Pose found", "Keypoint det.", "Wrist det.",
              "Wrist vis.", "Hand seen", "Wrist jitter raw / sent %SW", "GT PCK@0.2 raw / sent"]
    if reference:
        header += ["vs. ref: median %SW", "vs. ref: within 0.2 SW", "Pose frames lost"]
    lines = []
    for r in rows:
        line = [r["session"], r["condition"], fmt(r["luma_mean"], 0), fmt(r.get("player_tracked_rate"), pct=True),
                fmt(r.get("pose_found_rate"), pct=True), fmt(r.get("detection_rate_mean"), pct=True),
                fmt(r.get("wrist_detection_rate"), pct=True), fmt(r.get("wrist_visibility"), 2),
                fmt(r.get("hand_seen_rate"), pct=True),
                f"{fmt(r.get('wrist_jitter_raw_pct_sw'), 2)} / {fmt(r.get('wrist_jitter_sent_pct_sw'), 2)}",
                f"{fmt(r.get('gt_raw_pck@0.2'), pct=True)} / {fmt(r.get('gt_pose_pck@0.2'), pct=True)}"]
        if reference:
            ref = r.get("vs_reference", {}).get("pose", {})
            line += [fmt(ref.get("median_pct_sw"), 2), fmt(ref.get("within_0.2sw"), pct=True),
                     ref.get("pose_lost_frames", "–")]
        lines.append(line)

    markdown = ("# Influence of lighting\n\nBrightness = mean frame brightness (0-255). Detection rates are "
                "shares of the pose model's runs; jitter is the stationary RMS (meaningful for sessions where "
                "the player holds still).\n\n" + table(header, lines) + "\n")
    out = Path(args.out)
    out.write_text(markdown)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    print(markdown)
    print(f"Saved {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
