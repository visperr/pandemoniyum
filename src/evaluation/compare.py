"""
Influence of the impact factors (lighting, distance, subject, ...): one row per session, recorded live under
different conditions or replayed with simulated lighting (main.py --replay SESSION --brightness 0.3 --record ...).

    python -m evaluation.compare SESSION [SESSION ...] [--by FACTOR ...] [--reference SESSION] [--out compare.md]

--by groups the sessions by one or more factors (condition, subject, distance, lux, brightness, noise, as
recorded with main.py --subject/--distance/--lux) and adds a table with the mean and standard deviation over
the sessions of each group, i.e. over the repetitions of that condition.
--reference is the normally lit run of the same video (for simulated lighting: a replay with brightness 1);
every session is then also compared with it frame by frame, which needs no ground truth.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluation import metrics
from evaluation.report import fmt, table
from evaluation.session import Session

# (row key, column title, digits, is a share): the numbers averaged over the repetitions of a group
GROUPED = (
    ("pose_found_rate", "Pose found", 1, True),
    ("detection_rate_mean", "Keypoint det.", 1, True),
    ("wrist_detection_rate", "Wrist det.", 1, True),
    ("wrist_jitter_sent_pct_sw", "Wrist jitter sent %SW", 2, False),
    ("gt_pose_pck@0.2", "GT PCK@0.2 sent", 1, True),
    ("gt_pose_mean_mm", "GT error sent mm", 0, False),
    ("fps_low_1pct", "FPS 1% low", 1, False),
    ("ref_median_pct_sw", "vs. ref: median %SW", 2, False),
)


def factor_label(value) -> str:
    if value is None:
        return "–"
    return fmt(float(value), 2).rstrip("0").rstrip(".") if isinstance(value, (int, float)) else str(value)


def mean_sd(values: list, digits: int, share: bool) -> str:
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return "–"
    scale, unit = (100.0, "%") if share else (1.0, "")
    text = f"{scale * np.mean(values):.{digits}f}"
    if len(values) > 1:
        text += f" ± {scale * np.std(values, ddof=1):.{digits}f}"
    return text + unit


def grouped_table(rows: list, by: list) -> str:
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r["factors"][name] for name in by), []).append(r)

    def order(key):  # numbers ascending, missing values last
        return [(v is None, v if isinstance(v, (int, float)) else 0, str(v)) for v in key]

    lines = []
    for key in sorted(groups, key=order):
        members = groups[key]
        lines.append([factor_label(v) for v in key] + [len(members)]
                     + [mean_sd([m.get(name) for m in members], digits, share) for name, _, digits, share in GROUPED])
    return table([name.capitalize() for name in by] + ["Sessions"] + [title for _, title, _, _ in GROUPED], lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--by", nargs="+", choices=metrics.FACTORS, default=[], metavar="FACTOR",
                        help="group by these factors (" + ", ".join(metrics.FACTORS) + ") and average over "
                             "the sessions of each group")
    parser.add_argument("--reference", help="normally lit replay of the same video")
    parser.add_argument("--person", type=int)
    parser.add_argument("--out", help="Markdown output, a .json is written next to it "
                                      "(default: compare.md, or compare_<factors>.md with --by)")
    args = parser.parse_args()

    reference = Session(args.reference) if args.reference else None
    rows = []
    for path in args.sessions:
        session = Session(path)
        pid = args.person if args.person is not None else (reference.main_person() if reference else None)
        row = metrics.session_summary(session, pid, reference)
        row["ref_median_pct_sw"] = row.get("vs_reference", {}).get("pose", {}).get("median_pct_sw")
        rows.append(row)
    rows.sort(key=lambda r: -r["luma_mean"])

    header = ["Session", "Condition", "Subject", "Distance m", "Lux", "Brightness", "Tracked", "Pose found",
              "Keypoint det.", "Wrist det.", "Wrist vis.", "Hand seen", "Wrist jitter raw / sent %SW",
              "GT PCK@0.2 raw / sent", "GT error sent mm", "FPS mean / 1% low"]
    if reference:
        header += ["vs. ref: median %SW", "vs. ref: within 0.2 SW", "Pose frames lost"]
    lines = []
    for r in rows:
        f = r["factors"]
        line = [r["session"], r["condition"], factor_label(f["subject"]), factor_label(f["distance"]),
                factor_label(f["lux"]), fmt(r["luma_mean"], 0), fmt(r.get("player_tracked_rate"), pct=True),
                fmt(r.get("pose_found_rate"), pct=True), fmt(r.get("detection_rate_mean"), pct=True),
                fmt(r.get("wrist_detection_rate"), pct=True), fmt(r.get("wrist_visibility"), 2),
                fmt(r.get("hand_seen_rate"), pct=True),
                f"{fmt(r.get('wrist_jitter_raw_pct_sw'), 2)} / {fmt(r.get('wrist_jitter_sent_pct_sw'), 2)}",
                f"{fmt(r.get('gt_raw_pck@0.2'), pct=True)} / {fmt(r.get('gt_pose_pck@0.2'), pct=True)}",
                fmt(r.get("gt_pose_mean_mm"), 0), f"{fmt(r['fps_mean'])} / {fmt(r['fps_low_1pct'])}"]
        if reference:
            ref = r.get("vs_reference", {}).get("pose", {})
            line += [fmt(ref.get("median_pct_sw"), 2), fmt(ref.get("within_0.2sw"), pct=True),
                     ref.get("pose_lost_frames", "–")]
        lines.append(line)

    title = "Influence of " + " and ".join(args.by) if args.by else "Session comparison"
    markdown = f"# {title}\n\n"
    if args.by:
        markdown += ("Mean ± standard deviation over the sessions of each group (the repetitions); a single "
                     "session shows no deviation.\n\n" + grouped_table(rows, args.by) + "\n\n## Per session\n\n")
    markdown += ("Brightness = mean frame brightness (0-255). Detection rates are shares of the pose model's "
                 "runs; jitter is the stationary RMS (meaningful for sessions where the player holds still); GT "
                 "columns need keypoint annotations (mm also needs --shoulder-width-mm).\n\n"
                 + table(header, lines) + "\n")
    out = Path(args.out or ("compare_" + "_".join(args.by) + ".md" if args.by else "compare.md"))
    out.write_text(markdown)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"by": args.by, "sessions": rows}, f, indent=1, default=float)
    print(markdown)
    print(f"Saved {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
