"""
All metrics of one recorded session as a Markdown report (plus the numbers as JSON).

    python -m evaluation.report SESSION [--person ID] [--camera-latency-ms 60] [--plots]

Writes SESSION/report.md and SESSION/metrics.json. Keypoint error and gesture delay need the annotations
from evaluation.annotate; the synthetic missing-keypoint test is evaluation.missing_keypoints.
"""

import argparse
import json
import math

import numpy as np

from evaluation import metrics
from evaluation.session import EVENT_ANNOTATIONS, KEYPOINT_ANNOTATIONS, Session


def fmt(value, digits: int = 1, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if pct:
        return f"{100.0 * value:.{digits}f}%"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def stat(s: dict, key: str = "median", digits: int = 1) -> str:
    return fmt(s.get(key), digits) if s and s.get("n") else "–"


def section_detection(det: dict) -> str:
    rows = [[name, fmt(k["detection_rate"], pct=True), fmt(k["mean_visibility"], 2),
             fmt(k["sent_measured_rate"], pct=True), fmt(k["sent_inferred_rate"], pct=True)]
            for name, k in det["keypoints"].items()]
    return (f"## 1a. Keypoint detection rate\n\n"
            f"Player tracked on {fmt(det['player_tracked_rate'], pct=True)} of {det['frames']} frames "
            f"(held/frozen box on {fmt(det['frozen_rate'], pct=True)} of those). The pose model ran "
            f"{det['pose_model_runs']} times and found the player's pose in {fmt(det['pose_found_rate'], pct=True)}.\n\n"
            + table(["Keypoint", "Detection rate", "Mean visibility", "Sent: measured", "Sent: inferred"], rows))


def section_error(err: dict) -> str:
    sw_mm = err.get("shoulder_width_mm")
    mm_note = (f" mm = scaled by the player's measured shoulder width ({fmt(float(sw_mm), 0)} mm): a 2D error in "
               "the image plane, exact for joints about as far from the camera as the shoulders."
               if sw_mm else " For errors in mm, record with --shoulder-width-mm or pass it to the report.")
    out = [f"## 1b. Keypoint error vs. ground truth\n\n{err['frames']} annotated frames. "
           "PCK@a = share of visible ground-truth keypoints predicted within a x shoulder width." + mm_note + "\n"]
    for kind, title in (("raw", "Pose model output (before smoothing)"), ("pose", "Sent pose (after smoothing)")):
        rows = [[name, r["gt_visible"], fmt(r["mean_px"]), fmt(r["median_px"]), fmt(r["mean_pct_sw"])]
                + ([fmt(r["mean_mm"], 0), fmt(r["median_mm"], 0)] if sw_mm else [])
                + [fmt(r["pck@0.1"], pct=True), fmt(r["pck@0.2"], pct=True), fmt(r["pck@0.5"], pct=True),
                   fmt(r["recall"], pct=True), fmt(r["precision"], pct=True)]
                for name, r in err[kind].items()]
        out.append(f"### {title}\n\n" + table(
            ["Keypoint", "GT visible", "Mean px", "Median px", "Mean %SW"]
            + (["Mean mm", "Median mm"] if sw_mm else [])
            + ["PCK@0.1", "PCK@0.2", "PCK@0.5", "Recall", "Precision"], rows))
    return "\n\n".join(out)


def section_jitter(jit: dict, shoulder_width_mm=None) -> str:
    def cell(px, pct):
        mm = f", {fmt(pct * shoulder_width_mm / 100.0, 1)} mm" if shoulder_width_mm and math.isfinite(pct) else ""
        return f"{fmt(px, 2)} ({fmt(pct, 2)}{mm})"

    rows = []
    for name, k in jit["keypoints"].items():
        r, s = k["raw"], k["sent"]
        rows.append([name, k["frames"],
                     cell(r["stationary_rms_px"], r["stationary_rms_pct_sw"]),
                     cell(s["stationary_rms_px"], s["stationary_rms_pct_sw"]),
                     fmt(k["reduction_stationary_pct"], 0) + "%",
                     cell(r["dynamic_resid_rms_px"], r["dynamic_resid_rms_pct_sw"]),
                     cell(s["dynamic_resid_rms_px"], s["dynamic_resid_rms_pct_sw"]),
                     fmt(k["reduction_dynamic_pct"], 0) + "%", fmt(k["lag_ms"], 0), fmt(r["mean_speed_px_s"], 0)])
    units = "px (%SW, mm)" if shoulder_width_mm else "px (%SW)"
    return (f"## 2. Jitter\n\nOn frames where the joint is clearly seen; {units}, shoulder width "
            f"{fmt(jit['shoulder_width_px'])} px, {fmt(jit['fps'])} fps. Use the stationary columns for a "
            "session where the player holds still and the dynamic columns for one with movement.\n\n"
            + table(["Keypoint", "Frames", "Stationary raw", "Stationary sent", "Reduction", "Dynamic raw",
                     "Dynamic sent", "Reduction", "Lag ms", "Speed px/s"], rows))


def section_delay(delay: dict) -> str:
    rows = []
    for kind, s in delay["summary"].items():
        if kind == "false_triggers":
            continue
        rows.append([kind, s["events"], fmt(s["detected_rate"], pct=True),
                     stat(s.get("delay_frames"), digits=1),
                     f"{stat(s.get('algorithmic_ms'), digits=0)} / {stat(s.get('algorithmic_ms'), 'p95', 0)}",
                     f"{stat(s.get('raw_algorithmic_ms'), digits=0)}",
                     f"{stat(s.get('end_to_end_ms'), digits=0)} / {stat(s.get('end_to_end_ms'), 'p95', 0)}",
                     f"{stat(s.get('total_ms'), digits=0)} / {stat(s.get('total_ms'), 'p95', 0)}"])
    ft = delay["summary"]["false_triggers"]
    false_line = ft.get("note") or f"{ft['count']} grab state flips without an annotated onset " \
                                   f"({fmt(ft['per_minute'], 2)} per minute)."
    cam = delay["camera_latency_ms"]
    return ("## 3. Motion / gesture recognition delay\n\nMedian / 95th percentile in ms from the annotated "
            "physical onset. Algorithmic = frames of debouncing/smoothing; end-to-end = camera delivers the "
            "onset frame -> packet sent; total adds the camera latency"
            + (f" ({fmt(cam, 0)} ms)" if cam is not None else " (pass --camera-latency-ms)") + ".\n\n"
            + table(["Event", "Count", "Detected", "Delay frames", "Algorithmic", "Raw (move)", "End-to-end",
                     "Total"], rows) + f"\n\n{false_line}")


def section_missing(miss: dict) -> str:
    rows = [[name, fmt(k["inferred_rate"], pct=True), k["gaps"], stat(k["gap_s"], digits=2),
             stat(k["gap_s"], "p95", 2), stat(k["gap_s"], "max", 2), stat(k["reentry_pop_pct_sw"], digits=1),
             stat(k["reentry_pop_pct_sw"], "max", 1), fmt(k["normal_step_p95_pct_sw"], 1)]
            for name, k in miss["keypoints"].items()]
    return ("## 4. Missing keypoints (as they occurred)\n\n"
            f"Whole pose missing on {fmt(miss['whole_pose_dropout_rate'], pct=True)} of tracked frames "
            f"(median gap {stat(miss['whole_pose_dropout_s'], digits=2)} s); pose model found nothing on "
            f"{fmt(miss['pose_not_found_rate'], pct=True)} of its runs.\n\n"
            + table(["Keypoint", "Inferred", "Gaps", "Median s", "p95 s", "Max s", "Re-entry pop %SW (median)",
                     "Pop max", "Normal step p95 %SW"], rows)
            + "\n\nFor the accuracy of the inferred positions run `python -m evaluation.missing_keypoints`.")


def section_latency(lat: dict) -> str:
    rows = [[name, stat(s, "mean"), stat(s), stat(s, "p95"), stat(s, "max"), s.get("n", 0)]
            for name, s in lat["stages_ms"].items()]
    live = "" if lat["live"] else ("\n\n**Not a live camera session**: capture timestamps are file reads, so "
                                   "end-to-end figures are processing time only.")
    preview = "" if not lat.get("preview") else " The preview window was on, which adds drawing time."
    e2e, interval = lat["end_to_end_ms"], lat["interval_ms"]
    summary = table(["Measure", "Mean", "Median", "p95", "p99", "Max"], [
        ["End-to-end latency (camera -> UDP) ms", stat(e2e, "mean"), stat(e2e), stat(e2e, "p95"), stat(e2e, "p99"),
         stat(e2e, "max")],
        ["  of which waiting for the loop ms", *(stat(lat["queue_wait_ms"], k) for k in
                                                 ("mean", "median", "p95", "p99", "max"))],
        ["  of which processing ms", *(stat(lat["processing_ms"], k) for k in ("mean", "median", "p95", "p99", "max"))],
        ["Packet interval ms", *(stat(interval, k) for k in ("mean", "median", "p95", "p99", "max"))],
    ])
    fps = lat.get("output_fps", {})
    fps_line = (f"Frame rate: mean {stat(fps, 'mean')}, 5% low {stat(fps, 'low_5pct')}, 1% low "
                f"{stat(fps, 'low_1pct')}, slowest second {stat(fps, 'min_1s', 0)} fps; "
                f"{fmt(fps.get('below_25'), pct=True)} of seconds below 25 fps (x% low = mean rate over the "
                "slowest x% of packet intervals).")
    rates = (f"Output rate **{fmt(lat['output_rate_hz'])} Hz**, camera delivered {fmt(lat['camera_rate_hz'])} Hz "
             f"({lat['dropped_frames']} stale frames skipped). Pose re-measured at {fmt(lat.get('pose_rate_hz'))} Hz, "
             f"each hand's grab state at {fmt(lat.get('hand_rate_hz'))} Hz.")
    return (f"## 5. Module latency and update rate\n\n{lat['frames']} frames over {fmt(lat['duration_s'])} s "
            f"(first {lat['warmup_frames_skipped']} warm-up frames left out)."
            f"{preview}{live}\n\n{summary}\n\n{rates}\n\n{fps_line}\n\n### Per stage (ms, frames where the stage ran)\n\n"
            + table(["Stage", "Mean", "Median", "p95", "Max", "Frames"], rows))


def plots(session: Session, pid: int, result: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    t = session.column("t")
    raw, sent = session.track(pid, "raw"), session.track(pid, "pose")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (axis, label) in zip(axes, ((0, "x"), (1, "y"))):
        j = 15
        ax.plot(t - t[0], raw.xy[:, j, axis], lw=0.8, color="#9aa0a6", label="raw (pose model)")
        ax.plot(t - t[0], sent.xy[:, j, axis], lw=1.2, color="#1a73e8", label="sent (smoothed)")
        ax.set_ylabel(f"left wrist {label} (px)")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    files.append(session.path / "plot_wrist_trace.png")
    fig.savefig(files[-1], dpi=120)
    plt.close(fig)

    e2e = 1000.0 * (session.column("t_send") - session.column("t_capture"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(e2e[np.isfinite(e2e)], bins=50, color="#1a73e8")
    ax.set_xlabel("end-to-end latency, camera -> UDP (ms)")
    ax.set_ylabel("frames")
    fig.tight_layout()
    files.append(session.path / "plot_latency.png")
    fig.savefig(files[-1], dpi=120)
    plt.close(fig)
    return files


def factors_line(session: Session) -> str:
    """The recording factors that were given (subject, distance, light level, shoulder width)."""
    labels = (("subject", "subject {}", None), ("distance_m", "distance {} m", 1), ("lux", "{} lux", 0),
              ("shoulder_width_mm", "shoulder width {} mm", 0))
    return ", ".join(label.format(fmt(float(value), digits) if digits is not None else value)
                     for key, label, digits in labels if (value := session.factor(key)) is not None)


def build_report(session: Session, pid: int, camera_latency_ms=None, shoulder_width_mm=None) -> tuple:
    if shoulder_width_mm is None:
        shoulder_width_mm = session.factor("shoulder_width_mm")
    result = {"session": session.name, "condition": session.condition, "person": pid,
              "detection": metrics.detection_rates(session, pid),
              "jitter": metrics.jitter(session, pid),
              "missing": metrics.missing_keypoints(session, pid),
              "latency": metrics.latency(session)}
    keypoint_annotations = session.annotations(KEYPOINT_ANNOTATIONS)
    if keypoint_annotations:
        result["keypoint_error"] = metrics.keypoint_error(session, keypoint_annotations, shoulder_width_mm)
    event_annotations = session.annotations(EVENT_ANNOTATIONS)
    if event_annotations:
        result["delay"] = metrics.gesture_delay(session, event_annotations, camera_latency_ms=camera_latency_ms)

    lighting = session.meta.get("lighting") or {}
    simulated = {k: v for k, v in lighting.items() if v != (0.0 if k == "noise" else 1.0)}
    factors = factors_line(session)
    parts = [f"# Tracking evaluation: {session.name}\n\n"
             f"Condition **{session.condition}**, source {session.meta.get('source')}, player {pid}, "
             f"mean frame brightness {fmt(float(np.nanmean(session.column('luma'))))} / 255"
             + (f", simulated lighting {simulated}" if simulated else "")
             + (f", {factors}" if factors else "")
             + (f". Notes: {session.meta['notes']}" if session.meta.get("notes") else "") + ".",
             section_detection(result["detection"])]
    parts.append(section_error(result["keypoint_error"]) if "keypoint_error" in result else
                 "## 1b. Keypoint error vs. ground truth\n\nNo annotations yet: "
                 "`python -m evaluation.annotate keypoints SESSION`.")
    parts.append(section_jitter(result["jitter"], shoulder_width_mm))
    parts.append(section_delay(result["delay"]) if "delay" in result else
                 "## 3. Motion / gesture recognition delay\n\nNo events annotated yet: "
                 "`python -m evaluation.annotate events SESSION`.")
    parts.append(section_missing(result["missing"]))
    parts.append(section_latency(result["latency"]))
    return result, "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session")
    parser.add_argument("--person", type=int, help="player id (default: the one tracked longest)")
    parser.add_argument("--camera-latency-ms", type=float, help="from evaluation.camera_latency, added to delays")
    parser.add_argument("--plots", action="store_true", help="also save wrist trace and latency histogram PNGs")
    parser.add_argument("--shoulder-width-mm", type=float,
                        help="player's measured shoulder width, for errors in mm (default: from the recording)")
    args = parser.parse_args()

    session = Session(args.session)
    pid = args.person if args.person is not None else session.main_person()
    if pid is None:
        raise SystemExit("No player was tracked in this session")
    result, markdown = build_report(session, pid, args.camera_latency_ms, args.shoulder_width_mm)
    (session.path / "report.md").write_text(markdown)
    with open(session.path / "metrics.json", "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(markdown)
    if args.plots:
        for path in plots(session, pid, result):
            print(f"Saved {path}")
    print(f"Saved {session.path / 'report.md'} and metrics.json")


if __name__ == "__main__":
    main()
