"""
The tracking module's metrics, computed from a recorded session (see EVALUATION.md for definitions).

Every function returns plain dicts/lists of numbers, so report.py can print them and store them as JSON.
Distances are in pixels of the processed frame and, normalised, in percent of the player's shoulder width
(%SW), which makes them independent of how far the player stands from the camera.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from evaluation.session import (CRITICAL_KEYPOINTS, HAND_KEYS, KEYPOINTS, VISIBLE, WRIST_OF_HAND, Session,
                                segments)

PCK_THRESHOLDS = (0.1, 0.2, 0.5)  # fractions of the shoulder width


def summarize(values) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def frame_rate(session: Session) -> float:
    dt = np.diff(session.column("t"))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if dt.size else 30.0


def _person_flags(session: Session, pid: int, key: str) -> np.ndarray:
    return np.array([bool((session.person(r, pid) or {}).get(key, False)) for r in session.frames])


def _inside(session: Session, xy: np.ndarray) -> np.ndarray:
    w, h = session.column("w"), session.column("h")
    return (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)


# ---------------------------------------------------------------------------- 1. keypoint detection rate / error

def detection_rates(session: Session, pid: int) -> dict:
    """
    How often the pose model delivers each keypoint, without ground truth.
    detection_rate: share of frames the pose model ran on this player in which the keypoint was found with
    visibility >= 0.5 inside the image. sent_measured/inferred: share of frames with a sent pose in which the
    joint was measured, or filled in by the missing-keypoint handling.
    """
    raw, sent = session.track(pid, "raw"), session.track(pid, "pose")
    ran = _person_flags(session, pid, "pose_ran")
    tracked = _person_flags(session, pid, "tracking")
    out = {
        "frames": len(session.frames),
        "player_tracked_rate": float(tracked.mean()),
        "frozen_rate": float(_person_flags(session, pid, "frozen")[tracked].mean()) if tracked.any() else 0.0,
        "pose_model_runs": int(ran.sum()),
        "pose_found_rate": float(raw.present[ran].mean()) if ran.any() else float("nan"),
        "keypoints": {},
    }
    for name, j in KEYPOINTS.items():
        detected = raw.present & (raw.vis[:, j] >= VISIBLE) & _inside(session, raw.xy[:, j])
        has_sent = sent.present
        out["keypoints"][name] = {
            "detection_rate": float(detected[ran].mean()) if ran.any() else float("nan"),
            "mean_visibility": float(np.nanmean(raw.vis[ran, j])) if raw.present[ran].any() else float("nan"),
            "sent_measured_rate": float((sent.vis[has_sent, j] >= VISIBLE).mean()) if has_sent.any() else 0.0,
            "sent_inferred_rate": float((sent.vis[has_sent, j] < VISIBLE).mean()) if has_sent.any() else 0.0,
        }
    return out


def keypoint_error(session: Session, annotations: dict, shoulder_width_mm: Optional[float] = None) -> dict:
    """
    Error against clicked ground truth (annotate.py keypoints), for the raw model output and the sent pose.
    error: pixel distance where the keypoint is visible in the ground truth and a pose exists.
    mm: the error scaled by the player's measured shoulder width on that frame (2D, in the image plane;
      exact only for joints about as far from the camera as the shoulders).
    pck@a: share of ground-truth-visible keypoints that were predicted within a x shoulder width (a frame
    without a pose counts as a miss). precision/recall: the visibility flag as a detector of "the joint can
    be seen" (a confident prediction of a joint that is actually hidden is a false positive).
    """
    pid = annotations["person"]
    if shoulder_width_mm is None:
        shoulder_width_mm = session.factor("shoulder_width_mm")
    result = {"person": pid, "frames": len(annotations["frames"]), "shoulder_width_mm": shoulder_width_mm}
    for kind in ("raw", "pose"):
        track = session.track(pid, kind)
        fallback_sw = session.track(pid, "pose").shoulder_width()
        per = {name: {"err": [], "norm": [], "mm": [], "hit": {a: [] for a in PCK_THRESHOLDS},
                      "tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in KEYPOINTS}
        for i_str, points in annotations["frames"].items():
            k = session.index.get(int(i_str))
            if k is None or (kind == "raw" and not session.frames[k].get("landmark_frame", True)):
                continue
            ls, rs = points.get("left_shoulder"), points.get("right_shoulder")
            sw = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])) if ls and rs else fallback_sw
            for name, j in KEYPOINTS.items():
                if name not in points:
                    continue
                gt = points[name]
                predicted_visible = bool(track.present[k] and track.vis[k, j] >= VISIBLE)
                stats = per[name]
                if gt is None:
                    stats["fp" if predicted_visible else "tn"] += 1
                    continue
                stats["tp" if predicted_visible else "fn"] += 1
                err = float(np.hypot(*(track.xy[k, j] - np.asarray(gt)))) if track.present[k] else np.inf
                if np.isfinite(err):
                    stats["err"].append(err)
                    stats["norm"].append(100.0 * err / sw)
                    if shoulder_width_mm:
                        stats["mm"].append(err / sw * shoulder_width_mm)
                for a in PCK_THRESHOLDS:
                    stats["hit"][a].append(err <= a * sw)

        rows, pooled = {}, {"err": [], "norm": [], "mm": [], "hit": {a: [] for a in PCK_THRESHOLDS},
                            "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for name, stats in per.items():
            rows[name] = _error_row(stats)
            for key in ("err", "norm", "mm"):
                pooled[key] += stats[key]
            for a in PCK_THRESHOLDS:
                pooled["hit"][a] += stats["hit"][a]
            for c in ("tp", "fp", "fn", "tn"):
                pooled[c] += stats[c]
        rows["all"] = _error_row(pooled)
        result[kind] = rows
    return result


def _error_row(stats: dict) -> dict:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    row = {
        "gt_visible": tp + fn,
        "gt_hidden": fp + stats["tn"],
        "mean_px": float(np.mean(stats["err"])) if stats["err"] else float("nan"),
        "median_px": float(np.median(stats["err"])) if stats["err"] else float("nan"),
        "mean_pct_sw": float(np.mean(stats["norm"])) if stats["norm"] else float("nan"),
        "mean_mm": float(np.mean(stats["mm"])) if stats["mm"] else float("nan"),
        "median_mm": float(np.median(stats["mm"])) if stats["mm"] else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }
    for a in PCK_THRESHOLDS:
        hits = stats["hit"][a]
        row[f"pck@{a}"] = float(np.mean(hits)) if hits else float("nan")
    return row


# ---------------------------------------------------------------------------- 2. jitter

def _measured_segments(session: Session, raw, sent, j: int, min_length: int) -> list:
    """Frame ranges where the pose model clearly saw joint j and both raw and sent positions exist."""
    landmark = np.array([r.get("landmark_frame", True) for r in session.frames])
    ok = raw.present & sent.present & landmark & (raw.vis[:, j] >= VISIBLE)
    return segments(ok, min_length)


def jitter(session: Session, pid: int, stationary_window_s: float = 1.0, dynamic_window_s: float = 0.3,
           keypoints=None) -> dict:
    """
    Jitter of the raw and the sent (smoothed) keypoints, on frames where the joint is clearly seen.

    stationary: the player holds still, so any motion is noise. rms = RMS distance from a centred 1 s moving
      average (which removes slow body sway); step = frame-to-frame displacement.
    dynamic: the player moves, so noise is separated from motion as the residual from a zero-phase
      Savitzky-Golay fit over 0.3 s (2nd order: follows real arm motion, not frame-to-frame noise).
      lag_ms: how far the sent signal trails the raw one (cross-correlation of their velocities): the price
      paid for the smoothing.
    """
    raw, sent = session.track(pid, "raw"), session.track(pid, "pose")
    fps = frame_rate(session)
    sw = raw.shoulder_width()
    w_stat = max(3, int(round(stationary_window_s * fps)) | 1)
    w_dyn = max(5, int(round(dynamic_window_s * fps)) | 1)
    out = {"fps": fps, "shoulder_width_px": sw, "keypoints": {}}

    for name in keypoints or KEYPOINTS:
        j = KEYPOINTS[name]
        acc = {kind: {"dev": [], "step": [], "resid": [], "speed": []} for kind in ("raw", "sent")}
        vel = {"raw": [], "sent": []}
        frames = 0
        for a, b in _measured_segments(session, raw, sent, j, w_stat):
            frames += b - a
            half = w_stat // 2
            for kind, track in (("raw", raw), ("sent", sent)):
                p = track.xy[a:b, j]
                trend = uniform_filter1d(p, size=w_stat, axis=0, mode="nearest")
                acc[kind]["dev"].append(np.linalg.norm(p - trend, axis=1)[half:-half or None])
                acc[kind]["step"].append(np.linalg.norm(np.diff(p, axis=0), axis=1))
                if b - a >= w_dyn:
                    smooth = savgol_filter(p, w_dyn, 2, axis=0, mode="interp")
                    edge = w_dyn // 2
                    acc[kind]["resid"].append(np.linalg.norm(p - smooth, axis=1)[edge:-edge or None])
                    v = np.gradient(smooth, axis=0) * fps
                    acc[kind]["speed"].append(np.linalg.norm(v, axis=1))
                    vel[kind].append(v)
        row = {"frames": frames}
        for kind in ("raw", "sent"):
            dev = np.concatenate(acc[kind]["dev"]) if acc[kind]["dev"] else np.array([])
            step = np.concatenate(acc[kind]["step"]) if acc[kind]["step"] else np.array([])
            resid = np.concatenate(acc[kind]["resid"]) if acc[kind]["resid"] else np.array([])
            speed = np.concatenate(acc[kind]["speed"]) if acc[kind]["speed"] else np.array([])
            rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")
            row[kind] = {
                "stationary_rms_px": rms(dev),
                "stationary_rms_pct_sw": 100.0 * rms(dev) / sw,
                "step_rms_px": rms(step),
                "step_p95_px": float(np.percentile(step, 95)) if step.size else float("nan"),
                "dynamic_resid_rms_px": rms(resid),
                "dynamic_resid_rms_pct_sw": 100.0 * rms(resid) / sw,
                "mean_speed_px_s": float(speed.mean()) if speed.size else float("nan"),
            }
        for metric in ("stationary_rms_px", "dynamic_resid_rms_px"):
            r, s = row["raw"][metric], row["sent"][metric]
            row[f"reduction_{metric.split('_')[0]}_pct"] = 100.0 * (1.0 - s / r) if r and np.isfinite(r) else float("nan")
        row["lag_ms"] = _lag_ms(vel["raw"], vel["sent"], fps)
        out["keypoints"][name] = row
    return out


def _lag_ms(raw_vel: List[np.ndarray], sent_vel: List[np.ndarray], fps: float, max_lag_s: float = 0.3) -> float:
    """Delay (ms) that best aligns the sent velocity with the raw velocity, over the measured segments."""
    if not raw_vel:
        return float("nan")
    scores = []
    for lag in range(0, int(max_lag_s * fps) + 1):
        num = den_a = den_b = 0.0
        for a, b in zip(raw_vel, sent_vel):
            if len(a) <= lag + 2:
                continue
            x, y = a[:len(a) - lag], b[lag:]
            num += float(np.sum(x * y))
            den_a += float(np.sum(x * x))
            den_b += float(np.sum(y * y))
        scores.append(num / np.sqrt(den_a * den_b) if den_a > 0 and den_b > 0 else -np.inf)
    scores = np.array(scores)
    if not np.isfinite(scores).any():
        return float("nan")
    best = int(np.argmax(scores))
    offset = 0.0
    if 0 < best < len(scores) - 1 and np.isfinite(scores[best - 1:best + 2]).all():
        # Parabola through the peak and its neighbours: sub-frame resolution
        a, b, c = scores[best - 1:best + 2]
        offset = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c != 0 else 0.0
    return 1000.0 * (best + offset) / fps


# ---------------------------------------------------------------------------- 3. motion / gesture delay

def gesture_delay(session: Session, annotations: dict, window_s: float = 1.5, move_threshold: float = 0.1,
                  camera_latency_ms: Optional[float] = None) -> dict:
    """
    Time from an annotated physical onset (annotate.py events) to the first frame whose sent signal shows it.

    close/open: the hand's sent "closed" flag flips.  sign: a hand sign is reported.
    move: the sent wrist has left its position before the onset by move_threshold x shoulder width
      (the same is measured on the raw model output, to separate the smoothing's share).
    algorithmic_ms: frame-time difference (debouncing, hysteresis, smoothing; holds for replays too).
    end_to_end_ms (live sessions): from the camera delivering the onset frame to the packet with the
      response being sent. total_ms adds the camera's own latency (camera_latency.py) when given.
    false_triggers: flips of the closed flag that no annotated event explains.
    """
    pid = annotations["person"]
    t, t_capture, t_send = session.column("t"), session.column("t_capture"), session.column("t_send")
    sent, raw = session.track(pid, "pose"), session.track(pid, "raw")
    sw = sent.shoulder_width()
    hands = {side: session.hands(pid, key) for side, key in HAND_KEYS.items()}
    window = window_s

    results = []
    for event in annotations["events"]:
        k0 = session.index.get(event["i"])
        if k0 is None:
            continue
        sides = list(HAND_KEYS) if event["hand"] == "either" else [event["hand"]]
        responses = []  # (frame index of the response, side)
        raw_response = None
        for side in sides:
            if event["type"] in ("close", "open"):
                want = event["type"] == "close"
                state = hands[side]["closed"]
                if k0 > 0 and state[k0 - 1] == want:
                    continue  # already in that state before the onset: nothing to detect for this hand
                k = _first(lambda k: state[k] == want, k0, t, window)
            elif event["type"] == "sign":
                gesture = hands[side]["gesture"]
                k = _first(lambda k: gesture[k] != "None", k0, t, window)
            else:  # move
                k = _moved(sent, KEYPOINTS[WRIST_OF_HAND[side]], k0, t, window, move_threshold * sw)
                kr = _moved(raw, KEYPOINTS[WRIST_OF_HAND[side]], k0, t, window, move_threshold * sw)
                if kr is not None and (raw_response is None or kr < raw_response):
                    raw_response = kr
            if k is not None:
                responses.append((k, side))
        entry = {"i": event["i"], "type": event["type"], "hand": event["hand"]}
        if not responses:
            entry["detected"] = False
        else:
            k, side = min(responses)
            entry.update({
                "detected": True, "response_i": int(session.frames[k]["i"]), "response_hand": side,
                "delay_frames": int(k - k0), "algorithmic_ms": 1000.0 * float(t[k] - t[k0]),
            })
            if session.is_live:
                entry["end_to_end_ms"] = 1000.0 * float(t_send[k] - t_capture[k0])
                if camera_latency_ms is not None:
                    entry["total_ms"] = entry["end_to_end_ms"] + camera_latency_ms
            if raw_response is not None:
                entry["raw_algorithmic_ms"] = 1000.0 * float(t[raw_response] - t[k0])
        results.append(entry)

    summary = {}
    for kind in ("close", "open", "sign", "move"):
        events = [e for e in results if e["type"] == kind]
        if not events:
            continue
        hit = [e for e in events if e["detected"]]
        row = {"events": len(events), "detected_rate": len(hit) / len(events)}
        for key in ("delay_frames", "algorithmic_ms", "raw_algorithmic_ms", "end_to_end_ms", "total_ms"):
            values = [e[key] for e in hit if key in e]
            if values:
                row[key] = summarize(values)
        summary[kind] = row

    summary["false_triggers"] = _false_triggers(session, annotations, hands, t, window)
    return {"person": pid, "camera_latency_ms": camera_latency_ms, "summary": summary, "events": results}


def _first(condition, k0: int, t: np.ndarray, window: float) -> Optional[int]:
    k = k0
    while k < len(t) and t[k] - t[k0] <= window:
        if condition(k):
            return k
        k += 1
    return None


def _moved(track, j: int, k0: int, t: np.ndarray, window: float, threshold: float) -> Optional[int]:
    before = track.xy[max(0, k0 - 5):k0 + 1, j]
    before = before[np.isfinite(before).all(axis=1)]
    if not len(before):
        return None
    start = before.mean(axis=0)
    return _first(lambda k: bool(np.isfinite(track.xy[k, j]).all()
                                 and np.hypot(*(track.xy[k, j] - start)) > threshold), k0, t, window)


def _false_triggers(session, annotations, hands, t, window) -> dict:
    """Closed-flag flips with no annotated close/open onset of the same direction in the preceding window."""
    onsets = {kind: [session.index[e["i"]] for e in annotations["events"]
                     if e["type"] == kind and e["i"] in session.index] for kind in ("close", "open")}
    if not onsets["close"] and not onsets["open"]:
        return {"note": "no close/open events annotated"}
    count = 0
    for state in (h["closed"] for h in hands.values()):
        for k in np.flatnonzero(state[1:] != state[:-1]) + 1:
            kind = "close" if state[k] else "open"
            if not any(0 <= t[k] - t[k0] <= window for k0 in onsets[kind]):
                count += 1
    minutes = (t[-1] - t[0]) / 60.0
    return {"count": count, "per_minute": count / minutes if minutes > 0 else float("nan")}


# ---------------------------------------------------------------------------- 4. missing keypoints (as recorded)

def missing_keypoints(session: Session, pid: int) -> dict:
    """
    How often keypoints were missing in this session and how the handling behaved.
    whole_pose_dropout: the player is tracked but no pose was sent. per keypoint: share of sent frames where the
    joint was inferred, the length of those gaps, and the "pop" when it is measured again (the jump between
    the last inferred and first measured position) next to the 95th percentile of normal frame-to-frame steps.
    """
    sent, raw = session.track(pid, "pose"), session.track(pid, "raw")
    t = session.column("t")
    sw = raw.shoulder_width()
    tracked = _person_flags(session, pid, "tracking")
    ran = _person_flags(session, pid, "pose_ran")
    dropout = tracked & ~sent.present
    out = {
        "frames_tracked": int(tracked.sum()),
        "whole_pose_dropout_rate": float(dropout[tracked].mean()) if tracked.any() else float("nan"),
        "whole_pose_dropout_s": summarize([t[min(b, len(t) - 1)] - t[a] for a, b in segments(dropout)]),
        "pose_not_found_rate": float((~raw.present[ran]).mean()) if ran.any() else float("nan"),
        "keypoints": {},
    }
    for name, j in KEYPOINTS.items():
        inferred = sent.present & (sent.vis[:, j] < VISIBLE)
        measured = sent.present & (sent.vis[:, j] >= VISIBLE)
        gaps = segments(inferred)
        steps = np.linalg.norm(np.diff(sent.xy[:, j], axis=0), axis=1)
        normal = steps[measured[1:] & measured[:-1]]
        pops = [steps[b - 1] for a, b in gaps if b < len(t) and measured[b] and inferred[b - 1]]
        out["keypoints"][name] = {
            "inferred_rate": float(inferred[sent.present].mean()) if sent.present.any() else float("nan"),
            "gaps": len(gaps),
            "gap_s": summarize([t[min(b, len(t) - 1)] - t[a] for a, b in gaps]),
            "reentry_pop_pct_sw": summarize(100.0 * np.asarray(pops) / sw),
            "normal_step_p95_pct_sw": 100.0 * float(np.percentile(normal, 95)) / sw if normal.size else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------- 5. module latency / rate

def fps_stats(times: np.ndarray) -> dict:
    """
    Frame rate from output timestamps (s). "x% low" = the average rate over the slowest x% of frame
    intervals (the usual games measure, it shows stutter that the mean hides); min_1s = the slowest
    one-second window; below_25 = share of one-second windows under 25 fps.
    """
    times = times[np.isfinite(times)]
    if len(times) < 2:
        return {"n": 0}
    interval = np.diff(times)
    slowest = np.sort(interval)[::-1]
    out = {"n": len(interval), "mean": len(interval) / float(times[-1] - times[0])}
    for share, key in ((0.05, "low_5pct"), (0.01, "low_1pct")):
        out[key] = 1.0 / float(np.mean(slowest[:max(1, int(round(share * len(slowest))))]))
    edges = np.arange(times[0], times[-1] - 1.0 + 1e-9, 0.5)  # 1 s windows every 0.5 s
    if len(edges):
        per_window = np.searchsorted(times, edges + 1.0) - np.searchsorted(times, edges)
        out["min_1s"] = float(per_window.min())
        out["below_25"] = float((per_window < 25).mean())
    return out


def latency(session: Session, warmup_s: float = 1.0) -> dict:
    """
    end_to_end_ms: camera delivering a frame -> its UDP packet sent (includes waiting for the loop).
    queue_wait_ms: camera delivering a frame -> the loop picking it up.
    stages_ms: time per pipeline stage on the frames where that stage really ran.
    output_rate_hz: packets per second; interval_ms: time between packets (its spread is the rate jitter).
    output_fps: mean, 5% / 1% low and slowest-second rate of the packets (fps_stats).
    camera_rate_hz: frames the camera delivered per second (processed + dropped as stale).
    pose_rate_hz / hand_rate_hz: how often a player's pose / each hand's state is actually re-measured.
    The first warmup_s seconds are left out: the models' first runs are much slower (initialisation).
    """
    t_send_all = session.column("t_send")
    start = int(np.searchsorted(t_send_all, t_send_all[0] + warmup_s))
    if start >= len(t_send_all) - 1:
        start = 0
    frames = session.frames[start:]
    column = lambda key: np.array([r.get(key, np.nan) for r in frames], dtype=float)
    t_capture, t_read, t_send = column("t_capture"), column("t_read"), column("t_send")
    duration = float(t_send[-1] - t_send[0]) if len(t_send) > 1 else float("nan")
    dropped = column("dropped")
    n = len(frames)

    stages: Dict[str, list] = {}
    for record in frames:
        for name, ms in record.get("stages", {}).items():
            if name == "Person tracker":
                name += " (detector ran)" if record.get("detector_ran") else " (predicted)"
            stages.setdefault(name, []).append(ms)

    out = {
        "live": session.is_live,
        "preview": session.meta.get("preview"),
        "duration_s": duration,
        "frames": n,
        "warmup_frames_skipped": start,
        "end_to_end_ms": summarize(1000.0 * (t_send - t_capture)),
        "queue_wait_ms": summarize(1000.0 * (t_read - t_capture)),
        "processing_ms": summarize(1000.0 * (t_send - t_read)),
        "interval_ms": summarize(1000.0 * np.diff(t_send)),
        "output_rate_hz": (n - 1) / duration if duration > 0 else float("nan"),
        "output_fps": fps_stats(t_send),
        "camera_rate_hz": (n - 1 + dropped[-1] - dropped[0]) / float(t_capture[-1] - t_capture[0])
        if n > 1 and t_capture[-1] > t_capture[0] else float("nan"),
        "dropped_frames": int(dropped[-1] - dropped[0]) if n else 0,
        "stages_ms": {name: summarize(values) for name, values in sorted(stages.items())},
    }
    pid = session.main_person()
    if pid is not None and duration > 0:
        raw_present = session.track(pid, "raw").present[start:]
        gestures = np.array([r.get("gestures_ran", False) for r in frames]) & raw_present
        out["pose_rate_hz"] = float(raw_present.sum()) / duration
        # The hand recognizer alternates between the two hands, so each hand is re-read every other run
        out["hand_rate_hz"] = float(gestures.sum()) / duration / 2.0
    return out


# ---------------------------------------------------------------------------- 6. lighting and other factors

FACTORS = ("condition", "subject", "distance", "lux", "brightness", "noise")


def factor_value(session: Session, name: str):
    """A session's value of one impact factor (see compare.py --by)."""
    if name == "condition":
        return session.condition
    if name in ("brightness", "noise"):
        return (session.meta.get("lighting") or {}).get(name, 1.0 if name == "brightness" else 0.0)
    return session.factor({"distance": "distance_m"}.get(name, name))


def session_summary(session: Session, pid: Optional[int] = None, reference: Optional[Session] = None) -> dict:
    """
    One row per session for comparing conditions (lighting, distance, subject, ...): brightness, detection,
    hand recognition, jitter, frame rate, error against ground truth (if annotated) and, for a replay with
    simulated lighting, the deviation from the reference (normally lit) replay of the same video.
    """
    pid = session.main_person() if pid is None else pid
    luma = session.column("luma")
    row = {"session": session.name, "condition": session.condition, "lighting": session.meta.get("lighting"),
           "factors": {name: factor_value(session, name) for name in FACTORS},
           "luma_mean": float(np.nanmean(luma)), "luma_std": float(np.nanstd(luma)), "person": pid}
    fps = latency(session)["output_fps"]
    row["fps_mean"], row["fps_low_1pct"] = fps.get("mean", float("nan")), fps.get("low_1pct", float("nan"))
    if pid is None:
        row["note"] = "no player tracked"
        return row

    det = detection_rates(session, pid)
    row["player_tracked_rate"] = det["player_tracked_rate"]
    row["pose_found_rate"] = det["pose_found_rate"]
    row["detection_rate_mean"] = float(np.mean([k["detection_rate"] for k in det["keypoints"].values()]))
    row["wrist_detection_rate"] = float(np.mean([det["keypoints"][k]["detection_rate"] for k in CRITICAL_KEYPOINTS]))
    row["wrist_visibility"] = float(np.mean([det["keypoints"][k]["mean_visibility"] for k in CRITICAL_KEYPOINTS]))

    ran = np.array([r.get("gestures_ran", False) for r in session.frames]) & session.track(pid, "pose").present
    seen = [session.hands(pid, key)["seen"][ran] for key in HAND_KEYS.values()]
    row["hand_seen_rate"] = float(np.mean([s.mean() for s in seen])) if ran.any() else float("nan")

    jit = jitter(session, pid, keypoints=CRITICAL_KEYPOINTS)["keypoints"]
    for kind in ("raw", "sent"):
        values = np.array([jit[k][kind]["stationary_rms_pct_sw"] for k in CRITICAL_KEYPOINTS], dtype=float)
        row[f"wrist_jitter_{kind}_pct_sw"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    annotations = session.annotations("annotations_keypoints.json")
    if annotations:
        err = keypoint_error(session, annotations)
        for kind in ("raw", "pose"):
            row[f"gt_{kind}_pck@0.2"] = err[kind]["all"]["pck@0.2"]
            row[f"gt_{kind}_mean_pct_sw"] = err[kind]["all"]["mean_pct_sw"]
            row[f"gt_{kind}_mean_mm"] = err[kind]["all"]["mean_mm"]

    if reference is not None:
        row["vs_reference"] = deviation_from(session, reference, pid)
    return row


def deviation_from(session: Session, reference: Session, pid: int) -> dict:
    """Distance of this session's keypoints from the reference session's on the same frames (%SW)."""
    out = {}
    for kind in ("raw", "pose"):
        a, b = session.track(pid, kind), reference.track(pid, kind)
        sw = b.shoulder_width()
        ka, kb = [], []
        for k, record in enumerate(session.frames):
            kr = reference.index.get(record["i"])
            if kr is not None and a.present[k] and b.present[kr]:
                ka.append(k)
                kb.append(kr)
        if not ka:
            out[kind] = {"frames": 0}
            continue
        joints = list(KEYPOINTS.values())
        d = np.linalg.norm(a.xy[ka][:, joints] - b.xy[kb][:, joints], axis=2)
        ok = b.vis[kb][:, joints] >= VISIBLE  # only where the reference itself saw the joint
        lost = sum(1 for k in range(len(reference.frames)) if b.present[k]) - len(kb)
        out[kind] = {
            "frames": len(ka),
            "pose_lost_frames": int(lost),
            "median_pct_sw": 100.0 * float(np.median(d[ok])) / sw if ok.any() else float("nan"),
            "within_0.2sw": float((d[ok] <= 0.2 * sw).mean()) if ok.any() else float("nan"),
        }
    return out
