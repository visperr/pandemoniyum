"""
Synthetic evaluation of the missing-keypoint handling in core/pose_smoother.py.

Real occlusions have no ground truth: nobody knows where a hidden hand really was. So this takes stretches
of a recorded session where the joints were clearly seen, hides them for a known time (visibility 0, which
is what the smoother gets for a joint that left the image or was occluded), replays the recorded pose-model
output through the PoseSmoother, and compares the filled-in positions with where the joint really was
(the same replay without hiding anything).

    python -m evaluation.missing_keypoints SESSION [--gaps 0.1 0.25 0.5 1 2] [--plots]

Scenarios: "hand" hides a wrist and its hand points, "arm" also the elbow, "pose" drops the whole pose
(the pose model finding nothing). Compared methods:
  pipeline          the smoother as configured (wrists hold, elbows follow the learned bone length)
  relax             the same, but unseen wrists relax towards hanging down (POSE_HOLD_UNSEEN_HANDS = False)
  hold-last         baseline: freeze the joint at its last measured position
  const-velocity    baseline: extrapolate the last measured velocity
  drop              baseline: no handling; the joint is simply unavailable (availability 0)
Writes SESSION/missing_keypoints.md and .json. Use a session with real arm movement (dynamic protocol).
"""

import argparse
import json

import numpy as np

import config
from core.pose_smoother import PoseSmoother
from evaluation.report import fmt, table
from evaluation.session import L_SHOULDER, R_SHOULDER, VISIBLE, Session
from models.pose_data import PoseLandmark, PoseTrackingData

SCENARIOS = {
    # name -> per side: (hidden joints, evaluated joints)
    "hand": {"left": ([15, 17, 19, 21], [15]), "right": ([16, 18, 20, 22], [16])},
    "arm": {"left": ([13, 15, 17, 19, 21], [13, 15]), "right": ([14, 16, 18, 20, 22], [14, 16])},
    "pose": {"all": ("all", [0, 11, 12, 13, 14, 15, 16, 23, 24])},  # scored on those clearly seen
}
CLEAR = 0.8          # joints must be seen at least this confidently around a synthetic gap
PRE_S, POST_S = 0.5, 1.0
RECOVERED = 0.05     # error below 5% of the shoulder width counts as recovered after a gap


def make_smoother(session: Session, **overrides) -> PoseSmoother:
    """A PoseSmoother with the settings the session was recorded with (current config where missing)."""
    def cfg(name):
        return session.config(name, getattr(config, name))
    params = dict(
        enabled=True, min_cutoff=cfg("POSE_MIN_CUTOFF"), beta=cfg("POSE_BETA"), speed_floor=cfg("POSE_SPEED_FLOOR"),
        vis_low=cfg("POSE_VISIBILITY_LOW"), vis_high=cfg("POSE_VISIBILITY_HIGH"),
        rest_blend_seconds=cfg("POSE_REST_BLEND_SECONDS"), dropout_hold_seconds=cfg("POSE_DROPOUT_HOLD_SECONDS"),
        reset_seconds=cfg("POSE_RESET_SECONDS"), hold_unseen_hands=cfg("POSE_HOLD_UNSEEN_HANDS"),
        max_forearm_stretch=cfg("POSE_MAX_FOREARM_STRETCH"), max_hand_speed=cfg("POSE_MAX_HAND_SPEED"),
    )
    params.update(overrides)
    return PoseSmoother(**params)


def replay(session: Session, pid: int, smoother: PoseSmoother, mask=None):
    """
    Feed the recorded smoother inputs through `smoother`, hiding mask[k] (a joint list, or "all" for no
    pose) on frame k. Returns the output (F, 25, 2) pixels and (F, 25) visibility, NaN where nothing was sent.
    """
    mask = mask or {}
    n = len(session.frames)
    xy, vis = np.full((n, 25, 2), np.nan), np.full((n, 25), np.nan)
    for k, record in enumerate(session.frames):
        person = session.person(record, pid)
        if person is None or not person.get("pose_ran") or not person.get("crop"):
            continue
        hidden = mask.get(k, ())
        pose = None
        if person.get("raw") and hidden != "all":
            landmarks = [PoseLandmark(r[0], r[1], r[2], r[3], r[4] if len(r) > 4 else r[3]) for r in person["raw"]]
            for j in hidden:
                landmarks[j].visibility = landmarks[j].presence = 0.0
            pose = PoseTrackingData(landmarks=landmarks, tracking_active=True)
        box = tuple(person["crop"])
        out = smoother.update(pid, pose, box, (record["h"], record["w"]), record["t"])
        if out.is_tracking:
            x1, y1, x2, y2 = box
            rows = np.array([[lm.x, lm.y, lm.visibility] for lm in out.landmarks])
            xy[k, :, 0] = x1 + rows[:, 0] * (x2 - x1)
            xy[k, :, 1] = y1 + rows[:, 1] * (y2 - y1)
            vis[k] = rows[:, 2]
    return xy, vis


def place_gaps(session: Session, pid: int, joints, length_s: float) -> list:
    """Non-overlapping (start, end) frame ranges of length_s where `joints` were clearly seen around them."""
    raw = session.track(pid, "raw")
    t = session.column("t")
    clear = raw.present & (raw.vis[:, joints] >= CLEAR).all(axis=1)
    gaps, k = [], 0
    while k < len(t):
        start = int(np.searchsorted(t, t[k] + PRE_S))
        end = int(np.searchsorted(t, t[min(start, len(t) - 1)] + length_s))
        stop = int(np.searchsorted(t, t[min(end, len(t) - 1)] + POST_S))
        if stop >= len(t):
            break
        if clear[k:stop].all():
            gaps.append((start, end))
            # Leave time for the smoother to settle before the next gap
            k = int(np.searchsorted(t, t[stop] + 1.0))
        else:
            k += 1
    return gaps


def evaluate(session: Session, pid: int, lengths, max_gaps: int = 30) -> dict:
    t = session.column("t")
    reference_xy, reference_vis = replay(session, pid, make_smoother(session))
    recorded = session.track(pid, "pose")
    raw = session.track(pid, "raw")
    sw = raw.shoulder_width()
    both = recorded.present & np.isfinite(reference_xy[:, 0, 0])
    fidelity = float(np.nanmax(np.linalg.norm(reference_xy[both] - recorded.xy[both], axis=2))) if both.any() else None

    results = {"session": session.name, "person": pid, "shoulder_width_px": sw,
               "replay_fidelity_max_px": fidelity, "scenarios": {}}
    for scenario, sides in SCENARIOS.items():
        for length in lengths:
            per_method = {m: [] for m in ("pipeline", "relax", "hold-last", "const-velocity", "drop")}
            n_gaps = 0
            for side, (hidden, candidates) in sides.items():
                # A dropped pose is scored on the joints that were clearly seen around it (needs the shoulders)
                check = [L_SHOULDER, R_SHOULDER] if hidden == "all" else candidates
                gaps = place_gaps(session, pid, check, length)[:max_gaps]
                if not gaps:
                    continue
                n_gaps += len(gaps)
                mask = {k: hidden for a, b in gaps for k in range(a, b)}
                runs = {"pipeline": replay(session, pid, make_smoother(session), mask)}
                if scenario != "pose":
                    runs["relax"] = replay(session, pid, make_smoother(session, hold_unseen_hands=False), mask)
                for a, b in gaps:
                    stop = int(np.searchsorted(t, t[b - 1] + POST_S))
                    evaluated = [j for j in candidates if (raw.vis[a - 1:stop, j] >= CLEAR).all()]
                    truth = reference_xy[a:b][:, evaluated]
                    for method, (xy, vis) in runs.items():
                        per_method[method].append(_gap_scores(xy, vis, reference_xy, evaluated, a, b, stop, t, sw))
                    last = reference_xy[a - 1, evaluated]
                    per_method["hold-last"].append(_baseline_scores(np.broadcast_to(last, truth.shape), truth, sw))
                    k_prev = int(np.searchsorted(t, t[a - 1] - 0.1))
                    velocity = (last - reference_xy[k_prev, evaluated]) / max(t[a - 1] - t[k_prev], 1e-3)
                    guess = last[None] + velocity[None] * (t[a:b] - t[a - 1])[:, None, None]
                    per_method["const-velocity"].append(_baseline_scores(guess, truth, sw))
                    per_method["drop"].append({"availability": 0.0})
            results["scenarios"].setdefault(scenario, {})[str(length)] = {
                "gaps": n_gaps,
                "methods": {m: _aggregate(scores) for m, scores in per_method.items() if scores},
            }
    return results


def _gap_scores(xy, vis, reference, joints, a, b, stop, t, sw) -> dict:
    err = np.linalg.norm(xy[a:b][:, joints] - reference[a:b][:, joints], axis=2)
    available = np.isfinite(err).all(axis=1)
    after = np.linalg.norm(xy[b:stop][:, joints] - reference[b:stop][:, joints], axis=2).max(axis=1)
    recovered = np.flatnonzero(np.nan_to_num(after, nan=np.inf) < RECOVERED * sw)
    steps = np.linalg.norm(np.diff(xy[b - 1:stop][:, joints], axis=0), axis=2)
    return {
        "availability": float(available.mean()),
        "mean_err_pct_sw": 100.0 * float(np.nanmean(err)) / sw if available.any() else np.nan,
        "end_err_pct_sw": 100.0 * float(np.nanmean(err[-1])) / sw if available[-1] else np.nan,
        "flagged_inferred": float((vis[a:b][:, joints] < VISIBLE)[available].mean()) if available.any() else np.nan,
        "recovery_ms": 1000.0 * float(t[b + recovered[0]] - t[b - 1]) if recovered.size else np.nan,
        "reentry_pop_pct_sw": 100.0 * float(np.nanmax(steps)) / sw if np.isfinite(steps).any() else np.nan,
    }


def _baseline_scores(guess, truth, sw) -> dict:
    err = np.linalg.norm(guess - truth, axis=2)
    return {"availability": 1.0, "mean_err_pct_sw": 100.0 * float(err.mean()) / sw,
            "end_err_pct_sw": 100.0 * float(err[-1].mean()) / sw}


def _aggregate(scores: list) -> dict:
    out = {}
    for key in scores[0]:
        values = np.array([s.get(key, np.nan) for s in scores], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            out[key] = {"median": float(np.median(values)), "mean": float(values.mean()),
                        "p95": float(np.percentile(values, 95))}
    return out


def to_markdown(results: dict) -> str:
    def cell(method_scores, key, digits=1, pct=False):
        s = method_scores.get(key)
        return fmt(s["median"], digits, pct) if s else "–"

    parts = [f"# Missing-keypoint handling: {results['session']}\n\n"
             f"Synthetic gaps on clearly tracked stretches, errors against the unmasked replay in % of the shoulder "
             f"width ({fmt(results['shoulder_width_px'])} px), medians over gaps. Replay reproduces the recorded "
             f"output to within {fmt(results['replay_fidelity_max_px'], 2)} px."]
    for scenario, by_length in results["scenarios"].items():
        rows = []
        for length, entry in by_length.items():
            for method, scores in entry["methods"].items():
                rows.append([f"{length} s", entry["gaps"], method, cell(scores, "availability", 0, True),
                             cell(scores, "mean_err_pct_sw"), cell(scores, "end_err_pct_sw"),
                             cell(scores, "flagged_inferred", 0, True), cell(scores, "recovery_ms", 0),
                             cell(scores, "reentry_pop_pct_sw")])
        parts.append(f"## Scenario: {scenario}\n\n" + table(
            ["Gap", "Gaps", "Method", "Available", "Mean error %SW", "Error at gap end %SW", "Flagged inferred",
             "Recovery ms", "Re-entry pop %SW"], rows))
    return "\n\n".join(parts) + "\n"


def plot(results: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results["scenarios"]), figsize=(5 * len(results["scenarios"]), 4), squeeze=False)
    for ax, (scenario, by_length) in zip(axes[0], results["scenarios"].items()):
        lengths = sorted(by_length, key=float)
        methods = sorted({m for e in by_length.values() for m in e["methods"] if m != "drop"})
        for method in methods:
            ys = [by_length[l]["methods"].get(method, {}).get("mean_err_pct_sw", {}).get("median", np.nan)
                  for l in lengths]
            ax.plot([float(l) for l in lengths], ys, marker="o", label=method)
        ax.set_title(scenario)
        ax.set_xlabel("gap length (s)")
        ax.set_ylabel("mean error (% shoulder width)")
    if axes[0][0].lines:
        axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session")
    parser.add_argument("--person", type=int)
    parser.add_argument("--gaps", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0], help="gap lengths (s)")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    session = Session(args.session)
    pid = args.person if args.person is not None else session.main_person()
    results = evaluate(session, pid, args.gaps)
    markdown = to_markdown(results)
    (session.path / "missing_keypoints.md").write_text(markdown)
    with open(session.path / "missing_keypoints.json", "w") as f:
        json.dump(results, f, indent=1)
    print(markdown)
    if args.plots:
        plot(results, session.path / "plot_missing_keypoints.png")
        print(f"Saved {session.path / 'plot_missing_keypoints.png'}")


if __name__ == "__main__":
    main()
