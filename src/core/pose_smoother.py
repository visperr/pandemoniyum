"""
Temporal smoothing and off-screen limb handling for MediaPipe pose landmarks.

What it does, per tracked person:

1. Works in *frame* space. MediaPipe returns landmarks normalised to the person crop, and the
   crop itself jitters/resizes with the detector box. Filtering in crop space would smooth
   against a moving reference, so landmarks are converted to frame space, filtered, and
   converted back to the current crop before being sent (the UDP format does not change).
2. Filters every joint with a One Euro filter: heavy smoothing while a joint is nearly still
   (removes jitter), little smoothing while it moves fast (no lag).
3. Gates every joint by confidence (MediaPipe visibility/presence, plus "is it outside the
   frame"). MediaPipe still outputs a position for joints that are off-screen, but it is a
   guess. Low-confidence joints are replaced by a skeleton-consistent estimate: the child joint
   is placed at the parent joint plus the learned bone length, pointing in its last known
   direction and relaxing towards "hanging down" over time. The switch between measured and
   inferred is cross-faded, so joints do not pop when they enter or leave the frame.
4. Reports the gated confidence as `visibility`, so Unity can fade IK/animation weights for
   joints that are inferred rather than seen.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np

from models.pose_data import PoseLandmark, PoseTrackingData, POSE_LANDMARK_COUNT

N = POSE_LANDMARK_COUNT  # landmarks 0-24: face, arms, torso, hips (no legs)

L_SHOULDER, R_SHOULDER = 11, 12

# (child, parent, bone-length key). Parents are always listed before their children.
CHAIN = [
    (23, 11, "torso"), (24, 12, "torso"),
    (13, 11, "upper_arm"), (14, 12, "upper_arm"),
    (15, 13, "forearm"), (16, 14, "forearm"),
]

# Bone length priors as a multiple of shoulder width, used until a bone has been measured.
LENGTH_PRIOR = {"torso": 1.3, "upper_arm": 0.8, "forearm": 0.65}

# Joints that rigidly follow an anchor joint when they are not measured:
# face points follow the shoulder midpoint, hand points the wrist.
FACE = list(range(0, 11))
APPENDAGES = [
    (15, [17, 19, 21]), (16, [18, 20, 22]),
]

TORSO_JOINTS = [11, 12, 23, 24]


class OneEuroFilter:
    """
    Vectorised One Euro filter (Casiez, Roussel, Vogel; CHI 2012) over an (N, 3) array.

    The cutoff frequency grows with the joint's xy speed, so slow motion is smoothed
    strongly and fast motion is tracked closely. Timestamps may be irregular.
    """

    def __init__(self, min_cutoff: np.ndarray, beta: np.ndarray, d_cutoff: float = 1.0,
                 speed_floor: float = 0.0):
        self.min_cutoff = min_cutoff   # (N, 3), Hz
        self.beta = beta               # (N,), Hz per (person heights / second)
        self.d_cutoff = d_cutoff
        self.speed_floor = speed_floor  # speeds below this (person heights/s) count as noise
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float, speed_scale: float = 1.0) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Isotropic: one speed per joint (xy), so smoothing does not distort the direction.
        speed = np.linalg.norm(dx_hat[:, :2], axis=1) / speed_scale
        speed = np.maximum(speed - self.speed_floor, 0.0)
        cutoff = self.min_cutoff + (self.beta * speed)[:, None]
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat.copy()

    def seed(self, x: np.ndarray, mask: np.ndarray) -> None:
        """Force the state of the masked joints to `x` (zero velocity)."""
        if self.x_prev is None:
            return
        self.x_prev[mask] = x[mask]
        self.dx_prev[mask] = 0.0


class _PersonState:
    def __init__(self, filt: OneEuroFilter):
        self.filter = filt
        self.out: Optional[np.ndarray] = None   # (N, 3), frame-height units
        self.vis: Optional[np.ndarray] = None   # (N,)
        self.t: Optional[float] = None          # time of last successful update
        self.scale: Optional[float] = None      # person height, frame-height units (slow average)
        self.lengths: Dict[str, float] = {}
        self.shoulder_ref: Optional[float] = None
        self.offsets: Dict[int, np.ndarray] = {}


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-6 else fallback


def _slow_max(current: Optional[float], observed: float) -> float:
    """
    Track a length that is only ever *under*-estimated by a 2D projection (foreshortening):
    grow quickly towards larger observations, shrink very slowly, ignore spikes.
    """
    if current is None:
        return observed
    observed = min(observed, 1.3 * current)
    rate = 0.2 if observed > current else 0.01
    return current + rate * (observed - current)


class PoseSmoother:
    def __init__(
        self,
        enabled: bool = True,
        min_cutoff: float = 0.6,
        beta: float = 15.0,
        z_cutoff_factor: float = 0.5,
        d_cutoff: float = 8.0,
        speed_floor: float = 0.2,
        vis_low: float = 0.35,
        vis_high: float = 0.65,
        edge_margin: float = 0.05,
        rest_blend_seconds: float = 1.5,
        dropout_hold_seconds: float = 0.4,
        reset_seconds: float = 1.0,
    ):
        """
        min_cutoff / beta   One Euro parameters for the arms. Lower min_cutoff = smoother when
                            still; higher beta = less lag when moving. Speeds are measured in
                            person heights per second, so they do not depend on camera distance.
        d_cutoff            Smoothing of the speed estimate that drives the adaptive cutoff (Hz).
                            Too low and the filter reacts late to fast motion.
        speed_floor         Joint speeds below this (person heights/s) are treated as noise and do
                            not open up the filter. Lets beta be high without passing jitter.
        vis_low / vis_high  Confidence below vis_low is treated as "not seen" (fully inferred),
                            above vis_high as fully measured, with a smooth cross-fade between.
        edge_margin         A joint this far (in frame heights) outside the frame counts as
                            off-screen regardless of the reported confidence.
        rest_blend_seconds  How long an unseen limb takes to relax from its last known
                            direction to hanging down. Larger = holds the last direction longer.
        dropout_hold_seconds How long the last pose is held if the detector returns nothing.
        reset_seconds       If a person has had no pose for this long, filters start fresh.
        """
        self.enabled = enabled
        self.d_cutoff = d_cutoff
        self.speed_floor = speed_floor
        self.vis_low = vis_low
        self.vis_high = max(vis_high, vis_low + 1e-3)
        self.edge_margin = edge_margin
        self.rest_blend_seconds = rest_blend_seconds
        self.dropout_hold_seconds = dropout_hold_seconds
        self.reset_seconds = reset_seconds

        mc = np.full(N, float(min_cutoff))
        bt = np.full(N, float(beta))
        mc[FACE] *= 1.0
        bt[FACE] *= 0.5
        mc[TORSO_JOINTS] *= 0.75
        bt[TORSO_JOINTS] *= 0.5
        self._min_cutoff = np.stack([mc, mc, mc * z_cutoff_factor], axis=1)
        self._beta = bt

        self._states: Dict[int, _PersonState] = {}
        self._raw: Dict[int, PoseTrackingData] = {}

    # ------------------------------------------------------------------ public API

    def update(self, pid: int, pose: Optional[PoseTrackingData], box: Tuple[int, int, int, int],
               frame_shape, t: float) -> PoseTrackingData:
        """
        Feed a fresh detection. `box` is the crop (x1, y1, x2, y2) in frame pixels that the pose
        model ran on, `frame_shape` the frame's (h, w, ...) and `t` the frame time in seconds.
        Returns the smoothed pose, normalised to the same crop.
        """
        has_pose = pose is not None and pose.is_tracking and len(pose.landmarks) >= N

        if not self.enabled:
            if has_pose:
                self._raw[pid] = pose
                return pose
            self._raw.pop(pid, None)
            return PoseTrackingData(tracking_active=False)

        if not has_pose:
            return self._held_or_inactive(pid, box, frame_shape, t)

        frame_h, frame_w = frame_shape[0], frame_shape[1]

        st = self._states.get(pid)
        if st is not None and st.t is not None and t - st.t > self.reset_seconds:
            st = None
        if st is None:
            st = _PersonState(OneEuroFilter(self._min_cutoff, self._beta, self.d_cutoff, self.speed_floor))
            self._states[pid] = st

        meas, conf = self._to_frame_space(pose.landmarks, box, frame_w, frame_h)

        # Joints outside the frame are extrapolated by the model, never really seen.
        aspect = frame_w / frame_h
        ox = np.maximum(0.0, np.maximum(-meas[:, 0], meas[:, 0] - aspect))
        oy = np.maximum(0.0, np.maximum(-meas[:, 1], meas[:, 1] - 1.0))
        conf = conf * np.clip(1.0 - np.hypot(ox, oy) / self.edge_margin, 0.0, 1.0)
        w = _smoothstep((conf - self.vis_low) / (self.vis_high - self.vis_low))

        dt = max(t - st.t, 1e-3) if st.t is not None else 1.0 / 30.0
        box_h = max(box[3] - box[1], 1) / frame_h
        st.scale = box_h if st.scale is None else st.scale + 0.05 * (box_h - st.scale)

        filt = st.filter(meas, t, speed_scale=max(st.scale, 1e-3))
        out = self._resolve(st, meas, filt, w, dt)

        # Joints that were not measured must not leave stale state in the filter.
        st.filter.seed(out, w < 0.05)

        st.out, st.vis, st.t = out, w, t
        return self._to_pose(out, w, box, frame_h)

    def reproject(self, pid: int, box: Tuple[int, int, int, int], frame_shape,
                  t: float) -> Optional[PoseTrackingData]:
        """
        For frames without a new detection: the latest smoothed pose placed into the person's
        current crop. Returns None if this person has no pose yet.
        """
        if not self.enabled:
            return self._raw.get(pid)

        st = self._states.get(pid)
        if st is None or st.out is None:
            return None
        if t - st.t > self.dropout_hold_seconds:
            return PoseTrackingData(tracking_active=False)
        return self._to_pose(st.out, st.vis, box, frame_shape[0])

    # ------------------------------------------------------------------ internals

    def _held_or_inactive(self, pid, box, frame_shape, t) -> PoseTrackingData:
        """The detector found nothing: briefly hold the last pose instead of flickering off."""
        st = self._states.get(pid)
        if st is None or st.out is None or t - st.t > self.dropout_hold_seconds:
            return PoseTrackingData(tracking_active=False)
        return self._to_pose(st.out, st.vis, box, frame_shape[0])

    @staticmethod
    def _to_frame_space(landmarks, box, frame_w, frame_h):
        """Crop-normalised landmarks -> (N, 3) array in frame-height units, plus confidence."""
        bx1, by1, bx2, by2 = box
        cw, ch = max(bx2 - bx1, 1), max(by2 - by1, 1)

        arr = np.empty((N, 3))
        conf = np.empty(N)
        for i in range(N):
            lm = landmarks[i]
            arr[i, 0] = bx1 + lm.x * cw
            arr[i, 1] = by1 + lm.y * ch
            arr[i, 2] = lm.z * cw
            vis = getattr(lm, "visibility", None)
            pres = getattr(lm, "presence", None)
            conf[i] = min(1.0 if vis is None else vis, 1.0 if pres is None else pres)
        return arr / frame_h, conf

    @staticmethod
    def _to_pose(out, vis, box, frame_h) -> PoseTrackingData:
        """Frame-space joints -> crop-normalised landmarks (the format Unity already expects)."""
        bx1, by1, bx2, by2 = box
        cw, ch = max(bx2 - bx1, 1), max(by2 - by1, 1)
        px = out * frame_h
        xs = (px[:, 0] - bx1) / cw
        ys = (px[:, 1] - by1) / ch
        zs = px[:, 2] / cw
        landmarks = [
            PoseLandmark(float(xs[i]), float(ys[i]), float(zs[i]), float(vis[i]), float(vis[i]))
            for i in range(N)
        ]
        return PoseTrackingData(landmarks=landmarks, tracking_active=True)

    def _length(self, st: _PersonState, key: str, out: np.ndarray) -> float:
        known = st.lengths.get(key)
        if known is not None:
            return known
        shoulder = st.shoulder_ref
        if shoulder is None:
            shoulder = float(np.hypot(*(out[L_SHOULDER, :2] - out[R_SHOULDER, :2])))
        return LENGTH_PRIOR[key] * max(shoulder, 0.05)

    def _resolve(self, st: _PersonState, meas, filt, w, dt) -> np.ndarray:
        """Blend filtered measurements with skeleton-consistent estimates, joint by joint."""
        prev = st.out
        out = np.empty_like(meas)

        def mix(j, synth):
            out[j] = w[j] * filt[j] + (1.0 - w[j]) * synth

        # Shoulders: no parent. If unseen, hold the last position.
        for j in (L_SHOULDER, R_SHOULDER):
            mix(j, prev[j] if prev is not None else meas[j])

        # "Down" in the person's own frame: the normal of the shoulder line.
        v = out[L_SHOULDER, :2] - out[R_SHOULDER, :2]
        down = np.array([0.0, 1.0])
        normal = np.array([-v[1], v[0]])
        n = float(np.hypot(*normal))
        if n > 1e-6:
            normal = normal / n
            if normal[1] < 0:
                normal = -normal
            if normal[1] > 0.5:  # shoulders level enough to trust; otherwise use image down
                down = normal

        k = 1.0 if self.rest_blend_seconds <= 0 else 1.0 - math.exp(-dt / self.rest_blend_seconds)

        # Limbs: parent -> child chains.
        for child, parent, key in CHAIN:
            p = out[parent]
            length = self._length(st, key, out)
            last_dir = (prev[child, :2] - p[:2]) if prev is not None else np.zeros(2)
            last_dir = _unit(last_dir, down)
            direction = _unit((1.0 - k) * last_dir + k * down, down)
            synth = np.array([p[0] + direction[0] * length, p[1] + direction[1] * length, p[2]])
            mix(child, synth)

        # Face and hands: follow their anchor with the last known offset.
        anchors = [(j, (out[L_SHOULDER] + out[R_SHOULDER]) / 2.0) for j in FACE]
        for anchor_joint, joints in APPENDAGES:
            anchors += [(j, out[anchor_joint]) for j in joints]
        for j, anchor in anchors:
            offset = st.offsets.get(j)
            if offset is not None:
                synth = anchor + offset
            else:
                synth = prev[j] if prev is not None else meas[j]
            mix(j, synth)
            if w[j] >= 0.9:
                st.offsets[j] = out[j] - anchor

        # Learn bone lengths from well-measured limbs only.
        if w[L_SHOULDER] >= 0.9 and w[R_SHOULDER] >= 0.9:
            width = float(np.hypot(*(out[L_SHOULDER, :2] - out[R_SHOULDER, :2])))
            if width > 1e-3:
                st.shoulder_ref = _slow_max(st.shoulder_ref, width)
        for child, parent, key in CHAIN:
            if w[child] >= 0.9 and w[parent] >= 0.9:
                length = float(np.hypot(*(out[child, :2] - out[parent, :2])))
                if length > 1e-3:
                    st.lengths[key] = _slow_max(st.lengths.get(key), length)

        return out
