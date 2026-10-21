"""
Keeping each player's pose their own when two players overlap (e.g. one reaches over the other).

When a player's arm reaches across the other player, their detector box (and so their pose crop) also
covers the other player's body. Three safeguards, all applied to raw MediaPipe poses before smoothing:

1. select_pose: the pose model returns several candidate poses per crop; pick the one whose torso
   continues this player's torso from the previous frame, and reject one that belongs to the other
   player or jumps implausibly far (the smoother then briefly holds the last pose).
2. enforce_arm_reach: a wrist further from its shoulder than an arm can reach was taken from someone
   else; mark it unseen.
3. resolve_shared_wrists: when two players' wrists land on the same spot, only one of them can own
   that hand; the one whose arm doesn't plausibly reach it marks its wrist unseen.

"Unseen" means visibility 0 on the wrist and its hand points: the smoother then infers the hand from
the arm instead of following it, and Unity holds the chef's hand still.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.pose_data import PoseLandmark, PoseTrackingData, POSE_LANDMARK_COUNT

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
# (shoulder, elbow, wrist, hand points)
ARMS = ((11, 13, 15, (17, 19, 21)), (12, 14, 16, (18, 20, 22)))

# Upper arm + forearm is about 1.45 shoulder widths (see pose_smoother.LENGTH_PRIOR); allow slack for noise
MAX_ARM_REACH_SHOULDERS = 2.0
# Forearms (elbow to wrist) longer than this many shoulder widths don't belong to that elbow
MAX_FOREARM_SHOULDERS = 1.0
# Wrists of two players closer than this (in shoulder widths) are treated as the same hand
SHARED_WRIST_SHOULDERS = 0.35

Torso = Tuple[np.ndarray, float]  # (shoulder midpoint in frame pixels, shoulder width in pixels)


def frame_points(pose: PoseTrackingData, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop-normalised landmarks -> (N, 2) frame pixel coordinates."""
    bx1, by1, bx2, by2 = box
    cw, ch = max(bx2 - bx1, 1), max(by2 - by1, 1)
    return np.array([[bx1 + lm.x * cw, by1 + lm.y * ch] for lm in pose.landmarks[:POSE_LANDMARK_COUNT]])


def torso_of(points: np.ndarray) -> Torso:
    """Shoulder midpoint and shoulder width (pixels) of a pose."""
    ls, rs = points[L_SHOULDER], points[R_SHOULDER]
    return (ls + rs) / 2.0, max(float(np.linalg.norm(ls - rs)), 1.0)


def _visibility(lm) -> float:
    value = getattr(lm, "visibility", None)
    return 1.0 if value is None else float(value)


def editable(pose: PoseTrackingData) -> PoseTrackingData:
    """Copy of a pose with plain PoseLandmark objects, so joint visibility can be changed."""
    landmarks = [
        PoseLandmark(float(lm.x), float(lm.y), float(lm.z), _visibility(lm),
                     float(getattr(lm, "presence", 1.0) or 1.0))
        for lm in pose.landmarks
    ]
    return PoseTrackingData(landmarks=landmarks, tracking_active=pose.tracking_active)


def _hide_hand(pose: PoseTrackingData, arm) -> None:
    _, _, wrist, hand = arm
    for i in (wrist, *hand):
        pose.landmarks[i].visibility = 0.0
        pose.landmarks[i].presence = 0.0


def select_pose(candidates: Sequence[PoseTrackingData], box: Tuple[int, int, int, int],
                own_torso: Optional[Torso], other_torsos: Sequence[Torso],
                max_torso_jump: float) -> Optional[PoseTrackingData]:
    """
    Pick this player's pose from the candidates found in their crop.

    own_torso      this player's torso last frame (None if unknown)
    other_torsos   the other players' torsos last frame
    max_torso_jump largest believable torso move between frames, in shoulder widths
    Returns None if no candidate is believably this player.
    """
    candidates = [c for c in candidates if c is not None and c.is_tracking and len(c.landmarks) >= POSE_LANDMARK_COUNT]
    if not candidates:
        return None

    best, best_cost = None, float("inf")
    for cand in candidates:
        center, width = torso_of(frame_points(cand, box))

        if own_torso is not None:
            own_center, own_width = own_torso
            cost = np.linalg.norm(center - own_center) / own_width
            if cost > max_torso_jump:
                continue
        else:
            # No history yet: prefer the biggest (closest) body nearest the middle of the crop
            bx1, by1, bx2, by2 = box
            crop_center = np.array([(bx1 + bx2) / 2.0, by1 + (by2 - by1) * 0.35])
            cost = np.linalg.norm(center - crop_center) / max(bx2 - bx1, 1) - width / max(bx2 - bx1, 1)

        # A torso sitting where another player was is that player, not us
        if any(np.linalg.norm(center - oc) / ow < 0.5 for oc, ow in other_torsos):
            if own_torso is None or np.linalg.norm(center - own_torso[0]) / own_torso[1] > 0.25:
                continue

        if cost < best_cost:
            best, best_cost = cand, cost
    return best


def enforce_arm_reach(pose: PoseTrackingData, box: Tuple[int, int, int, int],
                      reference_width: Optional[float] = None) -> None:
    """
    Mark wrists that no arm of this body could reach as unseen (they belong to someone else).
    reference_width: the person's usual shoulder width in pixels. The current one shrinks when they turn
    sideways, which would wrongly flag a normal outstretched arm.
    """
    points = frame_points(pose, box)
    _, width = torso_of(points)
    if reference_width is not None:
        width = max(width, reference_width)
    for arm in ARMS:
        shoulder, elbow, wrist, _ = arm
        too_far = np.linalg.norm(points[wrist] - points[shoulder]) > MAX_ARM_REACH_SHOULDERS * width
        forearm_too_long = np.linalg.norm(points[wrist] - points[elbow]) > MAX_FOREARM_SHOULDERS * width
        if too_far or forearm_too_long:
            _hide_hand(pose, arm)


def resolve_shared_wrists(entries: List[Tuple[int, PoseTrackingData, Tuple[int, int, int, int]]]) -> None:
    """
    entries: (player id, editable raw pose, crop box) for every player this frame.
    When wrists of two players coincide, keep the hand only for the player whose own elbow plausibly
    reaches it (shorter forearm relative to their shoulder width; ties go to the more visible wrist).
    """
    measured = []
    for pid, pose, box in entries:
        points = frame_points(pose, box)
        measured.append((pid, pose, points, torso_of(points)[1]))

    for i in range(len(measured)):
        for j in range(i + 1, len(measured)):
            _, pose_a, pts_a, width_a = measured[i]
            _, pose_b, pts_b, width_b = measured[j]
            for arm_a in ARMS:
                for arm_b in ARMS:
                    wa, wb = arm_a[2], arm_b[2]
                    if pose_a.landmarks[wa].visibility <= 0.0 or pose_b.landmarks[wb].visibility <= 0.0:
                        continue
                    if np.linalg.norm(pts_a[wa] - pts_b[wb]) > SHARED_WRIST_SHOULDERS * min(width_a, width_b):
                        continue
                    forearm_a = np.linalg.norm(pts_a[wa] - pts_a[arm_a[1]]) / width_a
                    forearm_b = np.linalg.norm(pts_b[wb] - pts_b[arm_b[1]]) / width_b
                    if abs(forearm_a - forearm_b) > 0.1:
                        a_loses = forearm_a > forearm_b
                    else:
                        a_loses = pose_a.landmarks[wa].visibility < pose_b.landmarks[wb].visibility
                    _hide_hand(pose_a if a_loses else pose_b, arm_a if a_loses else arm_b)
