"""
Open/closed hand detection for grabbing, plus hand signs for emotes.

For each tracked wrist, a square crop around the hand (located by the pose landmarks) is run through
MediaPipe's gesture recognizer, which also returns 21 3D hand landmarks. Closure is measured from how
far the fingers are curled (fingertip vs knuckle distance to the wrist, in 3D so hand rotation doesn't
matter), blended with the classifier's Closed_Fist / Open_Palm verdicts. The result is smoothed with
hysteresis so a single misread frame does not drop or pick up an item.

The classifier's other verdicts (Thumb_Up, Thumb_Down, Victory, ILoveYou, Pointing_Up) are reported as the
hand's "gesture" once the same one has been read several times in a row, so a passing misread never shows.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Pose landmarks per hand: (wrist, pinky, index)
HAND_LANDMARKS = {15: (15, 17, 19), 16: (16, 18, 20)}

# MediaPipe hand landmarks: wrist, and (knuckle, tip) per finger (index, middle, ring, pinky; thumb ignored)
HAND_WRIST = 0
FINGERS = ((5, 8), (9, 12), (13, 16), (17, 20))

# Fingertip-to-wrist / knuckle-to-wrist ratio: about 2 for a flat open hand, about 1 for a fist
CURL_CLOSED_RATIO = 1.2
CURL_OPEN_RATIO = 1.6

# The hand model works best when the hand isn't tiny in its input
MIN_CROP_PIXELS = 256

# Skip wrists the pose model thinks are off-screen/hidden (saves the hand model's time)
MIN_WRIST_VISIBILITY = 0.3

# Hand signs reported to Unity (the grab gestures Closed_Fist / Open_Palm are covered by "closed")
SIGN_GESTURES = ("Thumb_Up", "Thumb_Down", "Victory", "ILoveYou", "Pointing_Up")
NO_GESTURE = "None"


@dataclass
class HandState:
    closed_prob: float = 0.0
    closed: bool = False
    seen: bool = False
    # Debounced hand sign (one of SIGN_GESTURES or NO_GESTURE) and the classifier's confidence in it
    gesture: str = NO_GESTURE
    gesture_score: float = 0.0
    _candidate: str = NO_GESTURE
    _candidate_count: int = 0


def curl_closed_score(world_landmarks) -> float:
    """0 = fingers extended, 1 = fist. Uses 3D world landmarks (metres), so it works at any hand rotation."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
    wrist = pts[HAND_WRIST]
    ratios = []
    for knuckle, tip in FINGERS:
        knuckle_dist = np.linalg.norm(pts[knuckle] - wrist)
        if knuckle_dist > 1e-6:
            ratios.append(np.linalg.norm(pts[tip] - wrist) / knuckle_dist)
    if not ratios:
        return 0.5
    ratio = float(np.mean(ratios))
    return float(np.clip((CURL_OPEN_RATIO - ratio) / (CURL_OPEN_RATIO - CURL_CLOSED_RATIO), 0.0, 1.0))


class HandGestureTracker:
    def __init__(self, model_dir: str, min_score: float = 0.5, crop_scale: float = 1.3,
                 smoothing: float = 0.6, close_threshold: float = 0.6, open_threshold: float = 0.35,
                 sign_min_score: float = 0.6, sign_confirm_readings: int = 3):
        """
        min_score        Minimum hand detection confidence to count a reading.
        crop_scale       Crop side length as a multiple of shoulder width.
        smoothing        Weight of the newest reading in the closed-probability filter (0..1).
        close/open_threshold  Hysteresis: the hand closes above close_threshold and opens below open_threshold.
        sign_min_score   Minimum classifier confidence for a hand sign reading to count.
        sign_confirm_readings  Same sign this many readings in a row before it is reported (and none to clear it).
        """
        self.min_score = min_score
        self.crop_scale = crop_scale
        self.smoothing = smoothing
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.sign_min_score = sign_min_score
        self.sign_confirm_readings = max(1, sign_confirm_readings)
        self.states: Dict[Tuple[int, int], HandState] = {}
        # Each update checks one hand per person, alternating, to halve the cost (~6 ms per hand)
        self._turn: Dict[int, int] = {}

        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=os.path.join(model_dir, "gesture_recognizer.task")),
            # Two, so that when both hands are close together we can pick the one at the expected wrist
            num_hands=2,
            min_hand_detection_confidence=min_score,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.model = vision.GestureRecognizer.create_from_options(options)

    def update(self, pid: int, pose, box: Tuple[int, int, int, int], frame: np.ndarray) -> None:
        """Run the recognizer on one hand of one person (alternating per call). `pose` is crop-normalised to `box`."""
        if pose is None or not pose.tracking_active or len(pose.landmarks) <= 20:
            for wrist in HAND_LANDMARKS:
                self._state(pid, wrist).seen = False
            return

        bx1, by1, bx2, by2 = box
        cw, ch = max(bx2 - bx1, 1), max(by2 - by1, 1)

        def to_px(i):
            lm = pose.landmarks[i]
            return np.array([bx1 + lm.x * cw, by1 + lm.y * ch])

        shoulder_width = np.linalg.norm(to_px(11) - to_px(12))
        side = int(max(64.0, shoulder_width * self.crop_scale))

        # Alternate hands; if the scheduled hand is off-screen, check the other one instead
        order = list(HAND_LANDMARKS.items())
        turn = self._turn.get(pid, 0)
        self._turn[pid] = 1 - turn
        order = order[turn:] + order[:turn]

        for wrist, ids in order:
            if getattr(pose.landmarks[ids[0]], "visibility", 1.0) < MIN_WRIST_VISIBILITY:
                # Off-screen or hidden: nothing to recognise, keep the last state
                self._state(pid, wrist).seen = False
                continue
            wrist_px = to_px(ids[0])
            # The hand extends past the pose's wrist towards its pinky/index points: centre the crop on it
            knuckles = (to_px(ids[1]) + to_px(ids[2])) / 2.0
            center = wrist_px + (knuckles - wrist_px) * 1.2
            reading, sign, sign_score = self._recognize(frame, center, side, wrist_px)
            st = self._state(pid, wrist)
            self._apply(st, reading)
            self._apply_sign(st, sign if reading is not None else NO_GESTURE, sign_score)
            break

    def hands_dict(self, pid: int) -> dict:
        """Per-wrist result for the UDP packet, keyed by pose landmark index (15/16)."""
        out = {}
        for wrist in HAND_LANDMARKS:
            st = self._state(pid, wrist)
            out[f"hand{wrist}"] = {
                "closed": st.closed,
                "closed_prob": round(st.closed_prob, 3),
                "seen": st.seen,
                "gesture": st.gesture,
                "gesture_score": round(st.gesture_score, 3),
            }
        return out

    def _state(self, pid: int, wrist: int) -> HandState:
        return self.states.setdefault((pid, wrist), HandState())

    def _recognize(self, frame: np.ndarray, center: np.ndarray, side: int,
                   expected_wrist: np.ndarray) -> Tuple[Optional[float], str, float]:
        """
        For the hand nearest the expected wrist: closed score 0..1 (None if no hand is found), and its hand sign
        (one of SIGN_GESTURES, or NO_GESTURE) with the classifier's confidence.
        """
        h, w = frame.shape[:2]
        x1 = int(np.clip(center[0] - side / 2, 0, w))
        y1 = int(np.clip(center[1] - side / 2, 0, h))
        x2 = int(np.clip(center[0] + side / 2, 0, w))
        y2 = int(np.clip(center[1] + side / 2, 0, h))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None, NO_GESTURE, 0.0

        crop = frame[y1:y2, x1:x2]
        scale = 1.0
        if max(crop.shape[:2]) < MIN_CROP_PIXELS:
            scale = MIN_CROP_PIXELS / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        result = self.model.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
        if not result.hand_landmarks:
            return None, NO_GESTURE, 0.0

        # Pick the detected hand whose wrist is closest to where the pose model puts this wrist
        ch, cw = crop.shape[:2]
        target = np.array([(expected_wrist[0] - x1) * scale / cw, (expected_wrist[1] - y1) * scale / ch])
        best = min(range(len(result.hand_landmarks)),
                   key=lambda i: np.linalg.norm(np.array([result.hand_landmarks[i][HAND_WRIST].x,
                                                          result.hand_landmarks[i][HAND_WRIST].y]) - target))

        score = curl_closed_score(result.hand_world_landmarks[best])

        # The classifier is reliable for clear fists / flat palms; "None" and other gestures leave the curl score alone
        sign, sign_score = NO_GESTURE, 0.0
        if result.gestures and best < len(result.gestures) and result.gestures[best]:
            top = result.gestures[best][0]
            if top.category_name == "Closed_Fist":
                score = max(score, top.score)
            elif top.category_name == "Open_Palm":
                score = min(score, 1.0 - top.score)
            elif top.category_name in SIGN_GESTURES and top.score >= self.sign_min_score:
                sign, sign_score = top.category_name, float(top.score)
        return score, sign, sign_score

    def _apply(self, st: HandState, reading: Optional[float]) -> None:
        st.seen = reading is not None
        if reading is None:
            return  # Keep the last state while the hand isn't found
        st.closed_prob += self.smoothing * (reading - st.closed_prob)
        if st.closed and st.closed_prob < self.open_threshold:
            st.closed = False
        elif not st.closed and st.closed_prob > self.close_threshold:
            st.closed = True

    def _apply_sign(self, st: HandState, sign: str, score: float) -> None:
        """Report a sign (or its end) only after sign_confirm_readings identical readings in a row."""
        if sign == st._candidate:
            st._candidate_count += 1
        else:
            st._candidate = sign
            st._candidate_count = 1
        if st._candidate_count >= self.sign_confirm_readings:
            st.gesture = st._candidate
        st.gesture_score = score if st.gesture == sign else st.gesture_score
        if st.gesture == NO_GESTURE:
            st.gesture_score = 0.0

    def forget(self, active_ids) -> None:
        """Drop state of persons that are no longer tracked."""
        self.states = {k: v for k, v in self.states.items() if k[0] in active_ids}
        self._turn = {k: v for k, v in self._turn.items() if k in active_ids}

    def close(self) -> None:
        if self.model:
            self.model.close()
