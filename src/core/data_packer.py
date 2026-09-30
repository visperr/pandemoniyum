import math
from typing import List, Optional
from models.tracking_data import (
    TrackingFrame, PlayerData, FaceData, PoseData, HandData, Landmark
)

class DataPacker:
    def __init__(self):
        self._frame_count: int = 0

    def package_frame(self, face_res, pose_res, gesture_res) -> TrackingFrame:
        self._frame_count += 1
        frame = TrackingFrame(frame_index=self._frame_count)

        # Initialise two fixed player slots
        p1 = PlayerData(player_id=1)
        p2 = PlayerData(player_id=2)

        # 1. Parse and sort Body Poses (determines screen left vs screen right)
        poses = self._extract_poses(pose_res)
        if len(poses) == 1:
            if poses[0].centre_x < 0.5:
                p1.pose = poses[0]
                p1.is_tracked = True
            else:
                p2.pose = poses[0]
                p2.is_tracked = True
        elif len(poses) >= 2:
            # Sort ascending by X (lowest X is on the left)
            poses.sort(key=lambda p: p.centre_x)
            p1.pose = poses[0]
            p1.is_tracked = True
            p2.pose = poses[1]
            p2.is_tracked = True

        # 2. Parse and assign Faces
        faces = self._extract_faces(face_res)
        self._assign_faces_to_players(faces, p1, p2)

        # 3. Parse and assign Hands/Gestures
        hands = self._extract_hands(gesture_res)
        self._assign_hands_to_players(hands, p1, p2)

        frame.players = [p1, p2]
        return frame

    def _extract_poses(self, pose_res) -> List[PoseData]:
        poses: List[PoseData] = []
        if not pose_res or not pose_res.pose_landmarks:
            return poses

        for raw_landmarks in pose_res.pose_landmarks:
            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=getattr(lm, 'visibility', 1.0))
                for lm in raw_landmarks
            ]
            poses.append(PoseData(landmarks=landmarks))
        return poses

    def _extract_faces(self, face_res) -> List[FaceData]:
        faces: List[FaceData] = []
        if not face_res or not face_res.face_landmarks:
            return faces

        for i in range(len(face_res.face_landmarks)):
            blendshapes = {
                category.category_name: category.score
                for category in face_res.face_blendshapes[i]
            }
            matrix = []
            if face_res.facial_transformation_matrixes and len(face_res.facial_transformation_matrixes) > i:
                matrix = face_res.facial_transformation_matrixes[i].flatten().tolist()

            faces.append(FaceData(blendshapes=blendshapes, head_matrix=matrix))
        return faces

    def _assign_faces_to_players(self, faces: List[FaceData], p1: PlayerData, p2: PlayerData):
        if not faces:
            return
        if len(faces) == 1:
            # Assign to whichever player is already tracked, or fallback to p1
            if p2.is_tracked and not p1.is_tracked:
                p2.face = faces[0]
            else:
                p1.face = faces[0]
        else:
            p1.face = faces[0]
            p2.face = faces[1]

    def _extract_hands(self, gesture_res) -> List[HandData]:
        hands: List[HandData] = []
        if not gesture_res or not gesture_res.hand_landmarks:
            return hands

        for idx, raw_landmarks in enumerate(gesture_res.hand_landmarks):
            # Handedness mirroring check
            raw_handedness = gesture_res.handedness[idx][0].category_name
            true_handedness = "Right" if raw_handedness == "Left" else "Left"

            # Gesture identification
            gesture_name = "None"
            gesture_score = 0.0
            if gesture_res.gestures and len(gesture_res.gestures) > idx and gesture_res.gestures[idx]:
                gesture_name = gesture_res.gestures[idx][0].category_name
                gesture_score = gesture_res.gestures[idx][0].score

            # Convert landmarks
            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in raw_landmarks
            ]

            # Calculate grab strength (distance: wrist to middle fingertip vs palm size)
            palm_size = math.hypot(landmarks[0].x - landmarks[9].x, landmarks[0].y - landmarks[9].y)
            finger_dist = math.hypot(landmarks[0].x - landmarks[12].x, landmarks[0].y - landmarks[12].y)
            ratio = finger_dist / (palm_size + 1e-4)
            grab = max(0.0, min(1.0, (2.0 - ratio) / 1.2))

            hands.append(HandData(
                handedness=true_handedness,
                gesture=gesture_name,
                gesture_score=gesture_score,
                grab_score=grab,
                landmarks=landmarks
            ))
        return hands

    def _assign_hands_to_players(self, hands: List[HandData], p1: PlayerData, p2: PlayerData):
        for hand in hands:
            wrist = hand.wrist
            if not wrist:
                continue

            # Compute distance to each player's respective wrist if pose is tracked
            d1 = self._wrist_distance(wrist, p1, hand.handedness)
            d2 = self._wrist_distance(wrist, p2, hand.handedness)

            if d1 is not None and d2 is not None:
                if d1 < d2:
                    p1.hands[hand.handedness] = hand
                else:
                    p2.hands[hand.handedness] = hand
            elif d1 is not None:
                p1.hands[hand.handedness] = hand
            elif d2 is not None:
                p2.hands[hand.handedness] = hand
            else:
                # Fallback: assign purely based on screen half
                if wrist.x < 0.5:
                    p1.hands[hand.handedness] = hand
                else:
                    p2.hands[hand.handedness] = hand

    @staticmethod
    def _wrist_distance(hand_wrist: Landmark, player: PlayerData, handedness: str) -> Optional[float]:
        if not player.pose.is_detected:
            return None
        target_wrist = player.pose.left_wrist if handedness == "Left" else player.pose.right_wrist
        if not target_wrist:
            return None
        return math.hypot(hand_wrist.x - target_wrist.x, hand_wrist.y - target_wrist.y)