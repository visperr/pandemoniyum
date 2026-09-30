class DataPacker:
    def __init__(self):
        # The clean schema Unity expects
        self.template = {
            "players": [
                {"id": 1, "blendshapes": {}, "head_matrix": [], "pose": [], "hands": {"Left": None, "Right": None}},
                {"id": 2, "blendshapes": {}, "head_matrix": [], "pose": [], "hands": {"Left": None, "Right": None}}
            ]
        }

    def package_frame(self, face_res, pose_res, gesture_res):
        payload = self.template.copy()

        # --- Clean and Map Face Data ---
        if face_res and face_res.face_landmarks:
            for i in range(min(len(face_res.face_landmarks), 2)):
                payload["players"][i]["blendshapes"] = {
                    c.category_name: round(c.score, 3) for c in face_res.face_blendshapes[i]
                }
                payload["players"][i]["head_matrix"] = face_res.facial_transformation_matrixes[i].flatten().tolist()

        # --- Clean and Map Pose Data ---
        if pose_res and pose_res.pose_landmarks:
            for i in range(min(len(pose_res.pose_landmarks), 2)):
                payload["players"][i]["pose"] = [
                    {"x": round(lm.x, 3), "y": round(lm.y, 3), "z": round(lm.z, 3)}
                    for lm in pose_res.pose_landmarks[i]
                ]

        # --- Clean and Map Gesture/Hand Data ---
        if gesture_res and gesture_res.hand_landmarks:
            for i, hand_landmarks in enumerate(gesture_res.hand_landmarks):
                # Determine which player this hand belongs to based on X-coordinate sorting logic (to be implemented)
                player_idx = 0  # Placeholder for spatial sorting logic

                handedness = gesture_res.handedness[i][0].category_name
                gesture = gesture_res.gestures[i][0].category_name if gesture_res.gestures else "None"

                payload["players"][player_idx]["hands"][handedness] = {
                    "gesture": gesture,
                    "joints": [{"x": round(lm.x, 3), "y": round(lm.y, 3)} for lm in hand_landmarks]
                }

        return payload