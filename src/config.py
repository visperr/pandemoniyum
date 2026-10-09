# Network Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Display & Debug
SHOW_PREVIEW = True
DEBUG_MODE = True
SAVE_PREVIEW = False
SAVE_PATH = "../output2.mp4"

# Video source selection
USE_CAMERA = True
CAMERA_INDEX = 0
VIDEO_PATH = "../WIN_20260923_15_21_55_Pro.mp4"

# Asset Paths
MODEL_DIR = "../models_data"

# Tracker Parameters

# FACE
FACE_NUM_FACES = 1
FACE_MIN_CONFIDENCE = 0.5
FACE_BLENDSHAPES = ['jawOpen']

# Person Trackerq
PERSON_NUM_PERSONS = 2
PERSON_CONFIDENCE_THRESHOLD = 0.40
PERSON_MAX_AGE_FRAMES = 90
PERSON_MAX_COSINE_DIST = 0.20
PERSON_REID_SIMILARITY_THRESHOLD = 0.65
PERSON_DETECTOR_IMAGE_SIZE = 480
PERSON_DETECTION_FRAME_SKIP = 2

# Pose Tracker
POSE_MODEL_FILE = "pose_landmarker_lite.task"  # "pose_landmarker_full.task" is more accurate/stable, but slower
POSE_NUM_POSES = 1
POSE_MIN_CONFIDENCE = 0.5
LANDMARK_FRAME_SKIP = 2

# Pose smoothing & off-screen handling (see src/core/pose_smoother.py)
POSE_SMOOTHING_ENABLED = True        # False = raw MediaPipe output, for A/B comparison
POSE_MIN_CUTOFF = 0.6                # Hz. Lower = smoother when still (less jitter), more lag at motion start
POSE_BETA = 15.0                     # Higher = less lag during fast motion, but lets more jitter through
POSE_SPEED_FLOOR = 0.2               # Joint speeds below this (person heights/s) are treated as noise
POSE_VISIBILITY_LOW = 0.35           # Joint confidence below this: position is inferred, not measured
POSE_VISIBILITY_HIGH = 0.65          # Joint confidence above this: position is fully measured
POSE_REST_BLEND_SECONDS = 1.5        # Unseen limbs relax from their last direction to hanging down over this time
POSE_DROPOUT_HOLD_SECONDS = 0.4      # Hold the last pose this long if the detector finds nothing
POSE_RESET_SECONDS = 1.0             # Start the filters fresh after this long without a pose
