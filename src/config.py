# Network Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Display & Debug
SHOW_PREVIEW = True
DEBUG_MODE = True
SAVE_PREVIEW = False
SAVE_PATH = "../output.mp4"

# Video source selection
USE_CAMERA = True
CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
VIDEO_PATH = "../WIN_20260923_15_21_55_Pro.mp4"

# Asset Paths
MODEL_DIR = "../models_data"

# Tracker Parameters

# FACE
FACE_NUM_FACES = 1
FACE_MIN_CONFIDENCE = 0.5
FACE_IMAGE_ENABLED = True            # Send each player's face picture so Unity can put it on the chef's head
FACE_IMAGE_SIZE = 128
FACE_IMAGE_MAX_BYTES = 5000          # JPEG size cap per face
FACE_BLENDSHAPES = ['jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'eyeBlinkLeft', 'eyeBlinkRight']

# Person Tracker
PERSON_NUM_PERSONS = 2
PERSON_CONFIDENCE_THRESHOLD = 0.40
PERSON_MAX_AGE_FRAMES = 30
PERSON_FREEZE_FRAMES = 30 # After detection is lost, keep the last box (and keep running pose/face) this many frames before releasing the lock. Must be < PERSON_MAX_AGE_FRAMES
PERSON_MAX_COSINE_DIST = 0.20
PERSON_REID_SIMILARITY_THRESHOLD = 0.65
PERSON_DETECTOR_IMAGE_SIZE = 480
PERSON_DETECTION_FRAME_SKIP = 2
PERSON_DETECTOR_MODEL = "yolov8n.pt"           # Small and fast; used if present in MODEL_DIR
PERSON_DETECTOR_FALLBACK_MODEL = "yolov8m.pt"  # Used when PERSON_DETECTOR_MODEL is missing

# Pose Tracker
POSE_MODEL_FILE = "pose_landmarker_lite.task"  # "pose_landmarker_full.task" is more accurate/stable, but slower
POSE_NUM_POSES = 2                   # Poses found per crop; 2 lets us pick the right one when players overlap
POSE_MAX_TORSO_JUMP = 1.0            # Largest believable torso move between frames, in shoulder widths
POSE_MIN_CONFIDENCE = 0.5
LANDMARK_FRAME_SKIP = 1              # Run pose/face every N frames (1 = every frame)
POSE_CROP_PADDING_X = 0.25           # Widen the person box by this fraction per side so reaching hands stay in the crop
POSE_CROP_PADDING_TOP = 0.10         # Extend the person box upwards by this fraction for raised hands

# Hand gestures (grabbing)
GESTURE_ENABLED = True
GESTURE_FRAME_SKIP = 1               # Run the hand recognizer every N frames (1 = every frame: fastest, most reliable grabs)
GESTURE_MIN_SCORE = 0.5              # Minimum hand detection confidence
GESTURE_CROP_SCALE = 1.3             # Palm crop side length as a multiple of shoulder width
GESTURE_SIGN_MIN_SCORE = 0.6         # Hand signs (thumbs up/down, victory, ...) need this classifier confidence
GESTURE_SIGN_CONFIRM_READINGS = 3    # ...and this many identical readings in a row (each hand is read every other frame)

# Camera image in the UDP packets. Unity asks for it only while the calibration screen shows it
VIDEO_DEFAULT_ON = True              # Until Unity says otherwise

# Pose smoothing & off-screen handling (see src/core/pose_smoother.py)
POSE_SMOOTHING_ENABLED = True        # False = raw MediaPipe output, for A/B comparison
POSE_MIN_CUTOFF = 0.6                # Hz. Lower = smoother when still (less jitter), more lag at motion start
POSE_BETA = 15.0                     # Higher = less lag during fast motion, but lets more jitter through
POSE_SPEED_FLOOR = 0.2               # Joint speeds below this (person heights/s) are treated as noise
POSE_VISIBILITY_LOW = 0.35           # Joint confidence below this: position is inferred, not measured
POSE_VISIBILITY_HIGH = 0.65          # Joint confidence above this: position is fully measured
POSE_REST_BLEND_SECONDS = 1.5        # Unseen limbs relax from their last direction to hanging down over this time
POSE_DROPOUT_HOLD_SECONDS = 0.4      # Hold the last pose this long if the detector finds nothing
POSE_HOLD_UNSEEN_HANDS = True        # Uncertain wrists hold their last position instead of drifting towards hanging down
POSE_MAX_FOREARM_STRETCH = 1.35      # Forearm this many times its learned length = misplaced wrist (held, not followed)
POSE_MAX_HAND_SPEED = 6.0            # Wrist faster than this (person heights/s) = glitch, unless it persists
POSE_RESET_SECONDS = 1.0             # Start the filters fresh after this long without a pose

# Evaluation (see EVALUATION.md): `python main.py --record` stores sessions here
RECORDINGS_DIR = "../recordings"
