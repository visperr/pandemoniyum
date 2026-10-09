# PANdemoniYUM Motion Tracking Engine

This is the standalone Python tracking backend for PANdemoniYUM[cite: 3]. It utilizes MediaPipe Tasks API to perform multi-modal tracking (Face, Pose, Hand/Gesture) for up to 2 simultaneous users and broadcasts the cleaned data via UDP to Unity.

## Setup Instructions (Development)
1. Ensure Python 3.10+ is installed.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the orchestrator: `python src/main.py`

## Build Instructions (Production)
To build the headless `.exe` for Unity integration:
1. Run `pyinstaller --noconsole --onefile src/main.py`
2. Move the generated `dist/main.exe` into your Unity project's `StreamingAssets/Tracker/` folder.
3. Copy the `models_data/` folder into the same `Tracker/` directory so the executable can locate the weights.

## Performance
The person detector and re-identification model use CUDA when available, otherwise Apple's
Metal Performance Shaders (MPS) on supported Macs, and CPU as a fallback. Person detection
defaults to a 480-pixel inference size; increase `PERSON_DETECTOR_IMAGE_SIZE` in `src/config.py`
if you need more detection detail.

Preview, per-frame debug output, and preview-video encoding are disabled by default to avoid
adding UI, terminal, and encoder overhead to the tracking loop. Enable `SHOW_PREVIEW`,
`DEBUG_MODE`, or `SAVE_PREVIEW` in `src/config.py` individually when needed.

Person detection/re-identification and face/pose inference run every second frame by default
(`PERSON_DETECTION_FRAME_SKIP` and `LANDMARK_FRAME_SKIP`); person positions are predicted and
the most recent face/pose results are reused between updates. Lower either skip value to `1`
for per-frame inference at the cost of throughput, or increase it if you need more headroom.
## Pose smoothing and off-screen limbs
Pose landmarks pass through `src/core/pose_smoother.py` before they are streamed:
- Each joint is smoothed with a One Euro filter (strong smoothing when still, little lag when moving), in frame
  space so that the jittering person box does not leak into the landmarks.
- Joints that MediaPipe reports with low visibility or places outside the frame (e.g. hands below the webcam
  image) are not trusted. They are placed at the parent joint plus the learned bone length, holding their last
  direction and slowly relaxing to "hanging down", and cross-fade back when they re-enter the frame.
- Only landmarks 0-24 (face, arms, torso, hips) are kept: everything below the hips (25-32) is dropped right
  after detection, so it is never processed, drawn or sent. The `pose.landmarks` array therefore has 25 entries.
- Every landmark in the UDP payload now has an extra `"visibility"` field (0 = inferred, 1 = measured).
- Tune with the `POSE_*` values in `src/config.py`; set `POSE_SMOOTHING_ENABLED = False` for raw output.
