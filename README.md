# PANdemoniYUM Motion Tracking Engine

This is the standalone Python tracking backend for PANdemoniYUM[cite: 3]. It utilizes MediaPipe Tasks API to perform multi-modal tracking (Face, Pose, Hand/Gesture) for up to 2 simultaneous users and broadcasts the cleaned data via UDP to Unity.

## Dependencies

All dependencies are listed in `requirements.txt` and are fetched using the setup instructions below:

## Setup and Execution Instructions (Development)
1. Ensure Python 3.10+ is installed.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the program: `cd src && python main.py`

## Build Instructions (Production)
To build the headless `.exe` for Unity integration:
1. Run `pyinstaller --noconsole --onefile src/main.py`
2. Move the generated `dist/main.exe` into your Unity project's `StreamingAssets/Tracker/` folder.
3. Copy the `models_data/` folder into the same `Tracker/` directory so the executable can locate the weights.

## AI Disclosure

AI Disclosure: During the preparation of this work the author(s) used Google Gemini and Anthropic Claude in order to write code and accompanying documentation for the Python application and the Unity game. After using these tools/services, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the work.” 

## Task owners

- Face Tracking and Augmented Effects: s2389428
- Body Pose estimation and Motion Tracking: s3416178
- Multiplayer Detection and Identity Tracking: s3503925
- Player Interaction and Game Control: s2506890
- Scene Processing and Real-Time Integration: s2506890

## File structure

```
PANdemoniYUM/
├── .gitignore
├── README.md                  : this file
├── EVALUATION.md              : metric definitions, recording protocol
├── EVALUATION_TOOLKIT.md      : how to run the evaluation tools
├── requirements.txt           : list of project dependencies
├── build.py                   : compile the program using pyinstaller for unity integration
├── play_session.py            : for playing a recorded session
├── play_session.command
└── src/
    ├── __init__.py
    ├── main.py                : entry point: capture → track → send (+ record/replay options)
    ├── config.py              : all tunable settings
    ├── utils.py               : utility functions
    │
    ├── core/                  : shared machinery
    │   ├── __init__.py
    │   ├── base_tracker.py         : abstract tracker interface
    │   ├── data_packer.py          : builds the UDP payload
    │   ├── network_streamer.py     : UDP sender to Unity
    │   ├── latest_frame_capture.py : camera reader, skips stale frames
    │   ├── pose_smoother.py        : One Euro filter + missing-joint handling
    │   ├── pose_ownership.py       : keeps each player's pose their own
    │   ├── perf_monitor.py         : per-stage timings
    │   ├── session_recorder.py     : records / replays sessions, lighting simulation
    │   ├── timed_video_writer.py
    │   └── macos_priority.py
    │
    ├── trackers/              : the detection modules
    │   ├── __init__.py
    │   ├── person_tracker.py       : YOLO + DeepSort + re-ID, player slots, freeze
    │   ├── pose_tracker.py         : MediaPipe pose landmarker
    │   ├── face_tracker.py         : MediaPipe face landmarker
    │   └── hand_gesture_tracker.py : open/closed hand + hand signs
    │
    ├── models/                : data classes sent to Unity
    │   ├── __init__.py
    │   ├── base_data.py            : abstract base data class
    │   ├── scene_data.py           : top-level scene data
    │   ├── person_data.py          : data per person detected
    │   ├── pose_data.py            : pose data per person
    │   └── face_data.py            : face data per person
    │
    ├── networks/
    │   └── osnet.py               : re-identification embedder model
    │
    └── evaluation/           : offline metrics from recorded sessions
        ├── __init__.py
        ├── session.py            : loads a session into arrays
        ├── metrics.py            : all metric computations
        ├── report.py             : report.md + metrics.json per session
        ├── annotate.py           : click ground-truth keypoints / event onsets
        ├── missing_keypoints.py  : synthetic occlusion test
        ├── compare.py            : tables by lighting, distance, subject
        ├── camera_latency.py     : evaluate camera's own delay
        └── udp_monitor.py        : rate as received (stands in for Unity)
```
