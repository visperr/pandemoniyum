# PANdeminiYUM Motion Tracking Engine

This is the standalone Python tracking backend for PANdeminiYUM[cite: 3]. It utilizes MediaPipe Tasks API to perform multi-modal tracking (Face, Pose, Hand/Gesture) for up to 2 simultaneous users and broadcasts the cleaned data via UDP to Unity.

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