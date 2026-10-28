# Evaluation toolkit: what was added and how to use it

This file has two parts: what was added to the tracker so it can be evaluated, and a step-by-step guide to
running the tools. For the meaning of each metric, the recording protocol and what to put in a report, see
[EVALUATION.md](EVALUATION.md).

---

## Part 1: What was added

### The idea

The tracker can now **record a session**: everything it saw and sent, frame by frame, plus the raw camera
video. All metrics are then computed **offline** from that recording, so:

- measuring has almost no effect on the tracker's speed (files are written on a background thread),
- the same recording can be analysed again, annotated, or **replayed** through the pipeline with different
  settings or simulated lighting,
- both the pose model's output **before** smoothing and the pose **sent** to Unity are stored, so the effect
  of the smoothing and of the missing-keypoint handling can be measured directly.

### The six metrics and where they are computed

| Metric | Tool | Needs annotation? |
|---|---|---|
| Keypoint detection rate | `evaluation.report` | No |
| Keypoint detection error (px, % shoulder width, mm, PCK, visibility precision/recall) | `evaluation.report` | Yes, `annotate keypoints` |
| Jitter, stationary and dynamic (raw vs. smoothed, plus the lag smoothing adds) | `evaluation.report` | No |
| Motion / gesture recognition delay (plus false grab triggers) | `evaluation.report`, `evaluation.camera_latency` | Yes, `annotate events` |
| Missing-keypoint handling, as it happened | `evaluation.report` | No |
| Missing-keypoint handling, synthetic occlusion test against known truth | `evaluation.missing_keypoints` | No |
| Module latency and update rate (incl. 5% / 1% low FPS) | `evaluation.report`, `evaluation.udp_monitor` | No |
| Influence of lighting (real or simulated), distance and subject, with the spread over repetitions | `evaluation.compare --by` | Optional |

### New files

| File | What it does |
|---|---|
| `src/core/session_recorder.py` | `SessionRecorder` writes a session folder; `SessionReplayCapture` plays one back with its original frame times; `simulate_lighting` darkens frames and adds noise |
| `src/evaluation/session.py` | Loads a session into arrays; defines the evaluated keypoints |
| `src/evaluation/metrics.py` | All metric computations |
| `src/evaluation/report.py` | Builds `report.md` + `metrics.json` (+ plots) for one session |
| `src/evaluation/annotate.py` | Window for clicking ground-truth keypoints and marking movement onsets |
| `src/evaluation/missing_keypoints.py` | Synthetic occlusion test of the missing-keypoint handling |
| `src/evaluation/compare.py` | Tables comparing sessions by lighting, distance, subject, ... (mean ± SD over repetitions) |
| `src/evaluation/camera_latency.py` | Measures the camera's own delay with a flashing screen |
| `src/evaluation/udp_monitor.py` | Listens in place of Unity and measures the received rate and latency |
| `EVALUATION.md` | Method: definitions, protocol, what to report |

### Changed files

| File | Change |
|---|---|
| `src/main.py` | Command-line options (`--record`, `--replay`, `--video`, `--brightness`, `--subject`, `--distance`, `--lux`, `--shoulder-width-mm`, ...); writes one record per frame while recording; fills `scene.capture_time` (epoch ms) and `scene.processing_time` (ms) in the UDP packet, which used to be always 0 |
| `src/core/latest_frame_capture.py` | Stamps each camera frame with its arrival time (`capture_time`) and counts the stale frames it skips (`dropped`) |
| `src/core/perf_monitor.py` | `snapshot()`: the per-stage timings of the current frame, for the log |
| `src/config.py` | `RECORDINGS_DIR = "../recordings"` |
| `README.md` | Short "Evaluation" section |

Without any options, `python main.py` behaves exactly as before. The `recordings/` folder is already in `.gitignore`.

### What was tested

Everything was run end to end on the sample `output.mp4`: recording, report, replay at normal and reduced
brightness, the synthetic occlusion test, the lighting comparison, and the ground-truth metrics with
generated annotations.

**Not tested yet:** a live camera session, the annotation windows (they need you clicking),
`camera_latency` and `udp_monitor`.

Results from that sample (it was not recorded for this purpose, so they only show the tools work):

- 29.4 Hz output, median 36 ms processing per frame (video file, so no camera timing)
- 30% brightness with sensor noise: the pose was found on 30% of frames instead of 93%
- a replay reproduces the recorded output to within 1.9 px (rounding in the log)
- **Finding:** when the pose model finds nothing, the last pose is held for 0.4 s, but it is still sent
  with "measured" visibility, so Unity cannot tell a held pose from a live one.

---

## Part 2: How to use it

### Before you start

Run everything **from the `src/` folder** with the project's virtual environment activated
(`source ../.venv/bin/activate` on macOS, `..\.venv\Scripts\activate` on Windows).
Every tool has `--help`.

### Step 1: Record a session

```bash
python main.py --record --condition stationary_office --subject P1 --distance 2.5 --lux 450 --shoulder-width-mm 380 --no-preview
```

- `--condition` is the label in the folder name and in reports. Use names like
  `stationary_office`, `dynamic_dim`, `gestures_backlit`.
- `--subject`, `--distance` (metres) and `--lux` are the impact factors. They are stored as fields, so
  `compare --by` can group sessions by them. Use anonymous codes like `P1` for subjects.
- `--shoulder-width-mm` is the player's shoulder width, measured with a tape from shoulder joint to shoulder
  joint. With it, errors and jitter are also reported in mm. You can also give it to the report later.
- `--notes` is free text stored with the session (anything else worth remembering).
- `--no-preview` turns off the preview window for clean latency numbers. Without it you can stop with `q`.
  With it, stop with `Ctrl+C` or use `--duration 60` to stop after 60 seconds.
- Recording to a different folder: `--recordings-dir PATH`. No video (smaller, but no annotation or replay):
  `--no-record-video`.

This creates `../recordings/<date>_<time>_<condition>/` containing:

| File | Content |
|---|---|
| `meta.json` | condition, impact factors, notes, source, simulated lighting, all config values |
| `frames.jsonl` | one line per frame: timestamps, stage timings, brightness, and per player the box, the pose before smoothing (`raw`), the sent pose (`pose`) and the hand states |
| `raw.mp4` | the camera video, unmirrored, exactly as the tracker received it |

Which sessions to record (per lighting condition) is in EVALUATION.md under "Recording protocol":
a stationary one, a dynamic one, and a gestures one. Record each one 3 times, and vary distance and
subject one at a time from a baseline (EVALUATION.md, "Impact factors").

### Step 2: Annotate ground truth (only for error and delay)

**Keypoints**, for the detection error:

```bash
python -m evaluation.annotate keypoints ../recordings/<session> --count 60
```

The window shows 60 evenly spaced frames, with the player you annotate in an orange box. For each keypoint
named at the top:

| Key | Action |
|---|---|
| click | place the keypoint |
| `n` | not visible (occluded or outside the image) |
| `u` | undo the last keypoint |
| `s` | skip this frame |
| `q` | save and quit; running the command again resumes |

"Left" and "right" always mean the **player's own** left and right. The image is mirrored as the tracker saw it.

**Movement onsets**, for the gesture delay (use a gestures session):

```bash
python -m evaluation.annotate events ../recordings/<session>
```

Find the **first frame where the movement physically starts** and press its key:

| Key | Action |
|---|---|
| `a` / `d` | previous / next frame (`A` / `D`: 10 frames) |
| space | play / pause |
| `c` | hand starts closing (grab) |
| `o` | hand starts opening (release) |
| `g` | hand sign formed (e.g. thumbs up) |
| `m` | arm starts moving |
| `h` | choose which hand the next marks are for: either / left / right |
| `x` | delete the marks on this frame |
| `v` | show/hide the tracker output (keep it **hidden** while marking, so it doesn't bias you) |
| `q` | save and quit |

Annotations are saved as `annotations_keypoints.json` / `annotations_events.json` in the session folder.
Use `--person ID` if two players were tracked and you want the other one, and `--scale` to zoom the window.

### Step 3: Measure the camera's own delay (once per camera)

```bash
python -m evaluation.camera_latency --trials 30
```

A fullscreen window flips between black and white. Point the webcam at it so it fills most of the image.
For a laptop's built-in camera, hold a mirror in front of the screen. At the end it prints the median,
e.g. `--camera-latency-ms 62`. Pass that number to the report in step 4. It includes about one screen refresh,
so treat it as an upper bound. If it says the camera barely sees the flips, move closer or disable auto-exposure.

### Step 4: Make the report for a session

```bash
python -m evaluation.report ../recordings/<session> --camera-latency-ms 62 --plots
```

Writes into the session folder:

- `report.md` with sections for detection rate, detection error, jitter, gesture delay, missing keypoints, and latency/rate
- `metrics.json` with all numbers, for your own tables or plots
- with `--plots`: `plot_wrist_trace.png` (raw vs. smoothed wrist) and `plot_latency.png` (latency histogram)

Sections that need annotations say so until you have made them. The first line lists the impact factors
of the session. If the session was recorded without `--shoulder-width-mm`, add `--shoulder-width-mm 380` here
to get the mm columns. How to read each section:

- **Error in mm**: px error scaled by the measured shoulder width on each frame. It is a 2D error in the image
  plane, exact for joints about as far from the camera as the shoulders; depth is not evaluated.

- **Jitter**: use the *stationary* columns for a stationary session and the *dynamic* columns for a dynamic one.
- **Latency**: only meaningful for live camera sessions recorded with `--no-preview`. For video files it shows
  processing time only, and the report says so.
- **Frame rate**: mean, 5% low and 1% low FPS (the average rate over the slowest 5% / 1% of frames), the
  slowest one-second window, and the share of seconds below 25 fps. The lows show stutter the mean hides.
- **Delay**: *algorithmic* = frames of smoothing/debouncing; *end-to-end* = camera delivers the frame → packet
  sent (live sessions only); *total* = end-to-end + camera delay.

### Step 5: Test the missing-keypoint handling

Use a **dynamic** session in which the arms and wrists are clearly visible most of the time:

```bash
python -m evaluation.missing_keypoints ../recordings/<dynamic session> --plots
```

This hides wrists, whole arms, or the whole pose for 0.1, 0.25, 0.5, 1 and 2 s on stretches where they
were clearly seen. It then compares what the pipeline fills in with where the joint really was, and
also scores the simple alternatives (hold last position, keep moving at the last speed, no handling).
Output: `missing_keypoints.md`, `.json` and with `--plots` `plot_missing_keypoints.png` (error vs. gap length).
Other gap lengths: `--gaps 0.5 1 3`. If a table is empty, the session had no stretches where those joints were
clearly visible long enough.

### Step 6: Compare lighting, distance and subjects

`compare` makes one row per session. With `--by`, it also makes a table with one row per level of a factor:
how many sessions (repetitions) it had, and the mean ± standard deviation of each number over them.
Factors: `condition`, `subject`, `distance`, `lux`, `brightness`, `noise`; several at once also work
(`--by subject distance`).

**Distance and subjects**:

```bash
python -m evaluation.compare ../recordings/*_stationary_* --by distance --out distance.md
```
```bash
python -m evaluation.compare ../recordings/*_stationary_* --by subject --out subjects.md
```

**Real lighting**: record the same protocol under each light (with `--lux`), then:

```bash
python -m evaluation.compare ../recordings/*_stationary_* --by lux --out lighting_real.md
```

**Simulated lighting**: replay one normally lit session darker. Replays reuse the original video and frame times:

```bash
python main.py --replay ../recordings/<session> --record --no-preview --condition sim_b1.0 --brightness 1.0
```
```bash
python main.py --replay ../recordings/<session> --record --no-preview --condition sim_b0.4 --brightness 0.4 --noise 4
```
```bash
python main.py --replay ../recordings/<session> --record --no-preview --condition sim_b0.2 --brightness 0.2 --noise 8
```
```bash
python -m evaluation.compare ../recordings/*_sim_* --by brightness --reference ../recordings/<..._sim_b1.0> --out lighting_sim.md
```

Use the **brightness 1.0 replay** as `--reference`, not the original session, so differences from
re-encoding the video are not counted as lighting effects. `--gamma` (>1 darkens shadows) is also available.
The per-session table goes from brightest to darkest. A replay keeps the subject, distance and lux of the
session it replayed, so use `--by brightness` (not `lux`) for simulated lighting. Replays use the annotations of the session
they replayed, so PCK also works for them.

### Step 7 (optional): Check the rate Unity receives

Close Unity first, since it uses the same port. Start the tracker, then:

```bash
python -m evaluation.udp_monitor --seconds 30
```

Prints received packets per second, 5% and 1% low FPS, interval spread, packet sizes, and the latency from
capture to arrival.

### Quick reference

| Command | Purpose |
|---|---|
| `python main.py --record --condition X --subject P1 --distance M --lux L --shoulder-width-mm MM --no-preview [--duration S]` | record a live session |
| `python main.py --video FILE --record ...` | record from a video file |
| `python main.py --replay SESSION --record [--brightness B --noise N --gamma G]` | replay a session, optionally darker |
| `python -m evaluation.annotate keypoints SESSION [--count N]` | click ground-truth keypoints |
| `python -m evaluation.annotate events SESSION` | mark movement onsets |
| `python -m evaluation.camera_latency` | camera delay |
| `python -m evaluation.report SESSION [--camera-latency-ms N] [--shoulder-width-mm MM] [--plots]` | all metrics for one session |
| `python -m evaluation.missing_keypoints SESSION [--plots]` | synthetic occlusion test |
| `python -m evaluation.compare SESSIONS... [--by FACTOR...] [--reference SESSION]` | tables per lighting, distance, subject |
| `python -m evaluation.udp_monitor` | rate as received (Unity closed) |

### Things to keep in mind

- Movement onsets are only as precise as one frame (about 33 ms at 30 fps).
- The person a tool analyses defaults to the player tracked longest. Use `--person ID` for the other one.
- Recording the raw video costs a little CPU. For the cleanest latency numbers, record one extra session
  with `--no-record-video` and compare.
- Each hand's grab state updates at about half the gesture rate, because the recognizer alternates between hands.
