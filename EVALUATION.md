# Evaluating the tracking module

How the motion-tracking module is measured: what each metric means, how it is implemented, how to record
the data and what to report. Run all commands from `src/` with the project's virtual environment.

## Workflow

1. **Record** a session: `python main.py --record --condition <label> --subject P1 --distance 2.0 --lux 450
   --shoulder-width-mm 380 --no-preview`.
   This writes `../recordings/<date>_<label>/` with `meta.json` (condition, impact factors, config values), `frames.jsonl`
   (per frame: timestamps, stage timings, frame brightness and, per player, the pose before smoothing, the
   pose that was sent and the hand states) and `raw.mp4` (the camera frames, for annotation and replay).
   Writing runs on a background thread so the recording hardly changes the timings it measures.
2. **Annotate** ground truth where a metric needs it: `python -m evaluation.annotate keypoints|events <session>`.
3. **Report**: `python -m evaluation.report <session> [--camera-latency-ms N] [--plots]` writes
   `report.md` and `metrics.json` into the session folder.
4. Separate tools: `evaluation.missing_keypoints` (synthetic occlusion test), `evaluation.compare` (tables
   per lighting level, distance, subject, with the spread over repetitions), `evaluation.camera_latency` (camera delay), `evaluation.udp_monitor` (rate as Unity receives it).

`python main.py --replay <session>` runs a recorded video back through the whole pipeline with the
original frame times, so changed settings or simulated lighting are tested on exactly the same input.
Distances are reported in pixels and in **% of shoulder width (%SW)**, which does not depend on how far the
player stands from the camera, and in **mm** when the player's shoulder width is known: measure it with a
tape from shoulder joint to shoulder joint (the points you click when annotating, roughly 1-2 cm inside the
bony edge of each shoulder) and pass `--shoulder-width-mm` when recording or to the report. Every px error
is then scaled by that frame's shoulder width, so mm values are comparable between distances. They are 2D
errors in the image plane: exact for joints about as far from the camera as the shoulders, and they say
nothing about depth (z is not evaluated). The tape measurement itself is uncertain by a few percent, which
scales all mm values by the same amount.

## Recording protocol

Use a tripod, the game's camera position and resolution (640x480), and one player first, then two. Per
lighting condition, record:

| Session | Duration | What the player does |
|---|---|---|
| `stationary_<light>` | 3 x 10 s | Stand still: arms hanging, arms stretched out sideways, both hands in front at grab height |
| `dynamic_<light>` | 30-60 s | Slow arm waves, fast reaches to the sides and up, reaching across the body |
| `gestures_<light>` | 2-3 min | Per hand 20 grab + release cycles, 10 thumbs-up, 10 arm moves from rest, pausing ~1 s between |

Measure the light level at the player with a phone lux meter app; it goes in `--lux` (see below).

### Impact factors: repetition, lighting, distance, subjects

Store each factor as a field, not only in the notes, so the results can be grouped by it:

| Factor | Option | Suggested levels |
|---|---|---|
| Subject | `--subject P1` (an anonymous code) | at least 3 players, differing in height, build and clothing (long sleeves vs. short, dark vs. light) |
| Distance | `--distance 2.0` (metres, floor to the camera) | the game's range, e.g. 1.5, 2.5 and 3.5 m |
| Lighting | `--lux 450` | see section 6 |
| Repetition | the same options recorded several times | 3 repetitions per combination |
| Shoulder width | `--shoulder-width-mm 380` | per subject, for errors in mm |

Vary one factor at a time from a baseline (e.g. P1, 2.5 m, office light) instead of recording every
combination. A replay inherits the factors of the session it replays.

`python -m evaluation.compare <sessions...> --by distance` (or `subject`, `lux`, `brightness`, `condition`,
or several such as `--by subject distance`) adds a table with one row per level: the number of sessions
(repetitions) and the mean ± standard deviation of pose found, keypoint and wrist detection, wrist jitter,
PCK@0.2, error in mm, 1% low FPS and the deviation from a reference. The standard deviation over
repetitions tells you whether a difference between two levels is bigger than the run-to-run variation.

---

## 1. Keypoint detection rate and error

**Keypoints.** The pipeline keeps MediaPipe landmarks 0-24 (legs are dropped). Evaluated (`session.KEYPOINTS`):

| Keypoint | MediaPipe index | Why |
|---|---|---|
| nose | 0 | head position / face crop |
| left/right shoulder | 11, 12 | body anchor, scale (shoulder width) |
| left/right elbow | 13, 14 | arm IK in Unity |
| left/right wrist | 15, 16 | **the chef's hands: placing and grabbing items (game-critical)** |
| left/right hip | 23, 24 | torso orientation |

"Left" is the player's own left. Coordinates are in the mirrored image the tracker processes.

**Detection rate** (no ground truth needed): the share of the pose model's runs on a player in which the
keypoint came out with visibility ≥ 0.5 and inside the image. Also reported: how often the player's pose was
found at all, how often the person box was frozen, and for the sent pose the share of measured vs. inferred joints.

**Detection error** (needs ground truth): `python -m evaluation.annotate keypoints <session> --count 60` shows
60 evenly spaced frames; you click each keypoint or press `n` when it cannot be seen. Per keypoint, for
the pose model's output and for the sent (smoothed) pose:

- mean and median pixel error, mean error in %SW (normalised mean error) and, with the shoulder width,
  mean and median error in mm
- **PCK@0.1 / 0.2 / 0.5**: the share of visible ground-truth keypoints predicted within 0.1 / 0.2 / 0.5
  shoulder widths (a frame without a pose counts as a miss)
- recall and precision of the visibility flag: does visibility ≥ 0.5 mean the joint really can be seen?
  A confident prediction of a hidden joint is a false positive, which is what makes the game follow a
  phantom hand.

Click accuracy limits the error you can measure: annotate 10 frames twice and report that difference as the floor.

**Report:** the per-keypoint table (detection rate, mean error px, %SW and mm, PCK@0.2, precision/recall), for raw
and sent, one player and two players, and the error per distance and per subject (`compare --by`). Emphasise
the wrists.

## 2. Jitter (stationary and dynamic)

Computed on frames where the pose model clearly saw the joint (visibility ≥ 0.5), for the raw output and
the sent signal on the same frames, so the effect of the One Euro filter is visible directly.

- **Stationary** (player holds still, so all motion is noise): RMS distance from a centred 1 s moving
  average, which removes slow body sway, plus the RMS and 95th percentile of the frame-to-frame step.
- **Dynamic** (player moves): noise is separated from real motion as the residual from a zero-phase
  Savitzky-Golay fit (2nd order, 0.3 s). Real arm motion is smooth at that scale; frame-to-frame jitter is not.
  The mean speed is listed to show how much motion there was.
- **Reduction** = 1 − sent/raw, and **lag**: how far the sent signal trails the raw one (cross-correlation of
  their velocities, sub-frame). Smoothing trades jitter for lag, so always report both together.

**Report:** for wrists, elbows and shoulders: stationary RMS (px and %SW) raw vs. sent, dynamic residual RMS
raw vs. sent, reduction and lag. Use `--plots` for a trace of the raw and sent wrist.

## 3. Motion / gesture recognition delay

The delay from a physical movement to a usable signal adds up from several parts. The table shows where each is measured:

| Part | Measured by |
|---|---|
| Camera: light → frame delivered to the program | `python -m evaluation.camera_latency` (flashes the screen in front of the camera; includes ~1 display refresh, so it is an upper bound) |
| Waiting for the loop + processing until the UDP packet is sent | logged per frame (`t_capture` → `t_send`) |
| Algorithmic: frames until the debounced/smoothed signal changes | annotated onsets vs. the sent signal |
| Network to Unity | `python -m evaluation.udp_monitor` (local UDP, under 1 ms) |

**Annotation:** `python -m evaluation.annotate events <session>` steps through the video frame by frame. Mark the
first frame where the movement physically starts: `c` fingers start to curl, `o` they start to open,
`g` a hand sign is formed, `m` the arm starts to move. Keep the tracker output hidden while marking (`v`
toggles it), so it does not bias you. Onsets are accurate to about one frame (33 ms at 30 fps).

**Matching** (`metrics.gesture_delay`): the response is the first frame within 1.5 s where the sent
signal shows the event: the hand's `closed` flag flips, a sign is reported, or the sent wrist has left its
rest position by 0.1 shoulder widths. For `move` the raw output is measured too, which separates the
smoothing's share. Flips of `closed` that no annotated onset explains are counted as **false triggers**.

Expected from the settings: `closed` needs about 2 readings (smoothing 0.6, threshold 0.6) and each hand is read
every other frame, so about 3-4 frames (100-130 ms at 30 fps). Signs need 3 identical readings, about 5-6
frames (170-200 ms).

**Report:** per event type: count, detection rate, delay in frames, algorithmic / end-to-end / total delay
(median and p95), false triggers per minute, plus the components above.

## 4. Missing keypoints

### How the pipeline handles them

Implemented in `core/pose_smoother.py`, `core/pose_ownership.py` and `trackers/person_tracker.py`:

1. **Confidence gate per joint.** Trust = min(visibility, presence), set to 0 for joints more than 5% of the frame
   height outside the image (MediaPipe extrapolates those). Trust below 0.35 means inferred, above 0.65
   means measured, with a smooth cross-fade in between so joints do not pop.
2. **Skeleton-based inference.** An untrusted elbow is placed at its shoulder plus the learned bone length,
   pointing in its last direction and relaxing towards "hanging down" over 1.5 s. Bone lengths are learned
   only from clearly seen joints, as a slow maximum, so foreshortening does not shrink them.
   Untrusted **wrists hold their last position** (they drive gameplay, so a lost hand should stay put).
   Face and hand points follow their anchor with the last measured offset.
3. **Implausibility gates.** A wrist with a forearm over 1.35x its learned length, or one moving faster than
   6 body heights/s, has its trust cut unless the reading persists for more than 2 frames.
   Wrists that are out of reach, or that the other player owns, are marked unseen (`pose_ownership`).
4. **Dropouts.** If no pose is found, the last pose is held for 0.4 s, then the pose is marked inactive.
   After 1 s without a pose the filters restart. A lost person box is frozen for up to 28 frames.
5. **Output.** Every sent landmark carries `visibility` (0 = inferred, 1 = measured), so Unity can fade IK weights.

### How it is evaluated

- **As it occurred** (`report.py`, section 4): per keypoint, the share of sent frames that were inferred,
  how long the gaps were, and the **re-entry pop** (the jump when a joint is measured again) next to the
  normal 95th-percentile step. Also the share of whole-pose dropouts.
- **Synthetic occlusion with ground truth** (`python -m evaluation.missing_keypoints <dynamic session> --plots`).
  Real occlusions have no ground truth, so joints that were clearly seen are hidden for 0.1-2 s, the
  recorded raw poses are replayed through the smoother, and the result is compared with the unmasked replay.
  The replay reproduces the live output to within about 2 px (log rounding). Scenarios: `hand` (wrist and
  hand points), `arm` (with the elbow), `pose` (whole pose missing). Methods compared: the pipeline;
  `relax` (wrists hang down instead of holding); baselines `hold-last`, `const-velocity` extrapolation, and
  `drop` (no handling, joint unavailable).

**Report:** error vs. gap length per method (mean and end-of-gap error in %SW), availability, the share of
gap frames correctly flagged as inferred, recovery time after the gap, and the re-entry pop. Add the
as-occurred statistics per lighting condition.

## 5. Module latency and update rate

From the per-frame timestamps of a **live** session recorded with `--no-preview`. The preview window adds
drawing and display time; record with it on as well if you want to show that cost.

- **End-to-end latency**: camera delivers a frame → its UDP packet is sent, split into waiting for the loop and processing.
- **Per-stage time** (person detector, which runs every 2nd frame, so its "ran" and "predicted" frames are
  listed separately; pose; smoother; face; gestures; network) on the frames where the stage ran.
- **Output rate** (packets/s) and **interval** spread (rate jitter); **camera rate** and stale frames skipped.
- **Low-percentile frame rate**: the mean only shows the average; stutter shows in the slow frames. **5% / 1%
  low** = the average rate over the slowest 5% / 1% of packet intervals (the usual measure in games), plus the
  slowest one-second window and the share of seconds below 25 fps. `udp_monitor` prints the same as received.
- **Effective update rates**: the pose is re-measured at the pose rate, but each hand's grab state only at
  about half the gesture rate, because the recognizer alternates hands.
- The first second (model warm-up) is left out.

**Report:** mean, median, p95, p99 and max of end-to-end latency; the stage table; output rate as mean, 5% and
1% low FPS; camera, pose and hand rates; the hardware (CPU/GPU, CUDA/MPS) and resolution; one vs. two players (the cost per player is
roughly the pose + face + gesture stages). Cross-check the rate Unity receives with `udp_monitor` (Unity closed).

## 6. Influence of lighting conditions

Two complementary experiments:

- **Real lighting.** Repeat the protocol under e.g. bright (~500 lux, office), dim (~50 lux, one lamp),
  backlit (window behind the player) and side-lit. Record the lux value with `--lux`; the mean frame
  brightness is logged automatically. Annotate ~30 keypoint frames per condition for PCK.
- **Simulated lighting on identical input.** Replay one normally lit session at decreasing brightness, with
  sensor noise like a camera raising its gain:

  ```bash
  python main.py --replay ../recordings/<session> --record --no-preview --condition sim_b1.0 --brightness 1.0
  python main.py --replay ../recordings/<session> --record --no-preview --condition sim_b0.4 --brightness 0.4 --noise 4
  python main.py --replay ../recordings/<session> --record --no-preview --condition sim_b0.2 --brightness 0.2 --noise 8
  ```

  Each run is compared frame by frame with the brightness-1.0 replay (use that replay as the reference,
  not the original session, so video re-encoding is not counted as an error).

`python -m evaluation.compare <sessions...> --by lux|brightness [--reference <sim_b1.0 session>] --out lighting.md`
makes a table per lighting level (mean ± SD over repetitions) and one row per session: brightness, player tracked, pose found, keypoint and wrist detection rate, wrist visibility,
hand recognized ("hand seen"), stationary wrist jitter, PCK and mm error against annotations, FPS, and the
deviation from the reference. Use `--by lux` (or `condition`) for real lighting and `--by brightness` for simulated.

**Report:** the table and a plot of detection rate / hand-seen rate against brightness or lux. Say at which level
each part degrades: usually the hand recognizer first, then the wrists, then the whole pose.

## What to report: checklist

- [ ] Experimental design: subjects, distances, lighting levels, repetitions per combination, camera and hardware
- [ ] Keypoint definition table and the annotation protocol (frames, annotator floor)
- [ ] Detection rate, error (px, %SW, mm), PCK@0.2, visibility precision/recall: raw vs. sent, 1 and 2 players
- [ ] Stationary and dynamic jitter, raw vs. sent, reduction and the lag it costs
- [ ] Gesture/motion delay per event type, split into camera, processing and algorithmic parts, plus false triggers/min
- [ ] How missing keypoints are handled (section 4), with the synthetic error-vs-gap-length results and the as-occurred statistics
- [ ] Latency distribution, per-stage times, output rate (mean, 5% and 1% low FPS) and update rates, hardware
- [ ] Results per distance and per subject (mean ± SD over repetitions)
- [ ] Lighting table (real and simulated) and the brightness at which tracking breaks down
