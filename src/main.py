import argparse
import cv2
import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import base64

import config
import utils
from core.data_packer import DataPacker
from core.latest_frame_capture import LatestFrameCapture
from core.macos_priority import keep_tracker_responsive
from core.timed_video_writer import TimedVideoWriter
from core.network_streamer import UDPStreamer
from core.perf_monitor import PerformanceMonitor
from core.pose_smoother import PoseSmoother
from core.session_recorder import (SessionRecorder, SessionReplayCapture, frame_luma, pose_rows,
                                   simulate_lighting)
from core import pose_ownership
from trackers.face_tracker import FaceTracker
from trackers.hand_gesture_tracker import HandGestureTracker
from trackers.person_tracker import PersonTracker
from trackers.pose_tracker import PoseTracker

# Base64 grows data by 4/3; this leaves room for the face images in the 65507-byte UDP limit
MAX_JPEG_BYTES = 30000
MAX_UDP_PACKET_BYTES = 65000


def get_asset_path(relative_path: str) -> str:
    """Resolve correct asset path when running normally or from a PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PANdemoniYUM motion tracking engine")
    ev = parser.add_argument_group("evaluation (see EVALUATION.md)")
    ev.add_argument("--record", action="store_true", help="record a session for the metrics in src/evaluation")
    ev.add_argument("--condition", default="default",
                    help="label stored with the session, e.g. stationary_office or dynamic_dim")
    ev.add_argument("--notes", default="", help="free text stored with the session")
    ev.add_argument("--subject", help="who is playing (a code such as P1), for grouping results by subject")
    ev.add_argument("--distance", type=float, metavar="M", help="player's distance from the camera in metres")
    ev.add_argument("--lux", type=float, help="measured light level at the player")
    ev.add_argument("--shoulder-width-mm", type=float, metavar="MM",
                    help="player's measured shoulder width (joint to joint), to report errors in mm")
    ev.add_argument("--recordings-dir", default=config.RECORDINGS_DIR)
    ev.add_argument("--no-record-video", action="store_true", help="do not store the raw camera video")
    ev.add_argument("--replay", metavar="SESSION", help="run on a recorded session's video with its frame times")
    ev.add_argument("--video", metavar="PATH", help="run on a video file instead of the camera")
    ev.add_argument("--brightness", type=float, default=1.0, help="simulated lighting: intensity scale")
    ev.add_argument("--gamma", type=float, default=1.0, help="simulated lighting: gamma (>1 darkens shadows)")
    ev.add_argument("--noise", type=float, default=0.0, help="simulated lighting: sensor noise std (0-255)")
    ev.add_argument("--duration", type=float, help="stop after this many seconds")
    ev.add_argument("--no-preview", action="store_true", help="no preview window (lower, cleaner latency)")
    ev.add_argument("--tracking-video", metavar="PATH",
                    help="save the annotated tracking view as a real-time video (used by play_session.py)")
    return parser.parse_args(argv)


def open_video_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.replay:
        return SessionReplayCapture(args.replay)
    if config.USE_CAMERA:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Always process the newest camera frame so tracking delay doesn't build up
        return LatestFrameCapture(cap)
    else:
        cap = cv2.VideoCapture(config.VIDEO_PATH)
    return cap

def open_video_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    if not output_path.parent.exists():
        raise Exception("Save path folder does not exist")
    
    print(f"Saving output video to: {output_path}")

    # Initialize video writer for test_output.mp4 using mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback FPS for webcams or unread video headers

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return out


def frame_time(cap: cv2.VideoCapture, frame_count: int) -> float:
    """
    Timestamp of the current frame in seconds, used by the pose filters.
    Camera: wall clock. Video file: position in the video, so playback speed does not matter.
    """
    if config.USE_CAMERA:
        return time.perf_counter()
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms and pos_ms > 0:
        return pos_ms / 1000.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    return frame_count / (fps if fps and fps > 0 else 30.0)


def clamped_box(person, frame: np.ndarray) -> tuple[int, int, int, int]:
    """The person box clipped to the frame: the exact region the pose/face models run on."""
    return (
        max(0, person.x1),
        max(0, person.y1),
        min(frame.shape[1], person.x2),
        min(frame.shape[0], person.y2),
    )


def pose_box(person, frame: np.ndarray) -> tuple[int, int, int, int]:
    """
    The person box widened (and raised) so arms reaching out of the detector box stay inside the
    pose model's crop, clipped to the frame. Pose landmarks are normalised to this box.
    """
    w, h = person.x2 - person.x1, person.y2 - person.y1
    pad_x = int(w * config.POSE_CROP_PADDING_X)
    pad_top = int(h * config.POSE_CROP_PADDING_TOP)
    return (
        max(0, person.x1 - pad_x),
        max(0, person.y1 - pad_top),
        min(frame.shape[1], person.x2 + pad_x),
        min(frame.shape[0], person.y2),
    )


def config_snapshot() -> dict:
    """The plain config values, stored with a recorded session."""
    return {name: value for name, value in vars(config).items()
            if name.isupper() and isinstance(value, (bool, int, float, str, list, type(None)))}


def session_record(frame_count, now, t_capture, t_read, t_send, cap, perf, frame, scene_data, raw_poses,
                   landmark_frame, run_gestures, detector_ran) -> dict:
    """One line of a recorded session's frames.jsonl (see core/session_recorder.py)."""
    persons = []
    for person in scene_data.persons:
        persons.append({
            "id": person.id,
            "box": [person.x1, person.y1, person.x2, person.y2],
            "crop": list(person.crop) if person.crop else None,
            "tracking": person.is_tracking,
            "frozen": person.frozen,
            # pose_ran: the pose model ran on this person this frame; raw is None if it found no (own) pose
            "pose_ran": landmark_frame and person.id in raw_poses,
            "raw": pose_rows(raw_poses.get(person.id), with_presence=True) if landmark_frame else None,
            "pose": pose_rows(person.pose),
            "face": person.face.tracking_active,
            "hands": person.hands,
        })
    return {
        "i": frame_count,
        "t": now,
        "t_capture": t_capture,
        "t_read": t_read,
        "t_send": t_send,
        "dropped": getattr(cap, "dropped", 0),
        "w": frame.shape[1],
        "h": frame.shape[0],
        "luma": round(frame_luma(frame), 2),
        "landmark_frame": landmark_frame,
        "gestures_ran": run_gestures,
        "detector_ran": detector_ran,
        "stages": {name: round(seconds * 1000, 3) for name, seconds in perf.snapshot().items()},
        "persons": persons,
    }


def main(args: argparse.Namespace):
    if args.replay or args.video:
        config.USE_CAMERA = False
        if args.video:
            config.VIDEO_PATH = args.video
    if args.no_preview:
        config.SHOW_PREVIEW = False

    # Before any threads start: they inherit the priority
    if keep_tracker_responsive():
        print("Tracker marked as latency-critical (no App Nap while the game is in front)")

    streamer = UDPStreamer(ip=config.UDP_IP, port=config.UDP_PORT)
    packer = DataPacker()

    model_dir = get_asset_path(config.MODEL_DIR)
    face_tracker = FaceTracker(
        model_dir=model_dir,
        num_faces=config.FACE_NUM_FACES,
        min_tracking_confidence=config.FACE_MIN_CONFIDENCE,
        blendshapes=config.FACE_BLENDSHAPES
    )

    person_tracker = PersonTracker(
        model_dir=model_dir,
        num_persons=config.PERSON_NUM_PERSONS,
        frame_skip=config.PERSON_DETECTION_FRAME_SKIP,
        freeze_frames=config.PERSON_FREEZE_FRAMES,
    )

    pose_tracker = PoseTracker(
        model_dir=model_dir,
        num_poses=config.POSE_NUM_POSES,
        min_tracking_confidence=config.POSE_MIN_CONFIDENCE,
        model_file=config.POSE_MODEL_FILE
    )

    pose_smoother = PoseSmoother(
        enabled=config.POSE_SMOOTHING_ENABLED,
        min_cutoff=config.POSE_MIN_CUTOFF,
        beta=config.POSE_BETA,
        speed_floor=config.POSE_SPEED_FLOOR,
        vis_low=config.POSE_VISIBILITY_LOW,
        vis_high=config.POSE_VISIBILITY_HIGH,
        rest_blend_seconds=config.POSE_REST_BLEND_SECONDS,
        dropout_hold_seconds=config.POSE_DROPOUT_HOLD_SECONDS,
        reset_seconds=config.POSE_RESET_SECONDS,
        hold_unseen_hands=config.POSE_HOLD_UNSEEN_HANDS,
        max_forearm_stretch=config.POSE_MAX_FOREARM_STRETCH,
        max_hand_speed=config.POSE_MAX_HAND_SPEED,
    )

    gesture_tracker = None
    if config.GESTURE_ENABLED:
        gesture_tracker = HandGestureTracker(
            model_dir=model_dir,
            min_score=config.GESTURE_MIN_SCORE,
            crop_scale=config.GESTURE_CROP_SCALE,
            sign_min_score=config.GESTURE_SIGN_MIN_SCORE,
            sign_confirm_readings=config.GESTURE_SIGN_CONFIRM_READINGS,
        )

    if config.LANDMARK_FRAME_SKIP < 1:
        raise ValueError("LANDMARK_FRAME_SKIP must be at least 1")

    cap = open_video_capture(args)
    out = None

    if config.SAVE_PREVIEW:
        out = open_video_writer(cap, Path(config.SAVE_PATH))

    recorder = None
    if args.record:
        source = "replay" if args.replay else ("camera" if config.USE_CAMERA else "video")
        recorder = SessionRecorder(args.recordings_dir, args.condition, {
            "notes": args.notes,
            "factors": {"subject": args.subject, "distance_m": args.distance, "lux": args.lux,
                        "shoulder_width_mm": args.shoulder_width_mm},
            "source": source,
            "source_session": str(Path(args.replay).resolve()) if args.replay else None,
            "video_path": config.VIDEO_PATH if source == "video" else None,
            "lighting": {"brightness": args.brightness, "gamma": args.gamma, "noise": args.noise},
            "preview": config.SHOW_PREVIEW,
            "config": config_snapshot(),
        }, record_video=not args.no_record_video and not args.replay)
    tracking_video = TimedVideoWriter(args.tracking_video) if args.tracking_video else None
    # perf_counter times -> wall clock, to line the tracking video up with the gameplay recording
    epoch_offset = time.time() - time.perf_counter()
    # Fixed seed: the same simulated noise on every replay
    rng = np.random.default_rng(0)

    print("Tracking pipeline started. Press 'q' to exit.")

    perf = PerformanceMonitor()

    frame_count = 0
    last_faces = {}
    send_video = config.VIDEO_DEFAULT_ON
    t_begin = None

    try:
        while cap.isOpened():
            perf.start()
            perf.next("Frame capture")

            success, frame = cap.read()
            if not success:
                break
            t_read = time.perf_counter()
            # When the camera delivered the frame (stamped by the camera reader; a file frame when it was read)
            t_capture = getattr(cap, "capture_time", 0.0) or t_read
            t_begin = t_begin or t_read
            if args.duration and t_read - t_begin > args.duration:
                break
            raw_frame = frame
            frame = simulate_lighting(frame, args.brightness, args.gamma, args.noise, rng)

            frame_count += 1
            frame = cv2.flip(frame, 1)
            now = frame_time(cap, frame_count)

            perf.next("Person tracker")

            # Unity tells us whether it currently shows the camera image
            for message in streamer.poll_messages():
                if message == "video:1":
                    send_video = True
                elif message == "video:0":
                    send_video = False

            # 1. Process tracking components
            scene_data = person_tracker.process_frame(frame)
            if not person_tracker.ran_detection:
                # Detector skipped this frame: keep showing the cost of the last full run
                perf.keep_previous("Person tracker")
            notes = {}
            run_gestures = gesture_tracker is not None and (frame_count - 1) % config.GESTURE_FRAME_SKIP == 0

            landmark_frame = (frame_count - 1) % config.LANDMARK_FRAME_SKIP == 0

            # Pass 1: crops and raw poses for every player
            active = []  # (person, pose box, face crop, raw pose or None)
            for person in scene_data.persons:
                if not person.is_tracking or (person.x2 - person.x1) < 20 or (person.y2 - person.y1) < 20:
                    continue

                # TODO determine better/smaller subframe for face detection based on datapoints 0-10 from the pose detection output!

                # Face runs on the tight person box (padding could pull in the other player's face);
                # pose runs on the padded box so outstretched hands stay in view
                x1, y1, x2, y2 = clamped_box(person, frame)
                face_frame = frame[y1:y2, x1:x2]
                box = pose_box(person, frame)
                px1, py1, px2, py2 = box
                subframe = frame[py1:py2, px1:px2]
                person.crop = box

                if face_frame.shape[0] < 20 or face_frame.shape[1] < 20:
                    print(f"sf {face_frame.shape[0]}, {face_frame.shape[1]}")
                    continue

                raw_pose = None
                if landmark_frame:
                    perf.next("Pose tracker")
                    candidates = pose_tracker.process_frame_candidates(subframe)
                    # The crop can contain the other player (e.g. when reaching over them): pick our own body
                    own_torso = pose_smoother.torso(person.id, frame.shape[0], now)
                    other_torsos = [t for t in (pose_smoother.torso(p.id, frame.shape[0], now)
                                                for p in scene_data.persons if p.id != person.id) if t is not None]
                    raw_pose = pose_ownership.select_pose(candidates, box, own_torso, other_torsos,
                                                          config.POSE_MAX_TORSO_JUMP)
                    if raw_pose is not None:
                        raw_pose = pose_ownership.editable(raw_pose)
                        pose_ownership.enforce_arm_reach(raw_pose, box,
                                                         pose_smoother.shoulder_reference(person.id, frame.shape[0]))
                active.append((person, box, face_frame, raw_pose))

            # Pass 2: a hand both players claim belongs to only one of them
            if landmark_frame:
                pose_ownership.resolve_shared_wrists(
                    [(person.id, raw, box) for person, box, _, raw in active if raw is not None])

            # Pass 3: smoothing, face and gestures per player
            for person, box, face_frame, raw_pose in active:
                if landmark_frame:
                    perf.next("Pose smoother")
                    person.pose = pose_smoother.update(person.id, raw_pose, box, frame.shape, now)

                    perf.next("Face tracker")
                    person.face = face_tracker.process_frame(face_frame)
                    if config.FACE_IMAGE_ENABLED and person.face.tracking_active:
                        person.face.image, person.face.mustache = FaceTracker.aligned_face_image(
                            face_frame, person.face.landmarks, config.FACE_IMAGE_SIZE, config.FACE_IMAGE_MAX_BYTES)
                    last_faces[person.id] = person.face
                else:
                    # Between detections the smoothed pose is re-placed into the person's current box
                    reprojected = pose_smoother.reproject(person.id, box, frame.shape, now)
                    if reprojected is not None:
                        person.pose = reprojected
                    if person.id in last_faces:
                        person.face = last_faces[person.id]

                if gesture_tracker is not None:
                    if run_gestures:
                        perf.next("Gesture tracker")
                        gesture_tracker.update(person.id, person.pose, box, frame)
                    person.hands = gesture_tracker.hands_dict(person.id)

                # TODO for demonstration only
                if "jawOpen" in person.face.blendshapes and person.face.blendshapes["jawOpen"] > 0.05:
                    notes[person.id] = "(JAW OPEN)"
                signs = [h.get("gesture") for h in person.hands.values() if h.get("gesture", "None") != "None"]
                if signs:
                    notes[person.id] = f"({', '.join(signs)})"

            if gesture_tracker is not None:
                gesture_tracker.forget({p.id for p in scene_data.persons if p.is_tracking})

            perf.next("Network")

            # 2. Add tracking models incrementally
            # Epoch capture time, so a receiver on this machine can measure the end-to-end latency
            scene_data.capture_time = int((time.time() - (time.perf_counter() - t_capture)) * 1000)
            scene_data.processing_time = int((time.perf_counter() - t_capture) * 1000)
            # packer.add(face_data)
            packer.add(scene_data)
            # packer.add(gesture_data) # Ready for other modules

            packer._frame_data["frame_width"] = frame.shape[1]
            packer._frame_data["frame_height"] = frame.shape[0]

            if send_video:
                # Aspect-aware resize: Scale height to 240, adjust width proportionally
                scale = 240.0 / frame.shape[0]
                small_w = int(frame.shape[1] * scale)
                small_frame = cv2.resize(frame, (small_w, 240))

                # Compress as JPEG, lowering quality until it fits comfortably in one UDP datagram
                for quality in (40, 30, 20, 10):
                    success, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if success and len(buffer) <= MAX_JPEG_BYTES:
                        break
                if success and len(buffer) <= MAX_JPEG_BYTES:
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    packer.add_frame_image(frame_b64)

            # 3. Serialise and send accumulated payload
            frame_image = packer._frame_data.get("frame_base64")
            packet = packer.pack_frame()
            if packet and len(packet) > MAX_UDP_PACKET_BYTES and frame_image:
                # Too big for one datagram (e.g. two faces plus the calibration video): drop this camera frame
                packer.add(scene_data)
                packer._frame_data["frame_width"] = frame.shape[1]
                packer._frame_data["frame_height"] = frame.shape[0]
                packet = packer.pack_frame()
            if packet:
                streamer.send(packet)

                # 4. Debug network packages to terminal
                if config.DEBUG_MODE:
                    #print(f"[NETWORK DEBUG] {packet.decode('utf-8')}")
                    pass
            t_send = time.perf_counter()

            if recorder is not None:
                raw_poses = {person.id: raw_pose for person, _, _, raw_pose in active}
                recorder.add(session_record(frame_count, now, t_capture, t_read, t_send, cap, perf, frame,
                                            scene_data, raw_poses, landmark_frame, run_gestures,
                                            person_tracker.ran_detection), raw_frame)

            # 5. Preview and Debug overlay (also drawn for the tracking video when there's no preview window)
            if config.SHOW_PREVIEW or tracking_video is not None:
                if config.DEBUG_MODE:
                    perf.next("Debug drawing")

                    person_tracker.draw_debug(frame, notes)

                    for person in scene_data.persons:
                        x1, y1, x2, y2 = clamped_box(person, frame)
                        color = utils.player_color_from_id(person.id)
                        FaceTracker.draw_landmarks(frame[y1:y2, x1:x2], person.face.landmarks, color)
                        px1, py1, px2, py2 = person.crop or pose_box(person, frame)
                        PoseTracker.draw_landmarks(frame[py1:py2, px1:px2], person.pose.landmarks, color)

                    perf.end()
                    perf.draw(frame)

                if config.SAVE_PREVIEW:
                    out.write(frame)
                if tracking_video is not None:
                    tracking_video.add(frame, t_capture + epoch_offset)

                if config.SHOW_PREVIEW:
                    cv2.imshow("Tracking Preview", frame)
                    if cv2.pollKey() & 0xFF == ord('q'):
                        break

    except KeyboardInterrupt:
        # Ctrl+C, or play_session.py ending the session: stop cleanly so the recordings are complete
        print("Tracker stopped.")
    finally:
        face_tracker.close()
        pose_tracker.close()
        if gesture_tracker is not None:
            gesture_tracker.close()
        streamer.close()
        cap.release()
        if out is not None:
            out.release()
        if recorder is not None:
            recorder.close()
        if tracking_video is not None:
            tracking_video.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parse_args())
