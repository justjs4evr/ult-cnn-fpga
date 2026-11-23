"""Webots driving from only the existing six-class ULT-CNN checkpoint."""

import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from controller import Keyboard, Supervisor


ROOT = Path(__file__).resolve().parent
TRAINING = ROOT.parent / "webot" / "ult_cnn_training"
sys.path.insert(0, str(ROOT))
from inference import CLASSES, image_tensor, load_model, preprocess_pil  # noqa: E402
from timed_stop_policy import TimedStopPolicy  # noqa: E402
from context_passage_policy import ContextPassagePolicy  # noqa: E402

DEFAULT_CHECKPOINT = TRAINING / "output_current_six_binary" / "best_model.pt"
SELECTED_MODEL = ROOT / "selected_model.json"
OUTPUT = ROOT / "webots_runs"
WHEELS = {
    "straight": (2.0, 2.0),
    "left": (0.75, 1.6),
    "right": (1.6, 0.75),
    "sharp_left": (-0.45, 1.5),
    "sharp_right": (1.5, -0.45),
    "stop": (0.0, 0.0),
}
MAX_SIM_TIME_S = float(os.environ.get("ULT_CNN_ONLY_TEST_SECONDS", "0"))
SNAPSHOT_LIMIT = max(0, int(os.environ.get("ULT_CNN_SNAPSHOT_LIMIT", "200")))
STOP_DIAGNOSTIC_LIMIT = max(0, int(os.environ.get("ULT_CNN_STOP_DIAGNOSTIC_LIMIT", "50")))


def selected_checkpoint():
    value = os.environ.get("ULT_CNN_CHECKPOINT")
    if not value and SELECTED_MODEL.is_file():
        value = json.loads(SELECTED_MODEL.read_text(encoding="utf-8"))["checkpoint"]
    path = Path(value).expanduser() if value else DEFAULT_CHECKPOINT
    return (path if path.is_absolute() else ROOT / path).resolve()


def camera_image(camera):
    raw = camera.getImage()
    if raw is None:
        raise RuntimeError("Central camera did not return an image")
    return Image.frombytes("RGBA", (camera.getWidth(), camera.getHeight()), raw, "raw", "BGRA").convert("RGB")


def main():
    torch.set_num_threads(1)
    checkpoint_path = selected_checkpoint()
    model, checkpoint = load_model(checkpoint_path)
    settings_path = ROOT / "user_settings.json"
    config = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
    stop_enabled = config.get("stop_enabled", False)
    if not isinstance(stop_enabled, bool):
        raise ValueError("stop_enabled must be true or false")
    context_config=checkpoint.get('context_policy',{})
    context_policy=(ContextPassagePolicy(float(context_config['activate_probability']),
                    float(context_config['keep_probability']),float(context_config['clear_seconds']))
                    if context_config.get('enabled') and not stop_enabled else None)
    speed_multiplier = float(config.get("drive_speed_multiplier", 1.5))
    inference_interval = float(config.get("cnn_inference_interval_seconds", 0.08))
    if any(not math.isfinite(value) or value <= 0 for value in (speed_multiplier, inference_interval)):
        raise ValueError("Driving speed and inference interval must be finite and positive")
    stop_policy = TimedStopPolicy(
        float(config.get("stop_hold_seconds", 1.5)),
        float(config.get("stop_clear_seconds", 1.0)),
        float(config.get("stop_minimum_pass_seconds", 3.0)),
    )
    print(f"CNN-only model: {checkpoint_path}", flush=True)
    print("Automatic CNN driving. Focus 3D view: Space=pause/resume, R=resume, Q=stop controller.", flush=True)
    print(f"Driving speed multiplier: {speed_multiplier:g}; inference interval: {inference_interval:g}s", flush=True)
    print(f"STOP: wait {stop_policy.hold_seconds:g} simulation seconds" if stop_enabled else "STOP: continue straight without waiting", flush=True)

    robot = Supervisor()
    output = OUTPUT / Path(robot.getWorldPath()).stem / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime_metadata.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "world": robot.getWorldPath(),
        "preprocessing": checkpoint["preprocessing"], "classes": CLASSES,
        "stop_policy": "Timed wait then passage" if stop_enabled else "STOP maps to straight without waiting",
        "stop_enabled": stop_enabled, "drive_speed_multiplier": speed_multiplier,
        "inference_interval_seconds": inference_interval,
        "architecture": checkpoint.get('architecture','ult_cnn_1440'),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "context_policy": context_config,
        "stop_hold_seconds": stop_policy.hold_seconds,
        "stop_clear_seconds": stop_policy.clear_seconds,
        "stop_minimum_pass_seconds": stop_policy.minimum_pass_seconds,
    }, indent=2), encoding="utf-8")
    step_ms = int(robot.getBasicTimeStep())
    camera = robot.getDevice("cam_center")
    camera.enable(step_ms)
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
    motor_limit = min(left_motor.getMaxVelocity(), right_motor.getMaxVelocity())
    node = robot.getSelf()
    keyboard = robot.getKeyboard()
    keyboard.enable(step_ms)
    previous_keys = set()
    paused = False
    sample_count = 0
    last_time = None
    snapshots = 0
    trajectory = (output / "trajectory.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(trajectory)
    writer.writerow(("sim_time_s", "world_x_m", "world_y_m", "predicted_class", "confidence", "left_speed", "right_speed",
                     *("score_" + label for label in CLASSES), "executed_class", "stop_phase", "world_heading_deg",
                     "crossing_context_probability", "crossing_context_active"))
    trajectory.flush()
    stop_diagnostics = 0
    last_stop_diagnostic_time = -float("inf")
    counts = Counter()
    next_inference = 0.0
    previous = None
    start_xy = None
    max_displacement = 0.0
    blank_frames = 0
    first_frame_diagnostic = None
    reason = "simulation_stopped"
    try:
        while robot.step(step_ms) != -1:
            now = robot.getTime()
            keys = set()
            while True:
                key = keyboard.getKey()
                if key == -1:
                    break
                keys.add(key & Keyboard.KEY)
            pressed = keys - previous_keys
            previous_keys = keys
            if ord(" ") in pressed:
                paused = not paused
                print("CNN driving PAUSED" if paused else "CNN driving ON", flush=True)
            if ord("R") in pressed:
                paused = False
            if ord("Q") in pressed:
                reason = "user_quit"
                break
            if MAX_SIM_TIME_S > 0 and now >= MAX_SIM_TIME_S:
                reason = "time_limit"
                break
            stop_policy.advance(now, paused=paused)
            if paused:
                if context_policy is not None and context_policy.active:context_policy.last_keep=now
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                continue
            if now < next_inference:
                continue
            image = camera_image(camera)
            if first_frame_diagnostic is None:
                raw = np.frombuffer(camera.getImage(), dtype=np.uint8)
                first_frame_diagnostic = {
                    "raw_min": int(raw.min()), "raw_max": int(raw.max()),
                    "raw_nonzero": int(np.count_nonzero(raw)),
                    "rgb_extrema": image.getextrema(),
                    "camera_size": (camera.getWidth(), camera.getHeight()),
                }
                camera.saveImage(str(output / "camera_native_first.png"), 100)
                image.save(output / "camera_pil_first.png")
            if max(upper for _, upper in image.getextrema()) == 0:
                blank_frames += 1
                if blank_frames >= 5:
                    reason = "blank_camera"
                    break
                next_inference = now + inference_interval
                continue
            blank_frames = 0
            with torch.inference_mode():
                probabilities = model(image_tensor(image, checkpoint["preprocessing"])).softmax(dim=1)[0]
            index = int(probabilities.argmax())
            label = CLASSES[index]
            confidence = float(probabilities[index])
            previous_phase = stop_policy.phase
            executed_label = stop_policy.choose(label) if stop_enabled else ("straight" if label == "stop" else label)
            context_probability=None
            if context_policy is not None:
                context_probability=float(model.last_context_probability[0])
                executed_label=context_policy.choose(context_probability,label,now)
            if not stop_enabled:
                stop_policy.phase = "crossing_context_passage" if context_policy is not None and context_policy.active else "stop_bypass"
            speeds = WHEELS[executed_label]
            if stop_policy.phase == "passing":
                # Still seeing the handled STOP scene: advance slowly to leave it.
                # If CNN reports a steering class, keep that steering at half speed.
                speeds = tuple(speed * 0.5 for speed in speeds)
            speeds = tuple(speed * speed_multiplier for speed in speeds)
            peak = max(abs(speed) for speed in speeds)
            if peak > motor_limit:
                speeds = tuple(speed * motor_limit / peak for speed in speeds)
            if stop_policy.phase != previous_phase:
                print(f"STOP phase: {previous_phase} -> {stop_policy.phase}", flush=True)
            left_motor.setVelocity(speeds[0])
            right_motor.setVelocity(speeds[1])
            position = node.getPosition()
            x, y = position[0], position[1]
            if start_xy is None:
                start_xy = (x, y)
            displacement = ((x - start_xy[0]) ** 2 + (y - start_xy[1]) ** 2) ** 0.5
            max_displacement = max(max_displacement, displacement)
            if previous is not None and label != previous:
                counts["class_changes"] += 1
            previous = label
            counts[label] += 1
            sample_count += 1
            last_time = round(now, 3)
            writer.writerow((last_time, round(x, 4), round(y, 4), label,
                             round(confidence, 4), speeds[0], speeds[1],
                             *(round(float(value), 6) for value in probabilities), executed_label, stop_policy.phase,
                             round(math.degrees(math.atan2(node.getOrientation()[3],node.getOrientation()[0])),3),
                             round(context_probability,6) if context_probability is not None else '',
                             context_policy.active if context_policy is not None else False))
            if label == "stop" and stop_diagnostics < STOP_DIAGNOSTIC_LIMIT and now - last_stop_diagnostic_time >= 5.0:
                stem = f"stop_diagnostic_{sample_count:06d}"
                image.save(output / (stem + ".png"))
                cnn_input = np.rint(preprocess_pil(image, checkpoint["preprocessing"]) * 255).astype(np.uint8)
                Image.fromarray(cnn_input).save(output / (stem + "_input16.png"))
                if checkpoint['preprocessing'].get('paired_context64'):
                    image.convert('L').resize((64,64),Image.Resampling.BOX).save(output/(stem+'_context64.png'))
                (output / (stem + ".json")).write_text(json.dumps({
                    "sim_time_s": last_time, "position": [x, y], "checkpoint": str(checkpoint_path),
                    "scores": dict(zip(CLASSES, map(float, probabilities))),
                    "ground_truth": None,
                }, indent=2), encoding="utf-8")
                stop_diagnostics += 1
                last_stop_diagnostic_time = now
                trajectory.flush()
            if sample_count % 10 == 0:
                trajectory.flush()
                print(f"t={now:.2f} class={label} confidence={confidence:.3f} action={executed_label} phase={stop_policy.phase} xy=({x:.3f},{y:.3f})", flush=True)
            if sample_count % 30 == 1 and snapshots < SNAPSHOT_LIMIT:
                image.save(output / f"frame_{sample_count:06d}_pred_{label}.jpg", quality=90)
                snapshots += 1
            next_inference = now + inference_interval
    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        trajectory.close()
        report = {
            "checkpoint": str(checkpoint_path), "world": robot.getWorldPath(),
            "preprocessing": checkpoint["preprocessing"],
            "samples": sample_count, "sim_time_s": last_time,
            "class_counts": dict(counts), "max_displacement_m": max_displacement,
            "end_reason": reason, "blank_frames": blank_frames,
            "first_frame_diagnostic": first_frame_diagnostic,
            "saved_snapshots": snapshots, "snapshot_limit": SNAPSHOT_LIMIT,
            "stop_diagnostics": stop_diagnostics,
            "stop_events": stop_policy.events, "stop_phase": stop_policy.phase,
            "stop_hold_seconds": stop_policy.hold_seconds,
            "stop_enabled": stop_enabled, "drive_speed_multiplier": speed_multiplier,
            "limitations": "No ground-truth lane center or manual driving score; trajectory and camera snapshots require visual review.",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if os.environ.get('ULT_CNN_QUIT_WEBOTS_ON_FINISH')=='1':
            robot.simulationQuit(0)


if __name__ == "__main__":
    main()
