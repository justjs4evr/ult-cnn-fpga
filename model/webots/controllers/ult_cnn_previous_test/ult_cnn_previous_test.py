"""Bounded Webots closed-loop test of the six-class ULT-CNN checkpoint."""

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from controller import Supervisor


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "ult_cnn_training"
sys.path.insert(0, str(TRAINING))
from model import ULTCNN  # noqa: E402
from preprocess import CROP_TOP_RATIO, INTERMEDIATE_SIZE, INPUT_SIZE, WHITE_THRESHOLD  # noqa: E402

CHECKPOINT = TRAINING / "output_visible_pair_20260927" / "best_model.pt"
OUTPUT = TRAINING / "webots_previous_live_test"
CLASSES = ("straight", "left", "right", "sharp_left", "sharp_right", "stop")
WHEELS = {
    "straight": (2.0, 2.0),
    "left": (0.75, 1.6),
    "right": (1.6, 0.75),
    "sharp_left": (-0.45, 1.5),
    "sharp_right": (1.5, -0.45),
    "stop": (0.0, 0.0),
}
MAX_SIM_TIME_S = 30.0
INFERENCE_INTERVAL_S = 0.16


def camera_image(camera):
    raw = camera.getImage()
    if raw is None:
        raise RuntimeError("Central camera did not return an image")
    return Image.frombytes("RGBA", (camera.getWidth(), camera.getHeight()), raw, "raw", "BGRA").convert("RGB")


def preprocess(image):
    gray = image.convert("L")
    width, height = gray.size
    roi = gray.crop((0, int(height * CROP_TOP_RATIO), width, height))
    medium = roi.resize(INTERMEDIATE_SIZE, Image.Resampling.BOX)
    binary = np.where(np.asarray(medium) >= WHITE_THRESHOLD, 255, 0).astype(np.uint8)
    small = Image.fromarray(binary).resize(INPUT_SIZE, Image.Resampling.BOX)
    return torch.from_numpy(np.asarray(small, dtype=np.float32).copy() / 255).reshape(1, 1, 16, 16)


def main():
    torch.set_num_threads(1)
    checkpoint = torch.load(io.BytesIO(CHECKPOINT.read_bytes()), map_location="cpu", weights_only=False)
    if tuple(checkpoint["classes"]) != CLASSES or checkpoint["preprocessing"]["mode"] != "binary":
        raise ValueError("Unexpected checkpoint classes or preprocessing")
    model = ULTCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    robot = Supervisor()
    output = OUTPUT / Path(robot.getWorldPath()).stem
    output.mkdir(parents=True, exist_ok=True)
    step_ms = int(robot.getBasicTimeStep())
    camera = robot.getDevice("cam_center")
    camera.enable(step_ms)
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
    node = robot.getSelf()
    rows = []
    counts = Counter()
    next_inference = 0.0
    previous = None
    start_xy = None
    max_displacement = 0.0
    stopped_steps = 0
    blank_frames = 0
    first_frame_diagnostic = None
    reason = "time_limit"
    try:
        while robot.step(step_ms) != -1:
            now = robot.getTime()
            if now >= MAX_SIM_TIME_S:
                break
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
                next_inference = now + INFERENCE_INTERVAL_S
                continue
            blank_frames = 0
            with torch.inference_mode():
                probabilities = model(preprocess(image)).softmax(dim=1)[0]
            index = int(probabilities.argmax())
            label = CLASSES[index]
            confidence = float(probabilities[index])
            speeds = WHEELS[label]
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
            if label == "stop":
                stopped_steps += 1
            else:
                stopped_steps = 0
            rows.append((round(now, 3), round(x, 4), round(y, 4), label,
                         round(confidence, 4), speeds[0], speeds[1]))
            if len(rows) % 10 == 0:
                print(f"t={now:.2f} class={label} confidence={confidence:.3f} xy=({x:.3f},{y:.3f})", flush=True)
            if len(rows) % 30 == 1:
                image.save(output / f"frame_{len(rows):04d}_{label}.jpg", quality=90)
            if stopped_steps >= 30:
                reason = "continuous_stop_prediction"
                break
            next_inference = now + INFERENCE_INTERVAL_S
    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("sim_time_s", "world_x_m", "world_y_m", "predicted_class", "confidence", "left_speed", "right_speed"))
            writer.writerows(rows)
        report = {
            "checkpoint": str(CHECKPOINT), "world": robot.getWorldPath(),
            "samples": len(rows), "sim_time_s": rows[-1][0] if rows else None,
            "class_counts": dict(counts), "max_displacement_m": max_displacement,
            "end_reason": reason, "blank_frames": blank_frames,
            "first_frame_diagnostic": first_frame_diagnostic,
            "limitations": "No ground-truth lane center or manual driving score; trajectory and camera snapshots require visual review.",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
