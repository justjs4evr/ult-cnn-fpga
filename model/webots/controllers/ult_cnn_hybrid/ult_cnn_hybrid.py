"""Webots lane-center steering with the existing six-class CNN as a weak aid."""

import csv
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from controller import Supervisor

from lane_tracking import detect_lane


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "ult_cnn_training"
sys.path.insert(0, str(TRAINING))
from model import ULTCNN  # noqa: E402
from preprocess import CROP_TOP_RATIO, INTERMEDIATE_SIZE, INPUT_SIZE, WHITE_THRESHOLD  # noqa: E402


CHECKPOINT = TRAINING / "output_current_six_binary" / "best_model.pt"
OUTPUT = TRAINING / "webots_hybrid_live_test"
CLASSES = ("straight", "left", "right", "sharp_left", "sharp_right", "stop")
# Interactive driving has no automatic timeout. Set this only for bounded tests.
MAX_SIM_TIME_S = float(os.environ.get("ULT_CNN_HYBRID_TEST_SECONDS", "0"))
CONTROL_INTERVAL_S = 0.16


def camera_image(camera):
    raw = camera.getImage()
    if raw is None:
        raise RuntimeError("Center camera returned no image")
    return Image.frombytes("RGBA", (camera.getWidth(), camera.getHeight()), raw,
                           "raw", "BGRA").convert("RGB")


def cnn_input(image):
    gray = image.convert("L")
    width, height = gray.size
    roi = gray.crop((0, int(height * CROP_TOP_RATIO), width, height))
    medium = roi.resize(INTERMEDIATE_SIZE, Image.Resampling.BOX)
    binary = np.where(np.asarray(medium) >= WHITE_THRESHOLD, 255, 0).astype(np.uint8)
    small = Image.fromarray(binary).resize(INPUT_SIZE, Image.Resampling.BOX)
    return torch.from_numpy(np.asarray(small, dtype=np.float32).copy() / 255).reshape(1, 1, 16, 16)


def steering_for(lane, label, confidence, previous_error, lost_frames=0):
    """Real line geometry dominates; CNN bias is bounded and corroborated."""
    error = np.clip((lane.center - 79.5) / 79.5, -1.25, 1.25)
    error = float(error)
    steering = 0.85 * error + 0.32 * lane.heading + 0.18 * (error - previous_error)
    hint = {"left": -1, "sharp_left": -1, "right": 1, "sharp_right": 1}.get(label, 0)
    # Direction agreement avoids the known left/right CNN errors overriding two lines.
    if confidence >= 0.65 and hint and (steering == 0 or np.sign(steering) == hint):
        steering += hint * (0.08 if lane.pair_rows >= 2 else 0.17)
    steering = float(np.clip(steering, -1.1, 1.1))
    if lane.pair_rows >= 2:
        base = 1.8
        mode = "two_lines"
    elif lane.single_rows:
        base = 1.1
        mode = "one_line"
    else:
        base = 0.0
        mode = "lost"
    # Far-side pairs alone do not establish a drivable center at a square corner.
    # Require the near boundary shape and the CNN sharp-turn direction to agree.
    corner_confirmed = (lane.pair_rows >= 1 and lane.near_pair_rows == 0
                        and lane.single_rows >= 3 and label in ("sharp_left", "sharp_right")
                        and confidence >= 0.65 and abs(lane.heading) >= 0.15
                        and np.sign(lane.heading) == hint)
    if corner_confirmed:
        steering = 0.70 * hint
        base = 1.05
        mode = "confirmed_corner"
    base *= 1.0 - 0.28 * min(1.0, abs(steering))
    correction = 1.25 * steering
    left = float(np.clip(base + correction, -0.35, 2.8))
    right = float(np.clip(base - correction, -0.35, 2.8))
    if mode == "lost":
        left = right = 0.0
        # Short, bounded recovery for a square corner: the lines can be horizontal
        # and absent from the row-wise lane detector until the camera turns.
        if lost_frames <= 25 and confidence >= 0.60 and hint:
            left, right = (1.0, -0.2) if hint > 0 else (-0.2, 1.0)
            mode = "cnn_recovery"
    # A CNN stop alone is insufficient: it produced false stops on plain lanes.
    if label == "stop" and confidence >= 0.70 and lane.crosswalk_bands >= 5:
        left = right = 0.0
        mode = "crosswalk_stop"
    return left, right, error, steering, mode


def main():
    torch.set_num_threads(1)
    saved = torch.load(io.BytesIO(CHECKPOINT.read_bytes()), map_location="cpu", weights_only=False)
    if tuple(saved["classes"]) != CLASSES or saved["preprocessing"]["mode"] != "binary":
        raise ValueError("Unexpected checkpoint classes or preprocessing")
    model = ULTCNN()
    model.load_state_dict(saved["model_state"])
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
    next_control = 0.0
    previous_center = None
    previous_side = None
    previous_error = 0.0
    start_xy = None
    max_displacement = 0.0
    blank_frames = 0
    lost_frames = 0
    reason = "simulation_stopped"
    try:
        while robot.step(step_ms) != -1:
            now = robot.getTime()
            if MAX_SIM_TIME_S > 0 and now >= MAX_SIM_TIME_S:
                reason = "time_limit"
                break
            if now < next_control:
                continue
            image = camera_image(camera)
            gray = np.asarray(image.convert("L"))
            if gray.max() == 0:
                blank_frames += 1
                if blank_frames >= 5:
                    reason = "blank_camera"
                    break
                next_control = now + CONTROL_INTERVAL_S
                continue
            blank_frames = 0
            with torch.inference_mode():
                probs = model(cnn_input(image)).softmax(dim=1)[0]
            index = int(probs.argmax())
            label = CLASSES[index]
            confidence = float(probs[index])
            lane = detect_lane(gray, previous_center, previous_side, label)
            lost_frames = lost_frames + 1 if lane.confidence == 0 else 0
            left, right, error, steering, mode = steering_for(
                lane, label, confidence, previous_error, lost_frames)
            left_motor.setVelocity(left)
            right_motor.setVelocity(right)
            if lane.confidence >= 0.3:
                previous_center = 0.65 * (previous_center if previous_center is not None else 79.5) + 0.35 * lane.center
                previous_side = lane.inferred_side
            previous_error = error
            position = node.getPosition()
            x, y = position[0], position[1]
            if start_xy is None:
                start_xy = (x, y)
            max_displacement = max(max_displacement,
                                   ((x - start_xy[0]) ** 2 + (y - start_xy[1]) ** 2) ** 0.5)
            counts[mode] += 1
            counts[f"cnn_{label}"] += 1
            rows.append((round(now, 3), round(x, 4), round(y, 4), mode,
                         lane.pair_rows, lane.near_pair_rows, lane.single_rows, round(lane.center, 2),
                         round(lane.heading, 3), round(lane.confidence, 3), lane.threshold,
                         label, round(confidence, 3), round(steering, 3), left, right))
            if len(rows) % 20 == 1:
                image.save(output / f"frame_{len(rows):04d}_{mode}_{label}.jpg", quality=90)
            if len(rows) % 20 == 0:
                print(f"t={now:.1f} mode={mode} pair={lane.pair_rows}/{lane.near_pair_rows} "
                      f"center={lane.center:.1f} heading={lane.heading:.2f} "
                      f"cnn={label}:{confidence:.2f} steer={steering:.2f} "
                      f"wheels=({left:.2f},{right:.2f}) xy=({x:.3f},{y:.3f})", flush=True)
            next_control = now + CONTROL_INTERVAL_S
    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("sim_time_s", "world_x_m", "world_y_m", "mode", "pair_rows",
                             "near_pair_rows", "single_rows", "lane_center_px", "heading", "lane_confidence",
                             "white_threshold", "cnn_class", "cnn_confidence", "steering",
                             "left_speed", "right_speed"))
            writer.writerows(rows)
        report = {"checkpoint": str(CHECKPOINT), "world": robot.getWorldPath(),
                  "samples": len(rows), "sim_time_s": rows[-1][0] if rows else None,
                  "counts": dict(counts), "max_displacement_m": max_displacement,
                  "end_reason": reason, "blank_frames": blank_frames,
                  "limitations": "No ground-truth lane polygon or route completion score; camera frames require review."}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
