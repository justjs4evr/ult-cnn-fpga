"""Manual, class-labelled three-camera capture for the ULT-CNN pilot dataset."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from controller import Keyboard, Robot


CLASSES = (
    "straight", "left", "right", "sharp_left", "sharp_right", "stop"
)
CAMERA_NAMES = ("cam_center", "cam_left", "cam_right")
DRIVE_COMMAND_PATH = Path(__file__).resolve().parent / "drive_once.json"
DRIVE_RESULT_PATH = Path(__file__).resolve().parent / "drive_once_result.json"
DRIVE_PREVIEW_DIR = Path(__file__).resolve().parent / "drive_preview"
LABEL_KEYS = {
    ord("8"): 0,  # straight
    ord("7"): 1,  # left
    ord("9"): 2,  # right
    ord("4"): 3,  # sharp left
    ord("6"): 4,  # sharp right
    ord("5"): 5,  # stop line
}
MIN_SAVE_PERIOD_S = 0.20
MIN_TRAVEL_M = 0.02
MIN_TURN_RAD = 0.15
WHEEL_RADIUS_M = 0.02
AXLE_LENGTH_M = 0.052
BASE_SPEED = 1.2  # e-puck wheel speed, rad/s
TURN_GAIN = 0.65


def next_frame_id(images_dir):
    ids = []
    for path in images_dir.glob("*_*.jpg"):
        try:
            ids.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return max(ids, default=0) + 1


def main():
    robot = Robot()
    step_ms = int(robot.getBasicTimeStep())
    keyboard = robot.getKeyboard()
    keyboard.enable(step_ms)

    cameras = {name: robot.getDevice(name) for name in CAMERA_NAMES}
    for camera in cameras.values():
        camera.enable(step_ms)

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_sensor = robot.getDevice("left wheel sensor")
    right_sensor = robot.getDevice("right wheel sensor")
    left_sensor.enable(step_ms)
    right_sensor.enable(step_ms)
    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
    max_speed = min(left_motor.getMaxVelocity(), right_motor.getMaxVelocity())

    # A one-shot command lets an external operator verify real motor control
    # without changing the manual labels or recording unreviewed images.
    drive_command = None
    if DRIVE_COMMAND_PATH.exists():
        command = json.loads(DRIVE_COMMAND_PATH.read_text(encoding="utf-8"))
        duration = float(command["duration_s"])
        left = float(command["left_speed"])
        right = float(command["right_speed"])
        if not 0.0 < duration <= 5.0:
            raise ValueError("drive_once duration_s must be in (0, 5]")
        if max(abs(left), abs(right)) > max_speed:
            raise ValueError("drive_once wheel speed exceeds motor limit")
        drive_command = {"duration_s": duration, "left_speed": left, "right_speed": right}
        DRIVE_COMMAND_PATH.unlink()
        print(f"One-shot drive: left={left:.2f}, right={right:.2f}, duration={duration:.2f}s")

    output = Path(__file__).resolve().parent / "dataset"
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    labels_path = output / "labels.csv"
    needs_header = not labels_path.exists() or labels_path.stat().st_size == 0
    frame_id = next_frame_id(images)
    episode_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "episode_id": episode_id,
        "world": "ult_data_collection.wbt",
        "classes": list(CLASSES),
        "cameras": {
            name: {"width": cam.getWidth(), "height": cam.getHeight()}
            for name, cam in cameras.items()
        },
        "min_save_period_s": MIN_SAVE_PERIOD_S,
        "min_travel_m": MIN_TRAVEL_M,
        "min_turn_rad": MIN_TURN_RAD,
        "label_source": "manual_operator",
        "label_keys": {chr(key): CLASSES[value] for key, value in LABEL_KEYS.items()},
        "steering_sign": "negative_left_positive_right",
    }
    (output / f"episode_{episode_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    label = None
    recording = False
    previous_keys = set()
    last_save_time = float("-inf")
    last_saved_label = None
    distance_since_save = 0.0
    turn_since_save = 0.0
    previous_wheels = None
    drive_start_wheels = None
    drive_end_time = None
    fields = [
        "episode_id", "frame_id", "sim_time_s", "label_id", "label_name",
        "steering", "left_speed", "right_speed", "center_image",
        "left_image", "right_image",
    ]
    print("Focus Webots 3D view. Num Lock ON for keypad class keys.")
    print("Arrows: drive; Space: brake; keypad 8/7/4/9/6/5: class.")
    print("R: start/resume recording; S: pause recording; Q: quit.")
    print("Saves after 2 cm travel or 9 deg turn; no repeated idle frames.")
    print("Select a class before recording.")

    try:
        with labels_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if needs_header:
                writer.writeheader()
                stream.flush()

            while robot.step(step_ms) != -1:
                wheels = (left_sensor.getValue(), right_sensor.getValue())
                if previous_wheels is not None:
                    delta_left = wheels[0] - previous_wheels[0]
                    delta_right = wheels[1] - previous_wheels[1]
                    distance_since_save += abs(
                        (delta_left + delta_right) * WHEEL_RADIUS_M / 2.0
                    )
                    turn_since_save += abs(
                        (delta_right - delta_left) * WHEEL_RADIUS_M / AXLE_LENGTH_M
                    )
                previous_wheels = wheels

                keys = set()
                while True:
                    key = keyboard.getKey()
                    if key == -1:
                        break
                    keys.add(key & Keyboard.KEY)

                pressed = keys - previous_keys
                previous_keys = keys
                for key_code, class_id in LABEL_KEYS.items():
                    if key_code in pressed:
                        label = class_id
                        print(f"Class {label}: {CLASSES[label]}")
                if ord("R") in pressed:
                    if label is None:
                        print("Choose a class with the numeric keypad first.")
                    else:
                        if not recording:
                            recording = True
                            last_saved_label = None
                            distance_since_save = 0.0
                            turn_since_save = 0.0
                            print("Recording ON")
                if ord("S") in pressed:
                    if recording:
                        recording = False
                        print("Recording PAUSED")
                if ord("Q") in pressed:
                    break

                # Braking takes priority. Releasing keys stops the robot.
                throttle = 0.0 if ord(" ") in keys else (
                    float(Keyboard.UP in keys) - float(Keyboard.DOWN in keys)
                )
                steering = float(Keyboard.RIGHT in keys) - float(Keyboard.LEFT in keys)
                left_speed = max(-max_speed, min(max_speed,
                    throttle * BASE_SPEED * (1.0 + TURN_GAIN * steering)))
                right_speed = max(-max_speed, min(max_speed,
                    throttle * BASE_SPEED * (1.0 - TURN_GAIN * steering)))
                now = robot.getTime()
                if drive_command is not None:
                    if drive_end_time is None:
                        drive_start_wheels = wheels
                        drive_end_time = now + drive_command["duration_s"]
                        DRIVE_PREVIEW_DIR.mkdir(exist_ok=True)
                        for name, camera in cameras.items():
                            camera.saveImage(str(DRIVE_PREVIEW_DIR / f"{name}_start.jpg"), 95)
                    if now < drive_end_time:
                        left_speed = drive_command["left_speed"]
                        right_speed = drive_command["right_speed"]
                    else:
                        for name, camera in cameras.items():
                            camera.saveImage(str(DRIVE_PREVIEW_DIR / f"{name}_end.jpg"), 95)
                        delta_left = wheels[0] - drive_start_wheels[0]
                        delta_right = wheels[1] - drive_start_wheels[1]
                        DRIVE_RESULT_PATH.write_text(json.dumps({
                            "duration_s": drive_command["duration_s"],
                            "left_wheel_rad": delta_left,
                            "right_wheel_rad": delta_right,
                            "mean_wheel_travel_m":
                                (delta_left + delta_right) * WHEEL_RADIUS_M / 2.0,
                            "sim_time_s": now,
                        }, indent=2), encoding="utf-8")
                        print("One-shot drive complete; motors stopped")
                        drive_command = None
                        left_speed = 0.0
                        right_speed = 0.0
                left_motor.setVelocity(left_speed)
                right_motor.setVelocity(right_speed)

                if not recording:
                    continue
                new_class = label != last_saved_label
                enough_motion = (
                    distance_since_save >= MIN_TRAVEL_M
                    or turn_since_save >= MIN_TURN_RAD
                )
                if not new_class and (
                    now - last_save_time < MIN_SAVE_PERIOD_S or not enough_motion
                ):
                    continue

                paths = {
                    name: images / f"{name}_{frame_id:07d}.jpg"
                    for name in CAMERA_NAMES
                }
                if any(cam.getImage() is None for cam in cameras.values()):
                    continue
                try:
                    for name, camera in cameras.items():
                        if camera.saveImage(str(paths[name]), 95) != 0:
                            raise OSError(f"Could not save {name} image")
                except OSError as error:
                    for path in paths.values():
                        path.unlink(missing_ok=True)
                    print(error)
                    recording = False
                    continue

                writer.writerow({
                    "episode_id": episode_id,
                    "frame_id": frame_id,
                    "sim_time_s": f"{now:.3f}",
                    "label_id": label,
                    "label_name": CLASSES[label],
                    "steering": f"{steering:.3f}",
                    "left_speed": f"{left_speed:.3f}",
                    "right_speed": f"{right_speed:.3f}",
                    "center_image": paths["cam_center"].relative_to(output).as_posix(),
                    "left_image": paths["cam_left"].relative_to(output).as_posix(),
                    "right_image": paths["cam_right"].relative_to(output).as_posix(),
                })
                stream.flush()
                last_save_time = now
                last_saved_label = label
                distance_since_save = 0.0
                turn_since_save = 0.0
                if frame_id % 100 == 0:
                    print(f"Saved frame {frame_id}, class {CLASSES[label]}")
                frame_id += 1
    finally:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)


if __name__ == "__main__":
    main()
