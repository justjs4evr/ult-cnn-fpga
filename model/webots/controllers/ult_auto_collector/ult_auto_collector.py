"""Reviewable Webots pose runs for the ULT-CNN three-camera dataset."""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from controller import Supervisor


ROOT = Path(__file__).resolve().parent
JOB_PATH = ROOT / "auto_collection_job.json"
STATUS_PATH = ROOT / "auto_collection_status.json"
DATASET = ROOT.parent / "ult_data_collector" / "dataset"
IMAGES = DATASET / "images"
CAMERAS = ("cam_center", "cam_left", "cam_right")
CLASSES = ("straight", "left", "right", "sharp_left", "sharp_right", "stop")
SCENE_CONTEXTS = (
    "corner_90", "tight_s_curve", "lane_recovery", "gentle_curve",
    "straight", "stop_line", "unknown",
)
FIELDS = (
    "episode_id", "frame_id", "sim_time_s", "label_id", "label_name",
    "steering", "left_speed", "right_speed", "center_image",
    "left_image", "right_image",
)
POSE_FIELDS = (
    "episode_id", "frame_id", "pose_id", "map_x_px", "map_y_px",
    "heading_deg", "world_x_m", "world_y_m", "world_yaw_rad", "white_pixels",
)
ARENA_X = 0.389118
ARENA_Y = 0.0919846
ARENA_YAW = 1.47262
DEFAULT_ARENA_SIZE_M = 1.5
MAP_SIZE_PX = 3000
WHEEL_RADIUS_M = 0.02


def world_pose(pose, arena_size_m):
    x = (float(pose["map_x_px"]) / MAP_SIZE_PX - 0.5) * arena_size_m
    y = (0.5 - float(pose["map_y_px"]) / MAP_SIZE_PX) * arena_size_m
    c, s = math.cos(ARENA_YAW), math.sin(ARENA_YAW)
    world_x = ARENA_X + c * x - s * y
    world_y = ARENA_Y + s * x + c * y
    yaw = ARENA_YAW + math.radians(float(pose["heading_deg"]))
    return world_x, world_y, yaw


def map_pose_from_world(world_x, world_y, arena_size_m):
    dx, dy = world_x - ARENA_X, world_y - ARENA_Y
    c, s = math.cos(ARENA_YAW), math.sin(ARENA_YAW)
    x = c * dx + s * dy
    y = -s * dx + c * dy
    return ((x / arena_size_m + 0.5) * MAP_SIZE_PX,
            (0.5 - y / arena_size_m) * MAP_SIZE_PX)


def next_frame_id():
    ids = []
    for path in IMAGES.glob("cam_center_*.jpg"):
        try:
            ids.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return max(ids, default=0) + 1


def white_pixel_count(camera):
    """Count bright road marks in the lower center camera, excluding the sky."""
    raw = camera.getImage()
    if raw is None:
        return 0
    width, height = camera.getWidth(), camera.getHeight()
    count = 0
    for y in range(height // 2, height * 9 // 10):
        for x in range(width):
            offset = (y * width + x) * 4
            if min(raw[offset], raw[offset + 1], raw[offset + 2]) > 85:
                count += 1
    return count


def save_triplet(cameras, paths):
    if any(cam.getImage() is None for cam in cameras.values()):
        return False
    try:
        for name, camera in cameras.items():
            if camera.saveImage(str(paths[name]), 95) != 0:
                raise OSError(f"Could not save {name}")
    except OSError:
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise
    return True


def main():
    robot = Supervisor()
    step_ms = int(robot.getBasicTimeStep())
    cameras = {name: robot.getDevice(name) for name in CAMERAS}
    for cam in cameras.values():
        cam.enable(step_ms)
    motors = (robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor"))
    sensors = (robot.getDevice("left wheel sensor"), robot.getDevice("right wheel sensor"))
    for sensor in sensors:
        sensor.enable(step_ms)
    for motor in motors:
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    if not JOB_PATH.exists():
        print("No auto_collection_job.json; automatic collector idle.")
        return
    job = json.loads(JOB_PATH.read_text(encoding="utf-8-sig"))
    mode = job["mode"]
    if mode not in ("preview", "collect", "trajectory_preview"):
        raise ValueError("mode must be preview, collect, or trajectory_preview")
    poses = job["poses"]
    arena_size_m = float(job.get("arena_size_m", DEFAULT_ARENA_SIZE_M))
    if not 0.5 <= arena_size_m <= 10.0:
        raise ValueError("arena_size_m must be between 0.5 and 10")
    if not poses or len(poses) > 100:
        raise ValueError("poses must contain 1 to 100 entries")
    for pose in poses:
        if mode == "trajectory_preview":
            segments = pose.get("segments", [])
            if not segments or len(segments) > 20:
                raise ValueError("trajectory pose needs 1 to 20 segments")
            if sum(float(segment["duration_s"]) for segment in segments) > 20:
                raise ValueError("trajectory duration exceeds 20 seconds")
            for segment in segments:
                if not 0 < float(segment["duration_s"]) <= 5:
                    raise ValueError("segment duration must be 0 to 5 seconds")
                if max(abs(float(segment["left_speed"])), abs(float(segment["right_speed"]))) > min(m.getMaxVelocity() for m in motors):
                    raise ValueError("segment wheel speed exceeds motor limit")
        duration = float(pose.get("duration_s", 0.0))
        if not 0.0 <= duration <= 5.0:
            raise ValueError("duration_s must be between 0 and 5")
        if max(abs(float(pose.get("left_speed", 0))), abs(float(pose.get("right_speed", 0)))) > min(m.getMaxVelocity() for m in motors):
            raise ValueError("wheel speed exceeds motor limit")
        if mode == "collect" and pose["label"] not in CLASSES:
            raise ValueError("unknown class label")
        if pose.get("scene_context", "unknown") not in SCENE_CONTEXTS:
            raise ValueError("unknown scene_context")
    active_job = ROOT / "auto_collection_job_active.json"
    JOB_PATH.replace(active_job)
    episode = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    status = {"episode_id": episode, "mode": mode, "poses": len(poses), "saved": 0,
              "skipped_blank": 0, "preview_files": [], "trajectory_samples": []}
    node = robot.getSelf()
    translation = node.getField("translation")
    rotation = node.getField("rotation")
    preview_dir = ROOT / "preview" / episode
    if mode in ("preview", "trajectory_preview"):
        preview_dir.mkdir(parents=True, exist_ok=True)
    else:
        IMAGES.mkdir(parents=True, exist_ok=True)
    frame_id = next_frame_id()

    labels_path = DATASET / "labels.csv"
    pose_path = DATASET / "auto_pose_manifest.csv"
    labels_new = not labels_path.exists() or labels_path.stat().st_size == 0
    poses_new = not pose_path.exists() or pose_path.stat().st_size == 0
    try:
        with labels_path.open("a", newline="", encoding="utf-8") as labels_stream, \
             pose_path.open("a", newline="", encoding="utf-8") as pose_stream:
            labels_writer = csv.DictWriter(labels_stream, fieldnames=FIELDS)
            pose_writer = csv.DictWriter(pose_stream, fieldnames=POSE_FIELDS)
            if mode == "collect":
                if labels_new:
                    labels_writer.writeheader()
                if poses_new:
                    pose_writer.writeheader()
            for index, pose in enumerate(poses):
                pose_id = str(pose.get("id", f"pose_{index:02d}"))
                x, y, yaw = world_pose(pose, arena_size_m)
                for motor in motors:
                    motor.setVelocity(0.0)
                translation.setSFVec3f([x, y, 0.0])
                rotation.setSFRotation([0.0, 0.0, 1.0, yaw])
                node.resetPhysics()
                for _ in range(8):
                    if robot.step(step_ms) == -1:
                        raise RuntimeError("Webots simulation stopped during pose setup")
                print(f"Pose {pose_id}: map=({pose['map_x_px']},{pose['map_y_px']}), "
                      f"heading={pose['heading_deg']}, world=({x:.3f},{y:.3f})")
                if mode == "trajectory_preview":
                    sample_index = 0
                    interval = float(job.get("capture_interval_s", 0.25))
                    if not 0.1 <= interval <= 2.0:
                        raise ValueError("capture_interval_s must be 0.1 to 2 seconds")

                    def record_trajectory_sample(phase):
                        nonlocal sample_index
                        point = translation.getSFVec3f()
                        map_x, map_y = map_pose_from_world(point[0], point[1], arena_size_m)
                        stem = f"{pose_id}_{sample_index:03d}_{phase}"
                        paths = {name: preview_dir / f"{stem}_{name}.jpg" for name in CAMERAS}
                        if save_triplet(cameras, paths):
                            status["preview_files"].extend(str(path) for path in paths.values())
                            status["trajectory_samples"].append({
                                "pose_id": pose_id, "sample": sample_index, "phase": phase,
                                "sim_time_s": round(robot.getTime(), 3),
                                "map_x_px": round(map_x, 1), "map_y_px": round(map_y, 1),
                                "world_yaw_rad": round(rotation.getSFRotation()[3], 3),
                                "white_pixels": white_pixel_count(cameras["cam_center"]),
                                "center_image": str(paths["cam_center"]),
                            })
                        sample_index += 1

                    record_trajectory_sample("start")
                    for segment in pose["segments"]:
                        phase = str(segment.get("phase", "drive"))
                        motors[0].setVelocity(float(segment["left_speed"]))
                        motors[1].setVelocity(float(segment["right_speed"]))
                        segment_start = robot.getTime()
                        next_capture = segment_start + interval
                        while robot.getTime() - segment_start < float(segment["duration_s"]):
                            if robot.step(step_ms) == -1:
                                raise RuntimeError("Webots simulation stopped during trajectory")
                            if robot.getTime() >= next_capture:
                                record_trajectory_sample(phase)
                                next_capture += interval
                    motors[0].setVelocity(0.0)
                    motors[1].setVelocity(0.0)
                    continue
                if mode == "preview":
                    paths = {name: preview_dir / f"{pose_id}_{name}_start.jpg" for name in CAMERAS}
                    if save_triplet(cameras, paths):
                        status["preview_files"].extend(str(path) for path in paths.values())
                left_speed = float(pose.get("left_speed", 0.0))
                right_speed = float(pose.get("right_speed", 0.0))
                motors[0].setVelocity(left_speed)
                motors[1].setVelocity(right_speed)
                start_time = robot.getTime()
                last_save_time = start_time
                last_wheels = [sensor.getValue() for sensor in sensors]
                last_saved_wheels = list(last_wheels)
                while robot.getTime() - start_time < float(pose.get("duration_s", 0.0)):
                    if robot.step(step_ms) == -1:
                        raise RuntimeError("Webots simulation stopped during drive")
                    if mode != "collect":
                        continue
                    now = robot.getTime()
                    wheels = [sensor.getValue() for sensor in sensors]
                    wheel_motion = max(abs(wheels[i] - last_saved_wheels[i]) for i in (0, 1)) * WHEEL_RADIUS_M
                    if now - last_save_time < 0.4 or wheel_motion < 0.015:
                        continue
                    white_pixels = white_pixel_count(cameras["cam_center"])
                    last_save_time = now
                    last_saved_wheels = wheels
                    if white_pixels < 30:
                        status["skipped_blank"] += 1
                        continue
                    paths = {name: IMAGES / f"{name}_{frame_id:07d}.jpg" for name in CAMERAS}
                    if not save_triplet(cameras, paths):
                        continue
                    label = pose["label"]
                    labels_writer.writerow({
                        "episode_id": episode, "frame_id": frame_id,
                        "sim_time_s": f"{now:.3f}", "label_id": CLASSES.index(label),
                        "label_name": label,
                        "steering": f"{(left_speed - right_speed) / max(abs(left_speed) + abs(right_speed), 1e-6):.3f}",
                        "left_speed": f"{left_speed:.3f}", "right_speed": f"{right_speed:.3f}",
                        "center_image": paths["cam_center"].relative_to(DATASET).as_posix(),
                        "left_image": paths["cam_left"].relative_to(DATASET).as_posix(),
                        "right_image": paths["cam_right"].relative_to(DATASET).as_posix(),
                    })
                    pose_writer.writerow({
                        "episode_id": episode, "frame_id": frame_id, "pose_id": pose_id,
                        "map_x_px": pose["map_x_px"], "map_y_px": pose["map_y_px"],
                        "heading_deg": pose["heading_deg"], "world_x_m": f"{x:.6f}",
                        "world_y_m": f"{y:.6f}", "world_yaw_rad": f"{yaw:.6f}",
                        "white_pixels": white_pixels,
                    })
                    labels_stream.flush()
                    pose_stream.flush()
                    status["saved"] += 1
                    frame_id += 1
                    last_wheels = wheels
                motors[0].setVelocity(0.0)
                motors[1].setVelocity(0.0)
                if mode == "preview":
                    for _ in range(3):
                        if robot.step(step_ms) == -1:
                            raise RuntimeError("Webots simulation stopped after drive")
                    paths = {name: preview_dir / f"{pose_id}_{name}_end.jpg" for name in CAMERAS}
                    if save_triplet(cameras, paths):
                        status["preview_files"].extend(str(path) for path in paths.values())
    finally:
        for motor in motors:
            motor.setVelocity(0.0)
        STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
        (ROOT / f"auto_collection_status_{episode}.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8")
        active_job.replace(ROOT / f"auto_collection_job_{episode}.json")
    if mode == "collect":
        (DATASET / f"episode_{episode}.json").write_text(json.dumps({
            "episode_id": episode,
            "world": job.get("world", "ult_auto_collection.wbt"),
            "environment": job.get("environment", "baseline"),
            "arena_size_m": arena_size_m,
            "classes": list(CLASSES), "cameras": {
                name: {"width": cam.getWidth(), "height": cam.getHeight()}
                for name, cam in cameras.items()},
            "label_source": "reviewed_map_pose", "source_job": f"auto_collection_job_{episode}.json",
            "pose_contexts": {
                str(pose.get("id", f"pose_{index:02d}")): pose.get("scene_context", "unknown")
                for index, pose in enumerate(poses)
            },
            "min_white_pixels": 30,
        }, indent=2), encoding="utf-8")
    print(f"Automatic {mode} complete: {status['saved']} saved, "
          f"{status['skipped_blank']} blank frames skipped")


if __name__ == "__main__":
    main()
