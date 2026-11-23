"""Inventory known map contexts without guessing recovery scenes in manual runs."""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "controllers" / "ult_data_collector" / "dataset"
TURN_CLASSES = {"left", "right", "sharp_left", "sharp_right"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def scene_context(world, label, metadata, pose_id):
    if metadata.get("label_source") == "manual_operator":
        return "unreviewed_manual", "manual video has no pose or recovery marker"
    if "pose_contexts" in metadata:
        return metadata["pose_contexts"].get(pose_id, "unknown"), "pose annotation"
    if world.startswith(("ult_auto_iro_train_16", "ult_auto_iro_test_17")):
        return "s_course_corner_90", "known S layout made from 90-degree corner tiles"
    if world.startswith("ult_auto_iro_train_15"):
        return ("corner_90" if label.startswith("sharp_") else "gentle_curve"), "known map tile"
    if world.startswith("ult_auto_iro_lane_"):
        if label.startswith("sharp_"):
            return "corner_90", "official corner map and reviewed pose label"
        return "other_official_tile", "pose-level review still needed"
    return "unreviewed_map_pose", "world name alone cannot identify turn reason"


def main():
    rows = [row for row in read_csv(DATASET / "training_manifest.csv")
            if row["label_name"] in TURN_CLASSES]
    roles = {row["frame_id"]: row["split"] for row in read_csv(ROOT / "split_manifest.csv")}
    pose_ids = {(row["episode_id"], row["frame_id"]): row["pose_id"]
                for row in read_csv(DATASET / "auto_pose_manifest.csv")}
    episodes = {}
    for episode in {row["episode_id"] for row in rows}:
        episodes[episode] = json.loads(
            (DATASET / f"episode_{episode}.json").read_text(encoding="utf-8"))
    result = []
    for row in rows:
        metadata = episodes[row["episode_id"]]
        pose_id = pose_ids.get((row["episode_id"], row["frame_id"]), "")
        context, basis = scene_context(metadata["world"], row["label_name"], metadata, pose_id)
        result.append({"frame_id": row["frame_id"], "episode_id": row["episode_id"],
                       "split": roles[row["frame_id"]], "label_name": row["label_name"],
                       "world": metadata["world"], "scene_context": context,
                       "context_basis": basis})
    path = ROOT / "turn_context_inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    print("turn frames", len(result))
    for (split, context), count in sorted(Counter(
            (row["split"], row["scene_context"]) for row in result).items()):
        print(split, context, count)
    print(path)


if __name__ == "__main__":
    main()
