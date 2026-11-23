"""Evaluate a saved ULT-CNN on separately recorded Webots track groups."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from model import ULTCNN
from train import CLASSES, DEFAULT_DATASET, load_arrays, load_rows, score_model


def track_group(world):
    if world.startswith("ult_auto_case1_3m"):
        return "case1_correct_3m"
    if world.startswith("ult_auto_iro_lane_"):
        return "iro_official_tiles"
    if world.startswith("ult_auto_track_"):
        return "autonomous_tracks_v2"
    if world in ("ult_data_collection.wbt", "ult_auto_collection.wbt",
                 "ult_auto_collection_bright.wbt"):
        return "case1_old_1_5m"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "output" / "best_model.pt")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "domain_evaluation.json")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    rows = load_rows(dataset / "training_manifest.csv")
    metadata = {}
    for path in dataset.glob("episode_*.json"):
        episode = json.loads(path.read_text(encoding="utf-8"))
        metadata[episode["episode_id"]] = episode

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint classes do not match the manifest")
    mode = checkpoint["preprocessing"]["mode"]
    model = ULTCNN()
    model.load_state_dict(checkpoint["model_state"])
    training_episodes = set(checkpoint["training_episodes"])
    validation_episode = checkpoint["validation_episode"]
    groups = defaultdict(list)
    for row in rows:
        episode_id = row["episode_id"]
        world = metadata.get(episode_id, {}).get("world", "")
        groups[track_group(world)].append(row)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_validation_episode": validation_episode,
        "preprocess_mode": mode,
        "groups": {},
    }
    for group_name, group_rows in sorted(groups.items()):
        images, labels = load_arrays(group_rows, dataset, mode)
        result = score_model(model, images, labels)
        result.pop("predictions")
        signatures = [hashlib.sha256(image.numpy().tobytes()).digest()
                      for image in images]
        report["groups"][group_name] = {
            "frames": len(group_rows),
            "episodes": sorted({row["episode_id"] for row in group_rows}),
            "checkpoint_training_frames": sum(
                row["episode_id"] in training_episodes for row in group_rows
            ),
            "checkpoint_validation_frames": sum(
                row["episode_id"] == validation_episode for row in group_rows
            ),
            "unique_16x16_inputs": len(set(signatures)),
            "all_black_16x16_inputs": int((images.sum(dim=(1, 2, 3)) == 0).sum()),
            "accuracy": result["accuracy"],
            "macro_f1_present_classes": result["macro_f1"],
            "support": result["support"],
            "per_class_recall": result["per_class_recall"],
            "confusion_matrix": result["confusion_matrix"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in group.items()
                             if key in ("frames", "checkpoint_training_frames",
                                        "unique_16x16_inputs", "all_black_16x16_inputs",
                                        "accuracy", "macro_f1_present_classes", "support")}
                      for name, group in report["groups"].items()}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
