"""Assign curated frames to development and external map-family holdouts."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "controllers" / "ult_data_collector" / "dataset"
DEFAULT_DEV_EPISODE = "20260925T073817Z"
CLASSES = ("straight", "left", "right", "sharp_left", "sharp_right", "stop")


def world_family(world):
    if world in {"ult_data_collection.wbt", "ult_auto_collection.wbt",
                 "ult_auto_collection_bright.wbt"}:
        return "case1_old_1_5m"
    if world in {"ult_auto_case1_3m.wbt", "ult_auto_case1_3m_bright.wbt"}:
        return "case1_correct_3m"
    if world in {"ult_auto_iro_train_15.wbt", "ult_auto_iro_train_15_bright.wbt"}:
        return "iro_training_layout_15"
    if world in {"ult_auto_iro_train_16.wbt", "ult_auto_iro_train_16_bright.wbt"}:
        return "iro_training_sharp_s_16"
    if world in {"ult_auto_iro_test_17.wbt", "ult_auto_iro_test_17_bright.wbt"}:
        return "iro_mirrored_s_17"
    if world == "ult_auto_iro_test_18.wbt":
        return "iro_context_mixed_holdout_18"
    if world.startswith("ult_auto_iro_lane_") and world.endswith(".wbt"):
        return "iro_official_tiles"
    if world.startswith("ult_auto_track_") and world.endswith(".wbt"):
        return "autonomous_tracks_v2"
    raise ValueError(f"Unrecognized world: {world}. Assign its family before splitting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dev-episode", default=DEFAULT_DEV_EPISODE)
    args = parser.parse_args()
    dataset_dir = args.dataset.resolve()
    source = dataset_dir / "training_manifest.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Training manifest is empty")

    metadata = {}
    for episode in {row["episode_id"] for row in rows}:
        path = dataset_dir / f"episode_{episode}.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        if info["episode_id"] != episode:
            raise ValueError(f"Episode metadata mismatch: {episode}")
        metadata[episode] = info

    result = []
    seen_frames = set()
    for row in rows:
        frame = int(row["frame_id"])
        if frame in seen_frames:
            raise ValueError(f"Duplicate frame_id: {frame}")
        seen_frames.add(frame)
        if row["label_name"] not in CLASSES:
            raise ValueError(f"Unknown label on frame {frame}")
        episode = row["episode_id"]
        world = metadata[episode]["world"]
        family = world_family(world)
        if family == "case1_old_1_5m":
            split = "dev" if episode == args.dev_episode else "train"
        elif family in {"iro_training_layout_15", "iro_training_sharp_s_16"}:
            split = "train"
        else:
            split = f"test_{family}"
        result.append({"frame_id": frame, "episode_id": episode,
                       "label_name": row["label_name"], "world": world,
                       "track_family": family, "split": split})

    if not any(row["split"] == "dev" for row in result):
        raise ValueError(f"Development episode missing: {args.dev_episode}")
    roles = defaultdict(set)
    for row in result:
        roles[row["track_family"]].add(row["split"])
    if roles["case1_old_1_5m"] != {"train", "dev"}:
        raise ValueError("Old case1 must supply both training and development")
    for family, splits in roles.items():
        if family != "case1_old_1_5m" and len(splits) != 1:
            raise ValueError(f"Map family leaks across splits: {family}")
    train_classes = {row["label_name"] for row in result if row["split"] == "train"}
    if train_classes != set(CLASSES):
        raise ValueError(f"Training lacks classes: {set(CLASSES) - train_classes}")

    output = Path(__file__).resolve().parent / "split_manifest.csv"
    fields = ("frame_id", "episode_id", "label_name", "world", "track_family", "split")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)
    by_split = defaultdict(Counter)
    episodes_by_split = defaultdict(set)
    worlds_by_split = defaultdict(set)
    for row in result:
        by_split[row["split"]][row["label_name"]] += 1
        episodes_by_split[row["split"]].add(row["episode_id"])
        worlds_by_split[row["split"]].add(row["world"])
    report = {
        "development_episode": args.dev_episode,
        "total_frames": len(result),
        "splits": {
            split: {"frames": sum(counts.values()), "by_class": dict(counts),
                    "episodes": sorted(episodes_by_split[split]),
                    "worlds": sorted(worlds_by_split[split])}
            for split, counts in sorted(by_split.items())
        },
        "limitations": [
            "Development uses a separate episode of the same old case1 map; it is not a map-independent test.",
            "External map families have incomplete class coverage and small samples; report per-class support.",
            "Corrected 3m case1 shares artwork with old case1, but differs in scale and camera view.",
            "Training layout 15 and held-out IRO layouts share official source tiles; this tests layout and lighting transfer, not unseen artwork.",
            "Training S layout 16 and mirrored S holdout 17 share source tiles; the latter is a layout/pose holdout, not an independent material test.",
        ],
    }
    (output.parent / "split_manifest_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Split manifest: {output}")


if __name__ == "__main__":
    main()
