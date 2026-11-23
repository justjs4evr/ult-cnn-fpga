"""Find identical model inputs that cross train, dev, or test splits."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from train import DEFAULT_DATASET, load_arrays, load_rows


ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "split_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        roles = {row["frame_id"]: row["split"] for row in csv.DictReader(stream)}
    rows = load_rows(DEFAULT_DATASET / "training_manifest.csv")
    images, _ = load_arrays(rows, DEFAULT_DATASET, "binary")
    by_signature = defaultdict(list)
    for row, image in zip(rows, images):
        by_signature[hashlib.sha256(image.numpy().tobytes()).hexdigest()].append({
            "frame_id": int(row["frame_id"]), "episode_id": row["episode_id"],
            "label": row["label_name"], "split": roles[row["frame_id"]],
        })
    cross_split = [items for items in by_signature.values()
                   if len({item["split"] for item in items}) > 1]
    report = {
        "total_curated_frames": len(rows),
        "unique_16x16_inputs": len(by_signature),
        "cross_split_duplicate_groups": cross_split,
        "cross_split_duplicate_group_count": len(cross_split),
    }
    (ROOT / "split_overlap_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
