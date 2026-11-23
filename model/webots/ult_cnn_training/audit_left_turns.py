"""Inspect e-puck commands for context and left-turn image separability.

Wheel commands are not ground truth for the visual class: e-puck uses
differential drive, while the target vehicle maps FPGA classes to steering.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from preprocess import preprocess_image


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "controllers" / "ult_data_collector" / "dataset"
LABELS = {"left", "sharp_left"}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def representation(path, mode):
    if mode in {"binary16", "gray16"}:
        return preprocess_image(path, "binary" if mode == "binary16" else "grayscale").ravel()
    with Image.open(path) as image:
        gray = image.convert("L")
        w, h = gray.size
        roi = gray.crop((0, int(h * 0.25), w, h))
        medium = roi.resize((64, 64), Image.Resampling.BOX)
    values = np.asarray(medium, dtype=np.float32) / 255.0
    return (values >= (127 / 255.0)).astype(np.float32).ravel()


def distances(query, reference):
    # Mean squared pixel difference, clipped for floating-point roundoff.
    d = ((query * query).mean(1)[:, None]
         + (reference * reference).mean(1)[None, :]
         - 2 * (query @ reference.T) / query.shape[1])
    return np.maximum(d, 0)


def main():
    selected = [r for r in read_rows(DATASET / "training_manifest.csv") if r["label_name"] in LABELS]
    splits = {(r["episode_id"], r["frame_id"]): r["split"]
              for r in read_rows(ROOT / "split_manifest.csv")}
    sources = {}
    for path in DATASET.glob("episode_*.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        sources[metadata["episode_id"]] = metadata.get("label_source", "unknown")
    for r in selected:
        r["split"] = splits[(r["episode_id"], r["frame_id"])]
        r["source"] = sources.get(r["episode_id"], "unknown")

    commands = defaultdict(Counter)
    for r in selected:
        key = (r["label_name"], r["source"])
        commands[key][(r["left_speed"], r["right_speed"])] += 1
    print("E-PUCK COMMAND COUNTS FOR CONTEXT ONLY (left_speed, right_speed)")
    for key, counts in sorted(commands.items()):
        print(key, "n=", sum(counts.values()), counts.most_common(8))

    for mode in ("binary16", "gray16", "binary64"):
        vectors = np.stack([representation(DATASET / r["center_image"], mode) for r in selected])
        labels = np.array([r["label_name"] for r in selected])
        split = np.array([r["split"] for r in selected])
        train = split == "train"
        print("\nIMAGE MODE", mode)
        for group in ("dev", "test_iro_official_tiles", "test_iro_mirrored_s_17"):
            query = split == group
            if not query.any():
                continue
            d = distances(vectors[query], vectors[train])
            guess = labels[train][d.argmin(1)]
            true = labels[query]
            print(group, "n=", len(true), "1NN accuracy=", round(float((guess == true).mean()), 3),
                  "left=", f"{sum((guess == true) & (true == 'left'))}/{sum(true == 'left')}",
                  "sharp_left=", f"{sum((guess == true) & (true == 'sharp_left'))}/{sum(true == 'sharp_left')}")

        # Nearest cross-label pairs still present in the selected training set.
        left_ids = np.flatnonzero(train & (labels == "left"))
        sharp_ids = np.flatnonzero(train & (labels == "sharp_left"))
        d = distances(vectors[sharp_ids], vectors[left_ids])
        pairs = np.dstack(np.unravel_index(np.argsort(d, axis=None)[:8], d.shape))[0]
        print("nearest training cross-label pairs:")
        for si, li in pairs:
            a, b = selected[sharp_ids[si]], selected[left_ids[li]]
            print(a["frame_id"], a["episode_id"], "vs", b["frame_id"], b["episode_id"], "MSE", round(float(d[si, li]), 6))


if __name__ == "__main__":
    main()
