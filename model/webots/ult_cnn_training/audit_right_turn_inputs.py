"""Diagnose right, sharp-right and stop image similarity without changing labels."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from preprocess import preprocess_image
from train import DEFAULT_DATASET


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "right_turn_input_audit_20260927"
LABELS = {"right", "sharp_right", "stop"}
NEW_EPISODE = "20260927T055138768948Z"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def features(path, mode):
    if mode in ("binary16", "gray16"):
        return preprocess_image(path, "binary" if mode == "binary16" else "grayscale").ravel()
    with Image.open(path) as source:
        gray = source.convert("L")
        width, height = gray.size
        medium = gray.crop((0, int(height * 0.25), width, height)).resize((64, 64), Image.Resampling.BOX)
    values = np.asarray(medium, dtype=np.float32)
    if mode == "binary64":
        values = (values >= 127).astype(np.float32) * 255
    return (values / 255).ravel()


def distances(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    squared = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * (a @ b.T)
    return np.maximum(squared / a.shape[1], 0)


def pair_sheet(pairs, source):
    # Each row shows the new right image, nearest earlier right, nearest sharp right, nearest stop.
    sheet = Image.new("RGB", (4 * 250, len(pairs) * 174), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, pair in enumerate(pairs):
        for col, entry in enumerate(pair["neighbors"]):
            frame_id = str(entry["frame_id"])
            path = DEFAULT_DATASET / source[frame_id]["center_image"]
            with Image.open(path) as original:
                raw = original.convert("RGB").resize((160, 120))
            small = np.rint(preprocess_image(path) * 255).astype(np.uint8)
            cnn = Image.fromarray(small).resize((72, 72), Image.Resampling.NEAREST).convert("RGB")
            x, y = col * 250, index * 174
            sheet.paste(raw, (x + 2, y + 25))
            sheet.paste(cnn, (x + 166, y + 49))
            draw.text((x + 3, y + 4), f"{entry['label']} {frame_id} d={entry['mse']:.4f}", fill="black")
        draw.line((0, (index + 1) * 174 - 1, sheet.width, (index + 1) * 174 - 1), fill="#bbbbbb")
    sheet.save(OUT / "new_right_nearest_review.png")


def main():
    source = {row["frame_id"]: row for row in read_csv(DEFAULT_DATASET / "training_manifest.csv")}
    selected = [row for row in read_csv(ROOT / "split_manifest.csv") if row["label_name"] in LABELS]
    train_idx = np.array([i for i, row in enumerate(selected) if row["split"] == "train"])
    query_idx = np.array([i for i, row in enumerate(selected) if row["split"] != "train" and row["label_name"] == "right"])
    new_idx = np.array([i for i, row in enumerate(selected)
                        if row["episode_id"] == NEW_EPISODE and row["label_name"] == "right"])
    if len(new_idx) != 7:
        raise RuntimeError("Expected 7 curated one-line right inputs")
    old_train_idx = np.array([i for i in train_idx if selected[i]["episode_id"] != NEW_EPISODE])
    result = {"scope": "right vs sharp_right vs stop; one-nearest-neighbor pixel diagnostic, not CNN accuracy",
              "training_reference": len(train_idx), "heldout_right": len(query_idx),
              "new_right": len(new_idx), "modes": {}, "new_right_neighbors_binary16": []}
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in ("binary16", "gray16", "binary64", "gray64"):
        vectors = np.stack([features(DEFAULT_DATASET / source[row["frame_id"]]["center_image"], mode)
                            for row in selected])
        reference = vectors[train_idx]
        d = distances(vectors[query_idx], reference)
        nearest = train_idx[d.argmin(1)]
        by_split = {}
        for split_name in sorted({selected[i]["split"] for i in query_idx}):
            positions = np.array([j for j, i in enumerate(query_idx) if selected[i]["split"] == split_name])
            guesses = [selected[nearest[j]]["label_name"] for j in positions]
            by_split[split_name] = {"right_frames": len(positions), "nearest_label_counts": dict(Counter(guesses))}
        new_d = distances(vectors[new_idx], vectors[old_train_idx])
        new_nearest = old_train_idx[new_d.argmin(1)]
        new_guesses = [selected[i]["label_name"] for i in new_nearest]
        result["modes"][mode] = {"heldout_right": by_split,
                                  "new_right_nearest_earlier_training_labels": dict(Counter(new_guesses))}
        if mode == "binary16":
            for local_index, row_index in enumerate(new_idx):
                query = selected[row_index]
                neighbors = [{"frame_id": int(query["frame_id"]), "label": "new right", "mse": 0.0}]
                for label in ("right", "sharp_right", "stop"):
                    candidates = np.array([j for j, i in enumerate(old_train_idx)
                                           if selected[i]["label_name"] == label])
                    best_local = candidates[new_d[local_index, candidates].argmin()]
                    best = selected[old_train_idx[best_local]]
                    neighbors.append({"frame_id": int(best["frame_id"]), "label": label,
                                      "mse": round(float(new_d[local_index, best_local]), 6),
                                      "episode_id": best["episode_id"]})
                result["new_right_neighbors_binary16"].append({"frame_id": int(query["frame_id"]),
                                                                  "neighbors": neighbors})
    pair_sheet(result["new_right_neighbors_binary16"], source)
    (OUT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mode: info["new_right_nearest_earlier_training_labels"]
                      for mode, info in result["modes"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
