"""Find training inputs closest to reviewed lane-return images."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from preprocess import preprocess_image
from train import DEFAULT_DATASET, load_rows


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "recovery_candidate_manifest_20260927.csv"
OUTPUT = ROOT / "recovery_neighbor_audit_20260927.json"
SHEET = ROOT / "output" / "recovery_neighbor_review_20260927.png"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    candidates = read_csv(INPUT)
    roles = {row["frame_id"]: row["split"] for row in read_csv(ROOT / "split_manifest.csv")}
    train = [row for row in load_rows(DEFAULT_DATASET / "training_manifest.csv")
             if roles[row["frame_id"]] == "train"]
    train_pixels = np.stack([preprocess_image(DEFAULT_DATASET / row["center_image"], mode="binary")
                             for row in train]).reshape(len(train), -1)
    report = []
    for candidate in candidates:
        image_path = Path(candidate["center_image"])
        pixels = preprocess_image(image_path, mode="binary").reshape(-1)
        distances = np.count_nonzero(train_pixels != pixels, axis=1)
        best = np.argsort(distances)[:5]
        report.append({"pose_id": candidate["pose_id"], "sample": int(candidate["sample"]),
                       "expected_direction": candidate["expected_direction"],
                       "candidate_image": str(image_path),
                       "nearest": [{"frame_id": int(train[index]["frame_id"]),
                                    "label": train[index]["label_name"],
                                    "different_pixels": int(distances[index]),
                                    "image": str(DEFAULT_DATASET / train[index]["center_image"])}
                                   for index in best]})
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    width, height = 180, 160
    sheet = Image.new("RGB", (width * 6, height * len(report)), "#121212")
    draw = ImageDraw.Draw(sheet)
    for y, result in enumerate(report):
        members = [{"image": result["candidate_image"], "label": result["expected_direction"],
                    "different_pixels": 0}] + result["nearest"]
        for x, item in enumerate(members):
            original = Image.open(item["image"]).convert("RGB").resize((160, 120))
            sheet.paste(original, (x * width, y * height + 28))
            label = (f"candidate {result['pose_id']} {result['sample']}" if x == 0 else
                     f"{item['frame_id']} {item['label']} d={item['different_pixels']}")
            draw.text((x * width + 2, y * height + 4), label, fill="yellow")
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET)
    print("nearest top-5 labels", Counter(item["label"] for row in report for item in row["nearest"]))
    for row in report:
        print(row["pose_id"], row["sample"], row["expected_direction"],
              [(item["label"], item["different_pixels"]) for item in row["nearest"]])
    print(OUTPUT)
    print(SHEET)


if __name__ == "__main__":
    main()
