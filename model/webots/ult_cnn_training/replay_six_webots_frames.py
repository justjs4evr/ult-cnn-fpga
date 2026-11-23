"""Replay recorded Webots center-camera frames through the new six-class CNN."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from model import ULTCNN
from preprocess import preprocess_image


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "controllers" / "ult_data_collector" / "dataset"
CHECKPOINT = ROOT / "output_current_six_binary" / "best_model.pt"
OUTPUT = ROOT / "webots_six_replay"
WORLDS = {"ult_auto_iro_test_17.wbt", "ult_auto_case1_3m.wbt", "ult_auto_iro_lane_14.wbt"}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    classes = tuple(saved["classes"])
    if saved["preprocessing"]["mode"] != "binary":
        raise ValueError("Expected binary model")
    model = ULTCNN()
    model.load_state_dict(saved["model_state"])
    model.eval()
    worlds = {}
    for path in DATASET.glob("episode_*.json"):
        info = json.loads(path.read_text(encoding="utf-8-sig"))
        worlds[info["episode_id"]] = info["world"]
    with (DATASET / "labels.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if worlds.get(row["episode_id"]) in WORLDS]
    rows.sort(key=lambda row: (worlds[row["episode_id"]], row["episode_id"], int(row["frame_id"])))
    counts = defaultdict(Counter)
    outputs = []
    with torch.inference_mode():
        for row in rows:
            image = DATASET / row["center_image"]
            tensor = torch.from_numpy(preprocess_image(image, mode="binary").copy()).reshape(1, 1, 16, 16)
            probability = model(tensor).softmax(dim=1)[0]
            predicted = classes[int(probability.argmax())]
            world = worlds[row["episode_id"]]
            counts[world][predicted] += 1
            outputs.append({"world": world, "episode_id": row["episode_id"], "frame_id": row["frame_id"],
                            "sim_time_s": row["sim_time_s"], "recorded_label": row["label_name"],
                            "predicted_class": predicted, "confidence": round(float(probability.max()), 4),
                            "image": str(image)})
    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=outputs[0].keys())
        writer.writeheader()
        writer.writerows(outputs)
    report = {"checkpoint": str(CHECKPOINT), "frames": len(outputs),
              "by_world": {name: dict(count) for name, count in counts.items()},
              "scope": "Recorded Webots center-camera frame replay; not a live closed-loop run.",
              "limitations": "Many recorded images may overlap training data; recorded labels predate the user's manual reclassification."}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
