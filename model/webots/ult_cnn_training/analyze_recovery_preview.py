"""Review lane-return trajectories without inventing sharp-vs-gentle labels."""

import csv
import json
from pathlib import Path

import torch

from model import ULTCNN
from preprocess import preprocess_image
from train import CLASSES


ROOT = Path(__file__).resolve().parent
COLLECTOR = ROOT.parent / "controllers" / "ult_auto_collector"
EPISODE = "20260927T011427036215Z"
REVIEWED_SAMPLES = {17, 18}
MODEL_DIRS = ("output_sharp_s16", "output_visible_corner_20260927",
              "output_visible_pair_20260927", "output_mirror_20260927",
              "output_mirror_balanced_20260927")


def direction(label):
    if label in ("left", "sharp_left"):
        return "left"
    if label in ("right", "sharp_right"):
        return "right"
    return "none"


def main():
    status_path = COLLECTOR / f"auto_collection_status_{EPISODE}.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["episode_id"] != EPISODE or status["mode"] != "trajectory_preview":
        raise ValueError("The reviewed trajectory preview is not the current run")
    records = []
    for row in status["trajectory_samples"]:
        if row["sample"] not in REVIEWED_SAMPLES:
            continue
        if row["map_y_px"] <= 260:
            raise ValueError("Selected sample is no longer outside the inner lane line")
        expected = "left" if row["pose_id"] == "east_inside_exit_return" else "right"
        records.append({"episode_id": EPISODE, "pose_id": row["pose_id"],
                        "sample": row["sample"], "phase": row["phase"],
                        "map_x_px": row["map_x_px"], "map_y_px": row["map_y_px"],
                        "scene_context": "lane_recovery", "expected_direction": expected,
                        "class_label_status": "unassigned_strength",
                        "center_image": row["center_image"]})
    if len(records) != 4:
        raise ValueError("Expected four reviewed return-turn samples")
    csv_path = ROOT / "recovery_candidate_manifest_20260927.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    report = {"episode_id": EPISODE, "candidate_frames": len(records), "models": {}}
    for model_dir in MODEL_DIRS:
        checkpoint = torch.load(ROOT / model_dir / "best_model.pt", map_location="cpu",
                                weights_only=False)
        model = ULTCNN()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        mode = checkpoint["preprocessing"]["mode"]
        model_rows = []
        for row in records:
            pixels = preprocess_image(Path(row["center_image"]), mode=mode)
            image = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction = CLASSES[int(model(image).argmax(dim=1).item())]
            model_rows.append({"pose_id": row["pose_id"], "sample": row["sample"],
                               "expected_direction": row["expected_direction"],
                               "prediction": prediction, "predicted_direction": direction(prediction),
                               "direction_correct": direction(prediction) == row["expected_direction"]})
        report["models"][model_dir] = {
            "direction_correct": sum(row["direction_correct"] for row in model_rows),
            "total": len(model_rows), "frames": model_rows,
        }
    output = ROOT / "recovery_direction_probe_20260927.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {"direction_correct": item["direction_correct"],
                             "total": item["total"]}
                      for name, item in report["models"].items()}, indent=2))
    print(csv_path)
    print(output)


if __name__ == "__main__":
    main()
