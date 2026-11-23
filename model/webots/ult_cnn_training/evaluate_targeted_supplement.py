"""Compare the targeted-data model with the prior model on unchanged holdouts."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import ULTCNN
from preprocess import preprocess_image
from train import CLASSES, DEFAULT_DATASET


ROOT = Path(__file__).resolve().parent
COLLECTOR = ROOT.parent / "controllers" / "ult_auto_collector"
RECOVERY_EPISODE = "20260927T055335522964Z"
NEW_EPISODES = {"20260927T054940743428Z", "20260927T055138768948Z"}
MODELS = {"previous": ROOT / "output_visible_pair_20260927" / "best_model.pt",
          "supplemented": ROOT / "output_targeted_20260927" / "best_model.pt"}


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def direction(label):
    if label in ("left", "sharp_left"):
        return "left"
    if label in ("right", "sharp_right"):
        return "right"
    return "none"


def main():
    status = json.loads((COLLECTOR / f"auto_collection_status_{RECOVERY_EPISODE}.json").read_text(encoding="utf-8"))
    if status["mode"] != "trajectory_preview" or status["episode_id"] != RECOVERY_EPISODE:
        raise ValueError("Unexpected recovery capture")
    candidate_keys = {("east_exit_return_offset", 10), ("west_exit_return_offset", 10),
                      ("west_exit_return_offset", 11)}
    candidates = []
    for item in status["trajectory_samples"]:
        if (item["pose_id"], item["sample"]) not in candidate_keys:
            continue
        if item["map_y_px"] <= 260 or item["phase"] not in ("return_turn", "return_cross"):
            raise ValueError(f"Candidate not outside lane during return: {item}")
        candidates.append({"episode_id": RECOVERY_EPISODE, "pose_id": item["pose_id"],
                           "sample": item["sample"], "phase": item["phase"],
                           "map_x_px": item["map_x_px"], "map_y_px": item["map_y_px"],
                           "scene_context": "lane_recovery",
                           "expected_direction": "left" if item["pose_id"].startswith("east") else "right",
                           "class_label_status": "unassigned_strength", "center_image": item["center_image"]})
    if len(candidates) != len(candidate_keys):
        raise RuntimeError("Missing reviewed recovery candidates")
    manifest = ROOT / "recovery_candidate_supplement_20260927.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidates[0].keys())
        writer.writeheader()
        writer.writerows(candidates)

    manifest_rows = rows(DEFAULT_DATASET / "training_manifest.csv")
    source = {row["frame_id"]: row for row in manifest_rows}
    split_rows = rows(ROOT / "split_manifest.csv")
    holdout = [item for item in split_rows if item["split"] != "train"]
    new_rows = [row for row in manifest_rows if row["episode_id"] in NEW_EPISODES]
    if len(new_rows) != 27:
        raise RuntimeError(f"Expected 27 new curated frames, found {len(new_rows)}")
    capture_predictions = []
    report = {"recovery_episode": RECOVERY_EPISODE, "recovery_candidates": len(candidates),
              "new_curated_training_frames": len(new_rows), "models": {}}
    for name, checkpoint_path in MODELS.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError("Model class order mismatch")
        model = ULTCNN().eval()
        model.load_state_dict(checkpoint["model_state"])
        mode = checkpoint["preprocessing"]["mode"]
        images = np.stack([preprocess_image(DEFAULT_DATASET / source[item["frame_id"]]["center_image"], mode)
                           for item in holdout])
        with torch.no_grad():
            pred = model(torch.from_numpy(images[:, None])).argmax(dim=1).tolist()
        by_split = defaultdict(list)
        for item, label in zip(holdout, pred):
            by_split[item["split"]].append((item["label_name"], CLASSES[label]))
        groups = {}
        for split_name, pairs in sorted(by_split.items()):
            support = Counter(truth for truth, _ in pairs)
            correct = Counter(truth for truth, guess in pairs if truth == guess)
            groups[split_name] = {"frames": len(pairs), "correct": sum(correct.values()),
                                  "by_class": {label: {"correct": correct[label], "total": support[label]}
                                               for label in CLASSES if support[label]}}
        recovery_results = []
        for row in candidates:
            image = preprocess_image(Path(row["center_image"]), mode)
            with torch.no_grad():
                guess = CLASSES[int(model(torch.from_numpy(image[None, None])).argmax(dim=1).item())]
            recovery_results.append({"pose_id": row["pose_id"], "sample": row["sample"],
                                     "expected_direction": row["expected_direction"],
                                     "prediction": guess, "direction_correct": direction(guess) == row["expected_direction"]})
        new_images = np.stack([preprocess_image(DEFAULT_DATASET / row["center_image"], mode) for row in new_rows])
        with torch.no_grad():
            new_pred = model(torch.from_numpy(new_images[:, None])).argmax(dim=1).tolist()
        for row, label in zip(new_rows, new_pred):
            capture_predictions.append({"model": name, "frame_id": row["frame_id"],
                                        "episode_id": row["episode_id"], "truth": row["label_name"],
                                        "prediction": CLASSES[label], "center_image": row["center_image"]})
        report["models"][name] = {"checkpoint": str(checkpoint_path), "holdouts": groups,
                                  "recovery_direction_correct": sum(x["direction_correct"] for x in recovery_results),
                                  "recovery_results": recovery_results,
                                  "new_training_frames_correct": sum(row["label_name"] == CLASSES[label]
                                                                     for row, label in zip(new_rows, new_pred))}
    (ROOT / "targeted_supplement_comparison_20260927.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "targeted_capture_predictions_20260927.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=capture_predictions[0].keys())
        writer.writeheader()
        writer.writerows(capture_predictions)
    selected = [row for row in capture_predictions if row["model"] == "supplemented"]
    sheet = Image.new("RGB", (960, 160 * ((len(selected) + 2) // 3)), "#EEEEEE")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        path = DEFAULT_DATASET / row["center_image"]
        with Image.open(path) as source_image:
            original = source_image.convert("RGB").resize((160, 120))
        pixels = np.rint(preprocess_image(path, "binary") * 255).astype(np.uint8)
        cnn = Image.fromarray(pixels).resize((120, 120), Image.Resampling.NEAREST).convert("RGB")
        x, y = (index % 3) * 320, (index // 3) * 160
        sheet.paste(original, (x, y + 24))
        sheet.paste(cnn, (x + 170, y + 24))
        draw.text((x + 3, y + 4), f"{row['frame_id']} {row['truth']} > {row['prediction']}", fill="black")
    sheet.save(ROOT / "targeted_capture_review_20260927.png")
    for name, item in report["models"].items():
        print(name, {key: f"{value['correct']}/{value['frames']}" for key, value in item["holdouts"].items()},
              f"recovery={item['recovery_direction_correct']}/{len(candidates)}")


if __name__ == "__main__":
    main()
