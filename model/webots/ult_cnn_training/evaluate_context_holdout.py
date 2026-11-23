"""Score saved checkpoints on the six reviewed mixed-context holdout frames."""

import csv
import json
from pathlib import Path

import torch

from model import ULTCNN
from train import CLASSES, DEFAULT_DATASET, load_arrays, load_rows


ROOT = Path(__file__).resolve().parent
EPISODE = "20260927T010300033888Z"
MODELS = (
    "output_sharp_s16",
    "output_visible_corner_20260927",
    "output_visible_pair_20260927",
    "output_mirror_20260927",
    "output_mirror_balanced_20260927",
)


def main():
    rows = [row for row in load_rows(DEFAULT_DATASET / "training_manifest.csv")
            if row["episode_id"] == EPISODE]
    pose_ids = {(row["episode_id"], row["frame_id"]): row["pose_id"]
                for row in csv.DictReader((DEFAULT_DATASET / "auto_pose_manifest.csv").open(
                    newline="", encoding="utf-8-sig"))}
    metadata = json.loads((DEFAULT_DATASET / f"episode_{EPISODE}.json").read_text(encoding="utf-8"))
    contexts = metadata["pose_contexts"]
    result = {"episode_id": EPISODE, "frames": len(rows), "models": {}}
    for name in MODELS:
        checkpoint = torch.load(ROOT / name / "best_model.pt", map_location="cpu", weights_only=False)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError(f"Class mismatch: {name}")
        model = ULTCNN()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        images, _ = load_arrays(rows, DEFAULT_DATASET, checkpoint["preprocessing"]["mode"])
        with torch.no_grad():
            predictions = model(images).argmax(dim=1).tolist()
        frame_results = []
        for row, predicted in zip(rows, predictions):
            pose = pose_ids[(row["episode_id"], row["frame_id"])]
            frame_results.append({"frame_id": int(row["frame_id"]), "pose_id": pose,
                                  "scene_context": contexts[pose], "truth": row["label_name"],
                                  "prediction": CLASSES[predicted],
                                  "correct": CLASSES[predicted] == row["label_name"]})
        correct = sum(item["correct"] for item in frame_results)
        result["models"][name] = {"correct": correct, "total": len(frame_results),
                                  "by_context": {
                                      context: {"correct": sum(item["correct"] for item in frame_results
                                                               if item["scene_context"] == context),
                                                "total": sum(item["scene_context"] == context
                                                             for item in frame_results)}
                                      for context in sorted(set(contexts.values()))},
                                  "frames": frame_results}
    path = ROOT / "context_holdout_evaluation_20260927.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
