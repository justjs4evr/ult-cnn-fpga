"""Compare fixed equal-probability ensemble on the existing held-out image family."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import ULTCNN
from preprocess import preprocess_image


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "output_visible_pair_20260927" / "best_model.pt"
NEW = ROOT / "output_current_six_binary" / "best_model.pt"
MANIFEST = ROOT / "output_current_six_binary" / "split_manifest.csv"
OUTPUT = ROOT / "legacy_ensemble_evaluation.json"
CLASSES = ("straight", "left", "right", "sharp_left", "sharp_right", "stop")


def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(saved["classes"]) != CLASSES or saved["preprocessing"]["mode"] != "binary":
        raise ValueError(f"Incompatible checkpoint: {path}")
    model = ULTCNN()
    model.load_state_dict(saved["model_state"])
    return model.eval()


def score(predicted, truth):
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    recall = {}
    f1 = {}
    for i, name in enumerate(CLASSES):
        tp = matrix[i, i]
        support = matrix[i].sum()
        predicted_count = matrix[:, i].sum()
        recall[name] = float(tp / support) if support else None
        f1[name] = float(2 * tp / (support + predicted_count)) if support + predicted_count else None
    return {"accuracy": float(np.trace(matrix) / len(truth)),
            "macro_f1": float(np.mean([v for v in f1.values() if v is not None])),
            "recall": recall, "confusion_matrix": matrix.tolist()}


def main():
    torch.set_num_threads(2)
    old, new = load_model(OLD), load_model(NEW)
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test_img"]
    truth = np.asarray([CLASSES.index(row["training_class"]) for row in rows])
    probabilities = {"old": [], "new": []}
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            pixels = np.stack([preprocess_image(row["path"], mode="binary") for row in batch])
            inputs = torch.from_numpy(pixels[:, None].copy())
            probabilities["old"].append(old(inputs).softmax(1).numpy())
            probabilities["new"].append(new(inputs).softmax(1).numpy())
            if start % 2048 == 0:
                print(f"Processed {start + len(batch)}/{len(rows)}", flush=True)
    old_p = np.concatenate(probabilities["old"])
    new_p = np.concatenate(probabilities["new"])
    combined_p = (old_p + new_p) / 2
    old_pred, new_pred, combined_pred = (p.argmax(1) for p in (old_p, new_p, combined_p))
    report = {
        "models": {"old": str(OLD), "new": str(NEW)},
        "combination": "Equal average of the two softmax probability vectors; no retraining or fitted weights.",
        "evaluation": "test_img filename family from current six-class manifest; not a verified independent map or real robot.",
        "samples": len(rows),
        "class_support": {name: int(np.count_nonzero(truth == i)) for i, name in enumerate(CLASSES)},
        "old": score(old_pred, truth), "new": score(new_pred, truth),
        "equal_ensemble": score(combined_pred, truth),
        "model_disagreement": int(np.count_nonzero(old_pred != new_pred)),
        "ensemble_better_than_both": int(np.count_nonzero((combined_pred == truth) & (old_pred != truth) & (new_pred != truth))),
        "ensemble_worse_than_both": int(np.count_nonzero((combined_pred != truth) & (old_pred == truth) & (new_pred == truth))),
        "limitations": "The source-family split may contain similar scenes; old checkpoint training overlap was not independently audited.",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: report[name][key] for key in ("accuracy", "macro_f1")}
                      for name in ("old", "new", "equal_ensemble")}, indent=2))


if __name__ == "__main__":
    main()
