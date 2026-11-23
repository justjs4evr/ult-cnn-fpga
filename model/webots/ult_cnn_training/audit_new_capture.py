"""Audit newly captured episodes without changing images or labels."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from model import ULTCNN
from train import CLASSES, DEFAULT_DATASET, load_arrays, load_rows, score_model


def signature(image):
    return hashlib.sha256(image.numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "output_grouped" / "best_model.pt")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "new_capture_audit.json")
    args = parser.parse_args()
    rows = load_rows(args.dataset / "training_manifest.csv")
    chosen = set(args.episodes)
    if len(chosen) != len(args.episodes):
        raise ValueError("Episode ids must be unique")
    new_rows = [row for row in rows if row["episode_id"] in chosen]
    if {row["episode_id"] for row in new_rows} != chosen:
        raise ValueError("An episode is absent from the curated manifest")
    old_rows = [row for row in rows if row["episode_id"] not in chosen]

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(saved["classes"]) != CLASSES:
        raise ValueError("Checkpoint classes mismatch")
    mode = saved["preprocessing"]["mode"]
    model = ULTCNN()
    model.load_state_dict(saved["model_state"])
    new_x, new_y = load_arrays(new_rows, args.dataset, mode)
    old_x, _ = load_arrays(old_rows, args.dataset, mode)
    prior = defaultdict(set)
    for row, image in zip(old_rows, old_x):
        prior[signature(image)].add(row["label_name"])
    new_signatures = [signature(image) for image in new_x]
    new_by_signature = defaultdict(list)
    for row, sig in zip(new_rows, new_signatures):
        new_by_signature[sig].append({"frame_id": int(row["frame_id"]),
                                      "label": row["label_name"]})
    exact_overlap = [
        {"frame_id": int(row["frame_id"]), "label": row["label_name"],
         "prior_labels": sorted(prior[sig])}
        for row, sig in zip(new_rows, new_signatures) if sig in prior
    ]
    episode_results = {}
    for episode in sorted(chosen):
        indices = [i for i, row in enumerate(new_rows) if row["episode_id"] == episode]
        metrics = score_model(model, new_x[indices], new_y[indices])
        metrics.pop("predictions")
        episode_results[episode] = {
            "frames": len(indices),
            "unique_16x16_inputs": len({new_signatures[i] for i in indices}),
            "by_class": dict(Counter(new_rows[i]["label_name"] for i in indices)),
            "accuracy": metrics["accuracy"],
            "per_class_recall": metrics["per_class_recall"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "total_frames": len(new_rows),
        "unique_16x16_inputs": len(set(new_signatures)),
        "duplicate_inputs_within_new": [items for items in new_by_signature.values()
                                        if len(items) > 1],
        "exact_overlap_with_earlier_curated_inputs": exact_overlap,
        "episodes": episode_results,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
