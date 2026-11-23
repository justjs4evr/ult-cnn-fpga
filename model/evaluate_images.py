"""Evaluate a six-class ULT-CNN on labelled image folders or a split manifest.

Folder evaluation alone does not prove an independent test: images may have
been used for training.  A manifest selects the exact saved split membership;
its filename/source split still does not prove independent maps or drives.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference import (
    CLASSES, SOURCE_CLASSES, SOURCE_CLASS_MAPPING, image_paths, load_model,
    image_tensor,
)


def labelled_folder_rows(dataset):
    rows = []
    for source_class in SOURCE_CLASSES:
        if source_class == "unknown":
            continue
        folder = dataset / source_class
        if not folder.is_dir():
            continue
        for path in image_paths(folder):
            rows.append({"path": path, "source_class": source_class,
                         "true_class": SOURCE_CLASS_MAPPING[source_class],
                         "split": "folder"})
    return rows


def manifest_rows(manifest, dataset, split):
    # Historical runs retain their original manifests after dataset renaming.
    rename_map = {}
    rename_history = dataset / "rename_history.csv"
    if rename_history.is_file():
        with rename_history.open(newline="", encoding="utf-8-sig") as stream:
            rename_map = {str(Path(row["old_path"]).resolve()).casefold(): Path(row["new_path"])
                          for row in csv.DictReader(stream)}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "source_class", "training_class", "split"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Manifest needs these columns: {sorted(required)}")
        source_rows = list(reader)
    selected, seen = [], set()
    for row in source_rows:
        if row["split"] != split or row["source_class"] == "unknown":
            continue
        source_class, true_class = row["source_class"], row["training_class"]
        if source_class not in SOURCE_CLASS_MAPPING:
            raise ValueError(f"Unknown source_class in manifest: {source_class!r}")
        if true_class != SOURCE_CLASS_MAPPING[source_class]:
            raise ValueError(f"Manifest label mapping differs on {row['path']}")
        path = Path(row["path"]).expanduser()
        if not path.is_absolute():
            path = dataset / path
        path = path.resolve()
        if not path.is_file():
            path = rename_map.get(str(path).casefold(), path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        if path in seen:
            raise ValueError(f"Manifest repeats an image in {split}: {path}")
        seen.add(path)
        selected.append({"path": path, "source_class": source_class,
                         "true_class": true_class, "split": split})
    return selected


def classification_metrics(truth, prediction):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    support, precision, recall, f1 = {}, {}, {}, {}
    for i, name in enumerate(CLASSES):
        tp, total, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        support[name] = total
        precision[name] = tp / predicted if predicted else 0.0
        recall[name] = tp / total if total else None
        f1[name] = 2 * tp / (total + predicted) if total + predicted else 0.0
    present = [f1[name] for name in CLASSES if support[name]]
    return {
        "accuracy": float(np.trace(matrix) / len(truth)),
        "macro_f1": float(np.mean(list(f1.values()))),
        "macro_f1_present_classes": float(np.mean(present)),
        "macro_f1_definition": "Mean F1 over all six classes; absent classes have F1=0.",
        "support": support, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": "rows=true class, columns=predicted class; order=classes",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/main_classified_images"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for metrics.json and predictions.csv")
    parser.add_argument("--manifest", type=Path,
                        help="Saved split_manifest.csv from classified-image training")
    parser.add_argument("--split", choices=("test_img", "dev_frame", "train"),
                        help="Manifest split; defaults to test_img when --manifest is supplied")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.split and not args.manifest:
        parser.error("--split requires --manifest")
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"Dataset folder not found: {dataset}")
    split = (args.split or "test_img") if args.manifest else "folder"
    rows = (manifest_rows(args.manifest.expanduser().resolve(), dataset, split)
            if args.manifest else labelled_folder_rows(dataset))
    if not rows:
        parser.error("No labelled images found for the selected dataset/split")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, checkpoint = load_model(args.checkpoint)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            tensors = []
            for row in rows[start:start + args.batch_size]:
                with Image.open(row["path"]) as image:
                    tensors.append(image_tensor(image, checkpoint["preprocessing"]))
            tensor = torch.cat(tensors, dim=0)
            probabilities.extend(model(tensor).softmax(1).cpu().tolist())
    values = np.asarray(probabilities)
    prediction = values.argmax(1)
    truth = np.asarray([CLASSES.index(row["true_class"]) for row in rows])
    note = ("Selected saved manifest membership. Filename/source splits do not verify "
            "independent drives or maps; train split measures training-data fit."
            if args.manifest else
            "Folder evaluation can include training images. It is not a verified independent test.")
    report = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "dataset": str(dataset), "classes": CLASSES,
        "source_class_mapping": SOURCE_CLASS_MAPPING,
        "preprocessing": checkpoint["preprocessing"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "samples": len(rows), "split": split,
        "manifest": str(args.manifest.expanduser().resolve()) if args.manifest else None,
        "evaluation_note": note,
        **classification_metrics(truth, prediction),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = ["path", "split", "source_class", "true_class", "predicted_class",
              "confidence", "correct", *[f"prob_{name}" for name in CLASSES]]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            pred = int(prediction[i])
            writer.writerow({**row, "predicted_class": CLASSES[pred],
                             "confidence": float(values[i, pred]),
                             "correct": bool(pred == truth[i]),
                             **{f"prob_{name}": float(values[i, j])
                                for j, name in enumerate(CLASSES)}})
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"Images: {len(rows)}  Accuracy: {report['accuracy']:.2%}  Macro F1: {report['macro_f1']:.4f}")
    print(note)
    print(f"Saved: {output / 'metrics.json'}")
    print(f"Saved: {output / 'predictions.csv'}")


if __name__ == "__main__":
    main()
