"""Train a first six-class ULT-CNN baseline on curated Webots center images."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import ULTCNN
from preprocess import (
    CROP_TOP_RATIO,
    INPUT_SIZE,
    INTERMEDIATE_SIZE,
    WHITE_THRESHOLD,
    preprocess_image,
)


CLASSES = ("straight", "left", "right", "sharp_left", "sharp_right", "stop")
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "controllers" / "ult_data_collector" / "dataset"


def load_rows(manifest_path):
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("episode_id")]
    if not rows:
        raise ValueError(f"No labelled frames in {manifest_path}")
    for row in rows:
        label = int(row["label_id"])
        if label not in range(len(CLASSES)) or row["label_name"] != CLASSES[label]:
            raise ValueError(f"Invalid class on frame {row['frame_id']}")
    return rows


def load_arrays(rows, dataset_dir, preprocess_mode):
    images = np.stack([
        preprocess_image(dataset_dir / row["center_image"], mode=preprocess_mode)
        for row in rows
    ]).astype(np.float32)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    return torch.from_numpy(images[:, None]), torch.from_numpy(labels)


def score_model(model, images, labels):
    model.eval()
    with torch.no_grad():
        logits = model(images)
        prediction = logits.argmax(dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    f1_by_class = []
    recall_by_class = []
    for class_id in range(len(CLASSES)):
        tp = int(matrix[class_id, class_id])
        support = int(matrix[class_id].sum())
        predicted = int(matrix[:, class_id].sum())
        recall = tp / support if support else None
        precision = tp / predicted if predicted else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if recall and precision else 0.0
        recall_by_class.append(recall)
        f1_by_class.append(f1 if support else None)
    valid_f1 = [value for value in f1_by_class if value is not None]
    return {
        "accuracy": float(np.trace(matrix) / len(truth)),
        "macro_f1": float(np.mean(valid_f1)),
        "per_class_recall": dict(zip(CLASSES, recall_by_class)),
        "per_class_f1": dict(zip(CLASSES, f1_by_class)),
        "support": dict(zip(CLASSES, matrix.sum(axis=1).tolist())),
        "confusion_matrix": matrix.tolist(),
        "predictions": prediction,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output_grouped")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Default: ult_cnn_training/split_manifest.csv")
    parser.add_argument("--exclude-train-frames", type=Path, default=None,
                        help="JSON frame-id list to omit from training only for an experiment")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess-mode", choices=("binary", "grayscale"), default="binary")
    parser.add_argument("--mirror-augment", action="store_true",
                        help="Mirror training images and swap left/right labels")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch size, and patience must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    dataset_dir = args.dataset.resolve()
    manifest = dataset_dir / "training_manifest.csv"
    source = dataset_dir / "labels.csv"
    split_path = args.split_manifest or Path(__file__).resolve().parent / "split_manifest.csv"
    if source.stat().st_mtime > manifest.stat().st_mtime:
        raise RuntimeError("training_manifest.csv is stale; run prepare_ult_dataset.py first")
    if manifest.stat().st_mtime > split_path.stat().st_mtime:
        raise RuntimeError("split_manifest.csv is stale; run build_split_manifest.py first")
    rows = load_rows(manifest)
    with split_path.open(newline="", encoding="utf-8-sig") as stream:
        split_rows = list(csv.DictReader(stream))
    roles = {}
    for item in split_rows:
        key = item["frame_id"]
        if key in roles:
            raise ValueError(f"Duplicate frame in split manifest: {key}")
        roles[key] = item
    if set(roles) != {row["frame_id"] for row in rows}:
        raise ValueError("Split manifest and training manifest contain different frames")
    for row in rows:
        item = roles[row["frame_id"]]
        if (item["episode_id"], item["label_name"]) != (row["episode_id"], row["label_name"]):
            raise ValueError(f"Split manifest disagrees on frame {row['frame_id']}")
    train_rows = [row for row in rows if roles[row["frame_id"]]["split"] == "train"]
    val_rows = [row for row in rows if roles[row["frame_id"]]["split"] == "dev"]
    excluded_train_ids = set()
    if args.exclude_train_frames:
        frame_ids = json.loads(args.exclude_train_frames.read_text(encoding="utf-8"))
        if not isinstance(frame_ids, list) or any(not isinstance(frame_id, int) for frame_id in frame_ids):
            raise ValueError("--exclude-train-frames must contain a JSON list of integer frame IDs")
        excluded_train_ids = {str(frame_id) for frame_id in frame_ids}
        train_ids = {row["frame_id"] for row in train_rows}
        if excluded_train_ids - train_ids:
            raise ValueError(f"Frames absent from training split: {sorted(excluded_train_ids - train_ids)}")
        train_rows = [row for row in train_rows if row["frame_id"] not in excluded_train_ids]
    test_splits = sorted({item["split"] for item in split_rows if item["split"].startswith("test_")})
    if not train_rows or not val_rows or not test_splits:
        raise ValueError("Split manifest must have train, dev, and external tests")
    if any(item["split"] not in {"train", "dev", *test_splits} for item in split_rows):
        raise ValueError("Unknown split in split manifest")
    val_episodes = sorted({row["episode_id"] for row in val_rows})
    if len(val_episodes) != 1:
        raise ValueError("Development split must contain one whole episode")
    val_episode = val_episodes[0]
    if set(row["episode_id"] for row in train_rows) & set(val_episodes):
        raise ValueError("Training and development episodes overlap")
    train_counts = Counter(int(row["label_id"]) for row in train_rows)
    if any(train_counts[class_id] == 0 for class_id in range(len(CLASSES))):
        raise ValueError("Training split must contain every class")

    train_x, train_y = load_arrays(train_rows, dataset_dir, args.preprocess_mode)
    val_x, val_y = load_arrays(val_rows, dataset_dir, args.preprocess_mode)
    if args.mirror_augment:
        mirror_class = torch.tensor([0, 2, 1, 4, 3, 5], dtype=torch.long)
        original_x, original_y = train_x, train_y
        train_x = torch.cat((original_x, torch.flip(original_x, dims=(3,))), dim=0)
        train_y = torch.cat((original_y, mirror_class[original_y]), dim=0)
    effective_train_counts = Counter(train_y.tolist())
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = ULTCNN()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 1440, parameter_count
    weights = torch.tensor(
        [len(train_y) / (len(CLASSES) * effective_train_counts[i])
         for i in range(len(CLASSES))],
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    checkpoint = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_y)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()
        train_loss = loss_sum / len(train_y)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "preprocessing": {
                    "input_size": INPUT_SIZE,
                    "intermediate_size": INTERMEDIATE_SIZE,
                    "crop_top_ratio": CROP_TOP_RATIO,
                    "white_threshold": WHITE_THRESHOLD,
                    "mode": args.preprocess_mode,
                },
                "training_episodes": sorted({row["episode_id"] for row in train_rows}),
                "excluded_train_frames": sorted(int(frame_id) for frame_id in excluded_train_ids),
                "mirror_augment": args.mirror_augment,
                "validation_episode": val_episode,
                "split_manifest": str(split_path),
            }, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"])
    results = score_model(model, val_x, val_y)
    predictions = results.pop("predictions")
    external_tests = {}
    for split in test_splits:
        test_rows = [row for row in rows if roles[row["frame_id"]]["split"] == split]
        test_x, test_y = load_arrays(test_rows, dataset_dir, args.preprocess_mode)
        test_results = score_model(model, test_x, test_y)
        test_results.pop("predictions")
        external_tests[split] = test_results
    report = {
        "parameter_count": parameter_count,
        "training_frames": len(train_rows),
        "augmented_training_examples": len(train_y),
        "mirror_augment": args.mirror_augment,
        "validation_frames": len(val_rows),
        "train_by_class": dict(zip(CLASSES, [train_counts[i] for i in range(len(CLASSES))])),
        "effective_train_by_class": dict(zip(
            CLASSES, [effective_train_counts[i] for i in range(len(CLASSES))])),
        "excluded_train_frames": sorted(int(frame_id) for frame_id in excluded_train_ids),
        "validation_episode": val_episode,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "preprocessing": saved["preprocessing"],
        "validation": results,
        "external_tests": external_tests,
        "split_manifest": str(split_path),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode_id", "frame_id", "true_class", "predicted_class", "center_image"])
        for row, prediction in zip(val_rows, predictions):
            writer.writerow([
                row["episode_id"], row["frame_id"], row["label_name"],
                CLASSES[int(prediction)], row["center_image"],
            ])
    print(json.dumps({"best_epoch": best_epoch, "validation": results}, indent=2))
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
