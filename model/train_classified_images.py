"""Train the original ULT-CNN backbone on the manually sorted image folders.

Explicit train/val/test folders take priority. Flat folders use a reproducible,
class-stratified input-group split independent of image filenames. This fallback
does not establish independent collection sessions or a real-robot test.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import ULTCNN
from dataset_splits import grouped_stratified_splits
from preprocess import (
    CROP_TOP_RATIO, INPUT_SIZE, INTERMEDIATE_SIZE, WHITE_THRESHOLD,
    preprocess_image,
)


SOURCE_CLASSES = (
    "crossroads", "crosswalk", "gentle_left", "gentle_right",
    "sharp_left", "sharp_right", "straight", "unknown",
)
LEGACY_CLASSES = ("straight", "left", "right", "sharp_left", "sharp_right", "stop")
LEGACY_MAP = {
    "straight": "straight", "gentle_left": "left", "gentle_right": "right",
    "sharp_left": "sharp_left", "sharp_right": "sharp_right",
    "crossroads": "stop", "crosswalk": "stop",
}


def score(model, x, y, batch_size, classes):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            predictions.append(model(x[start:start + batch_size]).argmax(1).numpy())
    pred = np.concatenate(predictions)
    truth = y.numpy()
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    recall, precision, f1 = {}, {}, {}
    for i, name in enumerate(classes):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        recall[name] = tp / support if support else None
        precision[name] = tp / predicted if predicted else None
        f1[name] = (2 * tp / (support + predicted)) if support + predicted else None
    return {
        "accuracy": float(np.trace(matrix) / len(truth)),
        "macro_f1": float(np.mean([v for v in f1.values() if v is not None])),
        "support": dict(zip(classes, matrix.sum(axis=1).tolist())),
        "recall": recall, "precision": precision, "f1": f1,
        "confusion_matrix": matrix.tolist(),
    }, pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"C:\main_classified_images"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "user_runs" / ("train_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    parser.add_argument("--mode", choices=("binary", "grayscale"), default="binary")
    parser.add_argument("--crop-top-ratio", type=float, default=CROP_TOP_RATIO,
                        help="Fraction removed from the top: 0 = full frame (default), 0.25 = legacy crop")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mirror-augment", action="store_true")
    parser.add_argument("--label-scheme", choices=("eight", "legacy-six"), default="legacy-six")
    parser.add_argument("--prepare-only", action="store_true", help="Audit data and save the split manifest without training")
    parser.add_argument("--split-layout", choices=("auto", "filename"), default="auto",
                        help="auto: explicit folders or filename-independent grouped 80/10/10 split; filename: legacy reproduction only")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Initialize matching architecture from a prior checkpoint before training")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience and batch-size must be positive")
    if not np.isfinite(args.crop_top_ratio) or not 0 <= args.crop_top_ratio < 1:
        parser.error("crop-top-ratio must be finite and in [0, 1)")
    preprocessing = {"input_size": INPUT_SIZE, "intermediate_size": INTERMEDIATE_SIZE,
                     "crop_top_ratio": args.crop_top_ratio, "white_threshold": WHITE_THRESHOLD,
                     "mode": args.mode}
    classes = LEGACY_CLASSES if args.label_scheme == "legacy-six" else SOURCE_CLASSES
    mirror_label = (torch.tensor([0, 2, 1, 4, 3, 5], dtype=torch.long)
                    if args.label_scheme == "legacy-six" else
                    torch.tensor([0, 1, 3, 2, 5, 4, 6, 7], dtype=torch.long))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best_model.pt").exists():
        raise ValueError("Output already contains a model. Choose a new output folder to preserve prior results.")

    paths, labels, splits, source_labels = [], [], [], []
    for name in SOURCE_CLASSES:
        if args.label_scheme == "legacy-six" and name == "unknown":
            continue
        folder = args.dataset / name
        if not folder.is_dir():
            raise ValueError(f"Missing class folder: {folder}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            paths.append(path)
            labels.append(classes.index(LEGACY_MAP[name] if args.label_scheme == "legacy-six" else name))
            source_labels.append(name)
            relative = path.relative_to(folder)
            explicit = {"train": "train", "val": "dev_frame", "dev": "dev_frame", "test": "test_img"}
            split = explicit.get(relative.parts[0].lower()) if len(relative.parts) > 1 and args.split_layout == "auto" else None
            splits.append(split or (("test_img" if path.name.startswith("img_") else
                                     "dev_frame" if path.name.startswith("frame_") else "train")
                                    if args.split_layout == "filename" else None))
    if not paths:
        raise ValueError("No images found")

    print(f"Preprocessing {len(paths)} images ({args.mode}, top crop={args.crop_top_ratio:.0%})", flush=True)
    pixels = np.empty((len(paths), 16, 16), dtype=np.uint8)
    for i, path in enumerate(paths):
        try:
            pixels[i] = np.rint(preprocess_image(path, mode=args.mode, crop_top_ratio=args.crop_top_ratio) * 255).astype(np.uint8)
        except Exception as error:
            raise ValueError(f"Cannot read image: {path}: {error}") from error
        if (i + 1) % 10000 == 0:
            print(f"Preprocessed {i + 1}/{len(paths)}", flush=True)

    # Identical CNN inputs must never appear on both sides of a split.
    hashes = [hashlib.blake2b(image.tobytes(), digest_size=16).digest() for image in pixels]
    # Group mirror-equivalent inputs before splitting so augmentation cannot leak
    # an exact validation/test input into training.
    split_hashes = [min(value, hashlib.blake2b(np.flip(image, axis=1).tobytes(), digest_size=16).digest())
                    for value, image in zip(hashes, pixels)] if args.mirror_augment else hashes
    if args.split_layout == "auto":
        explicit_count = sum(split is not None for split in splits)
        if explicit_count and explicit_count != len(splits):
            raise ValueError("Mixed explicit and flat images. Put every image in train/val/test folders, or keep every image in flat class folders.")
        split_method = "explicit_folders" if explicit_count else "grouped_stratified_80_10_10"
        if not explicit_count:
            splits = grouped_stratified_splits(split_hashes, labels, args.seed)
            print("Using filename-independent grouped 80/10/10 split. Collection-session independence is not verified.", flush=True)
    else:
        split_method = "legacy_filename"
    test_hashes = {split_hashes[i] for i, split in enumerate(splits) if split == "test_img"}
    dev_hashes = {split_hashes[i] for i, split in enumerate(splits) if split == "dev_frame" and split_hashes[i] not in test_hashes}
    removed = Counter()
    for i, split in enumerate(splits):
        if split == "dev_frame" and split_hashes[i] in test_hashes:
            splits[i] = "excluded_duplicate"
            removed["dev_frame"] += 1
        elif split == "train" and split_hashes[i] in (test_hashes | dev_hashes):
            splits[i] = "excluded_duplicate"
            removed["train"] += 1
    groups = {}
    for i, digest in enumerate(hashes):
        groups.setdefault(digest, []).append(i)
    conflicting = [group for group in groups.values() if len({labels[i] for i in group}) > 1]
    with (output / "input_label_conflicts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("group", "path", "source_class", "training_class", "split"))
        for number, group in enumerate(conflicting, 1):
            writer.writerows((number, str(paths[i]), source_labels[i], classes[labels[i]], splits[i]) for i in group)
    if conflicting:
        print(f"WARNING: {len(conflicting)} identical 16x16 input groups have different labels. See input_label_conflicts.csv.", flush=True)
    indices = {name: np.asarray([i for i, split in enumerate(splits) if split == name], dtype=np.int64)
               for name in ("train", "dev_frame", "test_img")}
    for name, idx in indices.items():
        counts = Counter(labels[i] for i in idx)
        if not idx.size or any(counts[i] == 0 for i in range(len(classes))):
            missing = [classes[i] for i in range(len(classes)) if counts[i] == 0]
            raise ValueError(f"Split {name} needs every output class. Missing: {missing}. Put independent sessions in class/train, class/val and class/test folders.")
        print(f"{name}: {len(idx)} " + str({classes[i]: counts[i] for i in range(len(classes))}), flush=True)

    with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "source_class", "training_class", "split"])
        writer.writerows((str(path), source_label, classes[label], split)
                         for path, source_label, label, split in zip(paths, source_labels, labels, splits))
    audit = {
        "dataset": str(args.dataset.resolve()), "classes": classes, "mode": args.mode,
        "preprocessing": preprocessing,
        "mirror_augment": args.mirror_augment, "split_layout": args.split_layout,
        "split_method": split_method,
        "split_counts": {name: {classes[c]: int(sum(labels[i] == c for i in idx)) for c in range(len(classes))}
                         for name, idx in indices.items()},
        "excluded_duplicates": dict(removed), "input_label_conflict_groups": len(conflicting),
        "input_label_conflict_images": sum(map(len, conflicting)),
        "limitations": "Content-group stratification or explicit folders do not verify map/session independence; near duplicates may remain.",
    }
    (output / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(f"Data audit saved: {output}", flush=True)
        return

    def tensors(split):
        idx = indices[split]
        return (torch.from_numpy(pixels[idx].astype(np.float32)[:, None] / 255),
                torch.tensor(np.asarray(labels, dtype=np.int64)[idx]))

    train_x, train_y = tensors("train")
    dev_x, dev_y = tensors("dev_frame")
    test_x, test_y = tensors("test_img")
    if args.mirror_augment:
        original_x, original_y = train_x, train_y
        train_x = torch.cat([original_x, torch.flip(original_x, dims=(3,))])
        train_y = torch.cat([original_y, mirror_label[original_y]])
    counts = Counter(train_y.tolist())
    weights = torch.tensor([len(train_y) / (len(classes) * counts[i]) for i in range(len(classes))], dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = ULTCNN(num_classes=len(classes))
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if tuple(initial["classes"]) != tuple(classes):
            raise ValueError("Initial checkpoint class order does not match this training run")
        if initial["preprocessing"]["mode"] != args.mode:
            raise ValueError("Initial checkpoint preprocessing mode does not match")
        if initial["preprocessing"]["crop_top_ratio"] != args.crop_top_ratio:
            raise ValueError("Initial checkpoint crop differs. Train from scratch, or use the checkpoint's --crop-top-ratio.")
        model.load_state_dict(initial["model_state"])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    checkpoint = output / "best_model.pt"
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
            dev_losses = []
            for start in range(0, len(dev_x), args.batch_size):
                dev_losses.append((len(dev_x[start:start + args.batch_size]),
                                   criterion(model(dev_x[start:start + args.batch_size]),
                                             dev_y[start:start + args.batch_size]).item()))
        val_loss = sum(n * value for n, value in dev_losses) / len(dev_x)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_y), "dev_loss": val_loss})
        print(f"epoch {epoch:02d} train={history[-1]['train_loss']:.4f} dev={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "source_class_mapping": LEGACY_MAP if args.label_scheme == "legacy-six" else None,
                        "preprocessing": preprocessing, "mirror_augment": args.mirror_augment,
                        "epoch": epoch,
                        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
                        "source_dataset": str(args.dataset.resolve()), "split_manifest": str(output / "split_manifest.csv")}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"])
    dev_metrics, dev_pred = score(model, dev_x, dev_y, args.batch_size, classes)
    test_metrics, test_pred = score(model, test_x, test_y, args.batch_size, classes)
    source_recall = {}
    for split, idx, pred in (("dev_frame", indices["dev_frame"], dev_pred),
                             ("test_img", indices["test_img"], test_pred)):
        source_recall[split] = {}
        for source in SOURCE_CLASSES:
            if args.label_scheme == "legacy-six" and source == "unknown":
                continue
            positions = [j for j, i in enumerate(idx) if source_labels[i] == source]
            expected = classes.index(LEGACY_MAP[source] if args.label_scheme == "legacy-six" else source)
            source_recall[split][source] = {"support": len(positions),
                                            "recall": float(np.mean(pred[positions] == expected)) if positions else None}
    report = {
        "classes": classes, "source_class_mapping": LEGACY_MAP if args.label_scheme == "legacy-six" else None,
        "source_class_recall": source_recall,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "preprocess_mode": args.mode, "mirror_augment": args.mirror_augment,
        "preprocessing": preprocessing,
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "source_dataset": str(args.dataset.resolve()),
        "split_rule": split_method + "; exact input groups stay together or are purged with test priority, including mirror equivalents when augmented",
        "dataset_audit": audit,
        "excluded_duplicates": dict(removed),
        "best_epoch": best_epoch, "best_dev_loss": best_loss,
        "history": history, "development": dev_metrics, "test": test_metrics,
        "limitations": ["Split membership does not establish verified independent drives or maps.",
                        "Near-duplicate images may still cross the splits.",
                        "No physical robot camera images or driving test are included."],
    }
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "split", "source_class", "true_class", "predicted_class"])
        for split, idx, pred in (("dev_frame", indices["dev_frame"], dev_pred),
                                 ("test_img", indices["test_img"], test_pred)):
            writer.writerows((str(paths[i]), split, source_labels[i], classes[labels[i]], classes[int(p)])
                             for i, p in zip(idx, pred))
    print(json.dumps({"best_epoch": best_epoch, "development": dev_metrics, "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
