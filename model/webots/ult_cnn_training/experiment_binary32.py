"""Exploratory 32x32 binary-input CNN with the same 1,440 trainable parameters."""

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import CLASSES, DEFAULT_DATASET, score_model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_targeted_binary32_20260927"


class ULTCNN32(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(3, 6, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(96, 12), nn.ReLU(), nn.Linear(12, 6))

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess32(path):
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        medium = gray.crop((0, int(height * 0.25), width, height)).resize((64, 64), Image.Resampling.BOX)
    binary = np.where(np.asarray(medium) >= 127, 255, 0).astype(np.uint8)
    small = Image.fromarray(binary).resize((32, 32), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


def arrays(group):
    images = np.stack([preprocess32(DEFAULT_DATASET / row["center_image"]) for row in group])
    labels = np.asarray([CLASSES.index(row["label_name"]) for row in group], dtype=np.int64)
    return torch.from_numpy(images[:, None]), torch.from_numpy(labels)


def main():
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    with (DEFAULT_DATASET / "training_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    with (ROOT / "split_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        roles = {row["frame_id"]: row["split"] for row in csv.DictReader(stream)}
    if set(roles) != {row["frame_id"] for row in rows}:
        raise ValueError("Training and split manifests differ")
    train_rows = [row for row in rows if roles[row["frame_id"]] == "train"]
    dev_rows = [row for row in rows if roles[row["frame_id"]] == "dev"]
    train_x, train_y = arrays(train_rows)
    dev_x, dev_y = arrays(dev_rows)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    model = ULTCNN32()
    count = sum(p.numel() for p in model.parameters())
    if count != 1440:
        raise AssertionError(count)
    counts = Counter(train_y.tolist())
    weights = torch.tensor([len(train_y) / (len(CLASSES) * counts[i]) for i in range(6)])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    OUT.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, 61):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_loss = criterion(model(dev_x), dev_y).item()
        history.append({"epoch": epoch, "dev_loss": dev_loss})
        if dev_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = dev_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "classes": CLASSES,
                        "preprocessing": {"mode": "binary32", "input_size": [32, 32],
                                          "intermediate_size": [64, 64], "crop_top_ratio": 0.25,
                                          "white_threshold": 127},
                        "architecture": "ULTCNN32_extra_maxpool", "split_manifest": str(ROOT / "split_manifest.csv")},
                       OUT / "best_model.pt")
        else:
            stale += 1
            if stale >= 12:
                break
    checkpoint = torch.load(OUT / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    scores = {}
    for split in sorted(set(roles.values()) - {"train"}):
        group = [row for row in rows if roles[row["frame_id"]] == split]
        x, y = (dev_x, dev_y) if split == "dev" else arrays(group)
        score = score_model(model, x, y)
        score.pop("predictions")
        scores[split] = score
    report = {"status": "exploratory_not_deployment_ready", "parameter_count": count,
              "training_frames": len(train_rows), "best_epoch": best_epoch,
              "best_dev_loss": best_loss, "scores": scores, "history": history,
              "compute_note": "32x32 input needs more convolution work and an extra pooling stage versus 16x16"}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: {"n": sum(score["support"].values()), "accuracy": score["accuracy"],
                              "right_recall": score["per_class_recall"]["right"]}
                      for split, score in scores.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
