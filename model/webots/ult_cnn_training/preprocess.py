"""Deterministic camera preprocessing shared by training and later deployment."""

from pathlib import Path

import numpy as np
from PIL import Image


INPUT_SIZE = (16, 16)
INTERMEDIATE_SIZE = (64, 64)
CROP_TOP_RATIO = 0.25
WHITE_THRESHOLD = 127


def preprocess_image(path: str | Path, mode: str = "binary") -> np.ndarray:
    """Return one 16x16 float32 grayscale image in the [0, 1] range."""
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        roi = gray.crop((0, int(height * CROP_TOP_RATIO), width, height))
        medium = roi.resize(INTERMEDIATE_SIZE, Image.Resampling.BOX)
    if mode == "binary":
        binary = np.where(np.asarray(medium) >= WHITE_THRESHOLD, 255, 0).astype(np.uint8)
        small = Image.fromarray(binary).resize(INPUT_SIZE, Image.Resampling.BOX)
    elif mode == "grayscale":
        small = medium.resize(INPUT_SIZE, Image.Resampling.BOX)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    return np.asarray(small, dtype=np.float32) / 255.0
