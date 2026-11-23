"""Deterministic camera preprocessing shared by training and later deployment."""

from pathlib import Path
import math

import numpy as np
from PIL import Image


INPUT_SIZE = (16, 16)
INTERMEDIATE_SIZE = (64, 64)
CROP_TOP_RATIO = 0.0
WHITE_THRESHOLD = 127


def preprocess_image(path: str | Path, mode: str = "binary", crop_top_ratio: float = CROP_TOP_RATIO) -> np.ndarray:
    """Return one 16x16 float32 grayscale image in the [0, 1] range."""
    if isinstance(crop_top_ratio, bool) or not math.isfinite(crop_top_ratio) or not 0 <= crop_top_ratio < 1:
        raise ValueError("crop_top_ratio must be finite and in [0, 1)")
    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        roi = gray.crop((0, int(height * crop_top_ratio), width, height))
        medium = roi.resize(INTERMEDIATE_SIZE, Image.Resampling.BOX)
    if mode == "binary":
        binary = np.where(np.asarray(medium) >= WHITE_THRESHOLD, 255, 0).astype(np.uint8)
        small = Image.fromarray(binary).resize(INPUT_SIZE, Image.Resampling.BOX)
    elif mode == "grayscale":
        small = medium.resize(INPUT_SIZE, Image.Resampling.BOX)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    return np.asarray(small, dtype=np.float32) / 255.0
