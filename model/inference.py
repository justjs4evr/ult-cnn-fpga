"""Shared CPU inference for the project's six-class ULT-CNN checkpoints.

``load_model(path)`` returns ``(model, checkpoint)``.  Use
``image_tensor(pil_image, checkpoint["preprocessing"])`` for camera images,
or ``predict_path(model, checkpoint, path)`` for a saved image.  Predictions
contain ``predicted_class``, ``confidence``, and all six ``probabilities``.
"""

import io
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import ULTCNN


CLASSES = ("straight", "left", "right", "sharp_left", "sharp_right", "stop")
SOURCE_CLASSES = (
    "crossroads", "crosswalk", "gentle_left", "gentle_right",
    "sharp_left", "sharp_right", "straight", "unknown",
)
SOURCE_CLASS_MAPPING = {
    "straight": "straight", "gentle_left": "left", "gentle_right": "right",
    "sharp_left": "sharp_left", "sharp_right": "sharp_right",
    "crossroads": "stop", "crosswalk": "stop",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate_preprocessing(preprocessing):
    """Validate the input contract and return normalized metadata.

    The CNN architecture requires 16x16 inputs.  The research pipeline uses
    a 64x64 intermediate image.  Crop and threshold are read from the model
    rather than silently replaced by the current training defaults.
    """
    if not isinstance(preprocessing, dict):
        raise ValueError("Checkpoint preprocessing must be a dictionary")
    for key, expected in (("input_size", (16, 16)),
                          ("intermediate_size", (64, 64))):
        value = preprocessing.get(key)
        if not isinstance(value, (tuple, list)) or tuple(value) != expected:
            raise ValueError(f"Checkpoint {key} must be {expected}, got {value!r}")
    crop = preprocessing.get("crop_top_ratio")
    if (isinstance(crop, bool) or not isinstance(crop, (int, float))
            or not math.isfinite(crop) or not 0 <= crop < 1):
        raise ValueError("Checkpoint crop_top_ratio must be finite and in [0, 1)")
    threshold = preprocessing.get("white_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, int)
            or not 0 <= threshold <= 255):
        raise ValueError("Checkpoint white_threshold must be an integer in [0, 255]")
    mode = preprocessing.get("mode")
    if mode not in {"binary", "grayscale"}:
        raise ValueError("Checkpoint mode must be binary or grayscale")
    return {
        "input_size": (16, 16), "intermediate_size": (64, 64),
        "crop_top_ratio": float(crop), "white_threshold": threshold,
        "mode": mode,
    }


def load_model(checkpoint_path):
    """Load a trusted local checkpoint on CPU and enforce six-class order."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu",
                            weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary, not a bare state_dict")
    if tuple(checkpoint.get("classes", ())) != CLASSES:
        raise ValueError(f"Checkpoint classes must be in this order: {CLASSES}")
    validate_preprocessing(checkpoint.get("preprocessing"))
    if "model_state" not in checkpoint:
        raise ValueError("Checkpoint has no model_state")
    if checkpoint.get('architecture') == 'crossing_context_64':
        from crossing_context64_model import CrossingContext64CNN
        model=CrossingContext64CNN(checkpoint['context_threshold'])
    elif checkpoint.get('architecture') == 'crossing_context_shared':
        from crossing_context_model import CrossingContextCNN
        model=CrossingContextCNN(checkpoint['context_threshold'])
    else:
        model = ULTCNN(num_classes=len(CLASSES))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def preprocess_pil(image, preprocessing):
    """Return the actual 16x16 float32 CNN input for one PIL image.

    Binary mode thresholds the intermediate image, then averages it down;
    therefore the final input can contain values between 0 and 1.
    """
    settings = validate_preprocessing(preprocessing)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    gray = image.convert("L")
    width, height = gray.size
    if width < 1 or height < 1:
        raise ValueError("Cannot preprocess an empty image")
    roi = gray.crop((0, int(height * settings["crop_top_ratio"]), width, height))
    medium = roi.resize(settings["intermediate_size"], Image.Resampling.BOX)
    if settings["mode"] == "binary":
        binary = np.where(np.asarray(medium) >= settings["white_threshold"],
                          255, 0).astype(np.uint8)
        small = Image.fromarray(binary).resize(settings["input_size"],
                                               Image.Resampling.BOX)
    else:
        small = medium.resize(settings["input_size"], Image.Resampling.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


def preprocess_path(path, preprocessing):
    """Preprocess an image file without changing its contents."""
    with Image.open(path) as image:
        return preprocess_pil(image, preprocessing)


def image_tensor(image, preprocessing):
    """Return the checkpoint's input: legacy16 or paired context16/context64."""
    if preprocessing.get('paired_context64'):
        cropped=preprocess_pil(image,preprocessing)
        repeated=np.repeat(np.repeat(cropped,4,axis=0),4,axis=1)
        full=np.asarray(image.convert('L').resize((64,64),Image.Resampling.BOX),dtype=np.float32)/255
        return torch.from_numpy(np.stack([repeated,full]).copy()).reshape(1,2,64,64)
    if preprocessing.get('paired_full_frame_context'):
        cropped=preprocess_pil(image,preprocessing)
        full=preprocess_pil(image,{**preprocessing,'crop_top_ratio':0.})
        return torch.from_numpy(np.stack([cropped,full]).copy()).reshape(1,2,16,16)
    array = preprocess_pil(image, preprocessing)
    return torch.from_numpy(array.copy()).reshape(1, 1, 16, 16)


def predict_image(model, checkpoint, image):
    """Predict one PIL image using the pair returned by ``load_model``."""
    if tuple(checkpoint.get("classes", ())) != CLASSES:
        raise ValueError("Prediction requires the six-class checkpoint order")
    with torch.inference_mode():
        probabilities = model(image_tensor(image, checkpoint["preprocessing"])).softmax(1)[0]
    values = probabilities.cpu().tolist()
    index = int(probabilities.argmax())
    return {
        "predicted_class": CLASSES[index], "confidence": float(values[index]),
        "probabilities": dict(zip(CLASSES, values)),
    }


def predict_path(model, checkpoint, path):
    """Predict one saved image; no label or filename convention is required."""
    with Image.open(path) as image:
        return predict_image(model, checkpoint, image)


def image_paths(path):
    """Find supported image files recursively, in a deterministic order."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image file or folder not found: {path}")
    return sorted(item for item in path.rglob("*")
                  if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
