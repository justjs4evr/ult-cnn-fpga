"""Predict a saved image or a folder with the six-class ULT-CNN."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from inference import CLASSES, image_paths, load_model, predict_path, preprocess_pil


def save_preview(path, preprocessing, prediction, destination):
    """Show the original and the exact input, enlarged without interpolation."""
    with Image.open(path) as image:
        original = image.convert("RGB")
        cnn_input = preprocess_pil(image, preprocessing)
    original.thumbnail((640, 480), Image.Resampling.LANCZOS)
    input_preview = Image.fromarray(np.rint(cnn_input * 255).astype(np.uint8)).resize(
        (320, 320), Image.Resampling.NEAREST).convert("RGB")
    width = max(360, original.width) + 352
    height = max(original.height, 320) + 92
    sheet = Image.new("RGB", (width, height), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), f"Prediction: {prediction['predicted_class']}  confidence={prediction['confidence']:.4f}",
              fill="black")
    draw.text((16, 38), "Original camera image", fill="black")
    second_x = max(360, original.width) + 16
    draw.text((second_x, 38), "Actual 16x16 CNN input", fill="black")
    sheet.paste(original, (16, 64))
    sheet.paste(input_preview, (second_x, 64))
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="One image or a folder searched recursively")
    parser.add_argument("--output", type=Path, help="Optional predictions CSV")
    parser.add_argument("--preview", type=Path,
                        help="Optional preview PNG of the first original image and its CNN input")
    args = parser.parse_args()
    paths = image_paths(args.input)
    if not paths:
        parser.error("No supported images found")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, checkpoint = load_model(args.checkpoint)
    stream = None
    try:
        writer = None
        if args.output:
            destination = args.output.expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = destination.open("w", newline="", encoding="utf-8-sig")
            writer = csv.DictWriter(stream, fieldnames=[
                "path", "predicted_class", "confidence", *[f"prob_{name}" for name in CLASSES]])
            writer.writeheader()
        for index, path in enumerate(paths):
            prediction = predict_path(model, checkpoint, path)
            probabilities = prediction["probabilities"]
            print(f"{path}: {prediction['predicted_class']} ({prediction['confidence']:.2%})")
            print("  " + "  ".join(f"{name}={probabilities[name]:.4f}" for name in CLASSES))
            if writer:
                writer.writerow({"path": str(path),
                                 "predicted_class": prediction["predicted_class"],
                                 "confidence": prediction["confidence"],
                                 **{f"prob_{name}": probabilities[name] for name in CLASSES}})
            if index == 0 and args.preview:
                save_preview(path, checkpoint["preprocessing"], prediction, args.preview)
    finally:
        if stream:
            stream.close()
    if args.output:
        print(f"Saved {len(paths)} predictions: {args.output.expanduser().resolve()}")
    if args.preview:
        print(f"Saved preview: {args.preview.expanduser().resolve()}")


if __name__ == "__main__":
    main()
