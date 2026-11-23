"""Make a contact sheet of held-out frames and their 16x16 CNN inputs."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from preprocess import preprocess_image


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "controllers" / "ult_data_collector" / "dataset"
DEFAULT_PREDICTIONS = Path(__file__).resolve().parent / "output" / "predictions.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--true-class", required=True)
    parser.add_argument("--predicted-class", required=True)
    parser.add_argument("--max-images", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as stream:
        matches = [
            row for row in csv.DictReader(stream)
            if row["true_class"] == args.true_class
            and row["predicted_class"] == args.predicted_class
        ]
    if not matches:
        raise ValueError("No matching predictions")
    positions = np.linspace(0, len(matches) - 1, min(args.max_images, len(matches)), dtype=int)
    selected = [matches[int(position)] for position in positions]
    columns = 3
    tile_width, tile_height = 320, 160
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image_path = args.dataset / row["center_image"]
        with Image.open(image_path) as source:
            raw = source.convert("RGB").resize((160, 120))
        processed = (preprocess_image(image_path) * 255).astype(np.uint8)
        cnn_input = Image.fromarray(processed).resize((120, 120), Image.Resampling.NEAREST)
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(raw, (x, y + 24))
        sheet.paste(cnn_input.convert("RGB"), (x + 170, y + 24))
        draw.text((x + 4, y + 4), f"frame {row['frame_id']}  {args.true_class} -> {args.predicted_class}", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Saved {len(selected)} examples to {args.output}")


if __name__ == "__main__":
    main()
