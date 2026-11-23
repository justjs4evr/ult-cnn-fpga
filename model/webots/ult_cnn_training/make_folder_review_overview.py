"""Make unlabeled original/16x16 overview pages for the folder review batch."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_folder_review import OUT
from preprocess import preprocess_image
from train import DEFAULT_DATASET


def main():
    with (OUT / "index.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for page, start in enumerate(range(0, len(rows), 8), start=1):
        sheet = Image.new("RGB", (1060, 1060), "#EEEEEE")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(rows[start:start + 8]):
            x, y = (offset % 2) * 530, (offset // 2) * 265
            path = DEFAULT_DATASET / row["source_image"]
            with Image.open(path) as source:
                original = source.convert("RGB").resize((320, 240), Image.Resampling.NEAREST)
            small = np.rint(preprocess_image(path) * 255).astype(np.uint8)
            cnn = Image.fromarray(small).resize((180, 180), Image.Resampling.NEAREST).convert("RGB")
            sheet.paste(original, (x + 2, y + 22))
            sheet.paste(cnn, (x + 340, y + 48))
            draw.text((x + 4, y + 3), f"frame {row['frame_id']}  {row['split']}", fill="black")
            draw.text((x + 348, y + 232), "CNN 16x16", fill="black")
            draw.line((x, y + 264, x + 529, y + 264), fill="#AAAAAA")
        output = OUT / f"사진목록_{page:02d}.png"
        sheet.save(output)
        print(output)


if __name__ == "__main__":
    main()
