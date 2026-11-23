"""Make a visual review sheet for an automatic pose preview episode."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: make_preview_sheet.py <preview episode directory>")
    directory = Path(sys.argv[1]).resolve()
    starts = sorted(directory.glob("*_cam_center_start.jpg"))
    if not starts:
        raise SystemExit("No center-camera preview images")
    width, height = 6 * 324, len(starts) * 266
    sheet = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for row, start in enumerate(starts):
        pose_id = start.name.removesuffix("_cam_center_start.jpg")
        for col, (stage, camera) in enumerate(
            (stage, camera)
            for stage in ("start", "end")
            for camera in ("cam_left", "cam_center", "cam_right")
        ):
            path = directory / f"{pose_id}_{camera}_{stage}.jpg"
            with Image.open(path) as source:
                image = source.convert("RGB").resize((320, 240), Image.Resampling.NEAREST)
            x, y = col * 324, row * 266
            sheet.paste(image, (x, y + 22))
            draw.text((x + 2, y + 3), f"{pose_id} {stage} {camera}", fill="yellow")
    output = directory / "review_sheet.png"
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
