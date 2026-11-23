"""Make a center-camera review sheet from the latest trajectory preview."""

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
status = json.loads((ROOT / "auto_collection_status.json").read_text(encoding="utf-8"))
if status["mode"] != "trajectory_preview":
    raise ValueError("Latest run is not a trajectory preview")
groups = defaultdict(list)
for row in status["trajectory_samples"]:
    groups[row["pose_id"]].append(row)
width, height = 180, 155
sheet = Image.new("RGB", (width * max(map(len, groups.values())), height * len(groups)), "#111111")
draw = ImageDraw.Draw(sheet)
for row_index, (pose, samples) in enumerate(groups.items()):
    for col_index, item in enumerate(samples):
        image = Image.open(item["center_image"]).convert("RGB")
        sheet.paste(image, (col_index * width, row_index * height + 30))
        draw.text((col_index * width + 2, row_index * height + 2),
                  f"{pose} {item['sample']}", fill="yellow")
        draw.text((col_index * width + 2, row_index * height + 15),
                  f"{item['phase']} y={item['map_y_px']:.0f}", fill="yellow")
path = ROOT / "preview" / status["episode_id"] / "trajectory_review_sheet.png"
sheet.save(path)
print(path)
