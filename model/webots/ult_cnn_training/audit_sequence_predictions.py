"""Audit prediction changes within held-out Webots capture episodes."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import ULTCNN
from preprocess import preprocess_image
from train import CLASSES, DEFAULT_DATASET


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "output_visible_pair_20260927" / "best_model.pt"
OUT = ROOT / "sequence_audit_20260927"
MAX_GAP_S = 1.5


def change_type(first, second):
    if "stop" in (first, second):
        return "stop_vs_moving"
    left = {"left", "sharp_left"}
    right = {"right", "sharp_right"}
    if (first in left and second in right) or (first in right and second in left):
        return "opposite_directions"
    if "straight" in (first, second):
        return "straight_vs_turn"
    return "gentle_vs_sharp"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def make_sheet(events, rows_by_id):
    chosen = events[:12]
    if not chosen:
        return
    width, row_height = 960, 135
    sheet = Image.new("RGB", (width, row_height * len(chosen)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, event in enumerate(chosen):
        y = index * row_height
        title = (f"{event['split']} / {event['episode_id']}   "
                 f"frames {event['frame_ids']}   truth {event['truth']}   "
                 f"pred {event['predictions']}")
        draw.text((8, y + 3), title, fill="black")
        for j, frame_id in enumerate(event["frame_ids_list"]):
            path = DEFAULT_DATASET / rows_by_id[frame_id]["center_image"]
            with Image.open(path) as source:
                image = source.convert("RGB")
            image.thumbnail((280, 102))
            sheet.paste(image, (8 + j * 315, y + 24))
            draw.text((8 + j * 315, y + 112), f"{frame_id}: {event['prediction_list'][j]}", fill="black")
        draw.line((0, y + row_height - 1, width, y + row_height - 1), fill="#BBBBBB")
    sheet.save(OUT / "aba_review.png")


def make_priority_sheet(events, rows_by_id):
    selected = sorted((e for e in events if e["change_type"] in ("opposite_directions", "stop_vs_moving")),
                      key=lambda e: (e["change_type"] != "opposite_directions", e["split"] != "dev"))[:12]
    if not selected:
        return
    width, row_height = 680, 140
    sheet = Image.new("RGB", (width, row_height * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, event in enumerate(selected):
        y = index * row_height
        draw.text((8, y + 3), f"{event['split']}  {event['change_type']}  truth {event['truth']}", fill="black")
        for j, side in enumerate(("a", "b")):
            frame_id = event[f"frame_{side}"]
            with Image.open(DEFAULT_DATASET / rows_by_id[frame_id]["center_image"]) as source:
                picture = source.convert("RGB")
            picture.thumbnail((300, 102))
            sheet.paste(picture, (8 + j * 330, y + 22))
            draw.text((8 + j * 330, y + 120), f"{frame_id}: {event[f'prediction_{side}']}", fill="black")
        draw.line((0, y + row_height - 1, width, y + row_height - 1), fill="#BBBBBB")
    sheet.save(OUT / "priority_review.png")


def main():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = ULTCNN().eval()
    model.load_state_dict(checkpoint["model_state"])
    rows_by_id = {row["frame_id"]: row for row in read_csv(DEFAULT_DATASET / "training_manifest.csv")}
    split = [row for row in read_csv(ROOT / "split_manifest.csv")
             if row["split"] == "dev" or row["split"].startswith("test_")]
    for item in split:
        source = rows_by_id[item["frame_id"]]
        if (item["episode_id"], item["label_name"]) != (source["episode_id"], source["label_name"]):
            raise ValueError(f"Manifest mismatch: {item['frame_id']}")
    images = np.stack([preprocess_image(DEFAULT_DATASET / rows_by_id[item["frame_id"]]["center_image"],
                                        mode=checkpoint["preprocessing"]["mode"]) for item in split])
    with torch.no_grad():
        logits = model(torch.from_numpy(images[:, None])).numpy()
    predicted = logits.argmax(axis=1)
    grouped = defaultdict(list)
    for item, prediction, scores in zip(split, predicted, logits):
        source = rows_by_id[item["frame_id"]]
        grouped[(item["split"], item["episode_id"])].append({
            "frame_id": item["frame_id"], "time": float(source["sim_time_s"]),
            "truth": item["label_name"], "prediction": CLASSES[int(prediction)],
            "margin": float(np.sort(scores)[-1] - np.sort(scores)[-2]),
        })
    pair_count = Counter()
    flips = Counter()
    flip_types = Counter()
    examples = []
    wrong_runs = []
    aba = []
    episode_stats = []
    for (split_name, episode), sequence in grouped.items():
        sequence.sort(key=lambda x: (x["time"], int(x["frame_id"])))
        for a, b in zip(sequence, sequence[1:]):
            gap = b["time"] - a["time"]
            if not (0 < gap <= MAX_GAP_S and a["truth"] == b["truth"]):
                continue
            key = (split_name, a["truth"])
            pair_count[key] += 1
            if a["prediction"] != b["prediction"]:
                flips[key] += 1
                kind = change_type(a["prediction"], b["prediction"])
                flip_types[(split_name, kind)] += 1
                examples.append({"split": split_name, "episode_id": episode,
                                 "frame_a": a["frame_id"], "frame_b": b["frame_id"],
                                 "gap_s": round(gap, 3), "truth": a["truth"],
                                 "prediction_a": a["prediction"], "prediction_b": b["prediction"],
                                 "change_type": kind,
                                 "margin_a": round(a["margin"], 4), "margin_b": round(b["margin"], 4)})
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            if not (0 < b["time"] - a["time"] <= MAX_GAP_S and
                    0 < c["time"] - b["time"] <= MAX_GAP_S):
                continue
            if a["truth"] == b["truth"] == c["truth"] and a["prediction"] == c["prediction"] != b["prediction"]:
                aba.append({"split": split_name, "episode_id": episode,
                            "frame_ids_list": [a["frame_id"], b["frame_id"], c["frame_id"]],
                            "frame_ids": ",".join(x["frame_id"] for x in (a, b, c)),
                            "truth": a["truth"], "prediction_list": [a["prediction"], b["prediction"], c["prediction"]],
                            "predictions": ",".join(x["prediction"] for x in (a, b, c)),
                            "middle_margin": round(b["margin"], 4)})
        run = []
        for current in sequence + [None]:
            continuous = (current is not None and run and current["truth"] == run[-1]["truth"] and
                          current["prediction"] == run[-1]["prediction"] and
                          0 < current["time"] - run[-1]["time"] <= MAX_GAP_S)
            if run and not continuous:
                if len(run) >= 3 and run[0]["prediction"] != run[0]["truth"]:
                    wrong_runs.append({"split": split_name, "episode_id": episode,
                                       "first_frame": run[0]["frame_id"], "last_frame": run[-1]["frame_id"],
                                       "frames": len(run), "duration_s": round(run[-1]["time"] - run[0]["time"], 3),
                                       "truth": run[0]["truth"], "prediction": run[0]["prediction"]})
                run = []
            if current is not None:
                run.append(current)
        episode_stats.append({"split": split_name, "episode_id": episode, "frames": len(sequence),
                              "correct": sum(x["truth"] == x["prediction"] for x in sequence),
                              "first_time_s": sequence[0]["time"], "last_time_s": sequence[-1]["time"]})
    OUT.mkdir(parents=True, exist_ok=True)
    aba.sort(key=lambda x: (x["split"] != "dev", -x["middle_margin"]))
    wrong_runs.sort(key=lambda x: -x["frames"])
    make_sheet(aba, rows_by_id)
    make_priority_sheet(examples, rows_by_id)
    for filename, data in (("same_label_flips.csv", examples), ("wrong_runs.csv", wrong_runs)):
        if data:
            with (OUT / filename).open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    aggregate = defaultdict(lambda: {"eligible_pairs": 0, "prediction_changes": 0})
    for (split_name, label), count in pair_count.items():
        aggregate[(split_name, label)]["eligible_pairs"] = count
        aggregate[(split_name, label)]["prediction_changes"] = flips[(split_name, label)]
    report = {"checkpoint": str(CHECKPOINT), "max_gap_s": MAX_GAP_S,
              "definition": "Adjacent selected frames in same episode, same human label, gap <= 1.5 simulated seconds; no driving-success claim",
              "frames": len(split), "episodes": len(grouped), "eligible_pairs": sum(pair_count.values()),
              "prediction_changes": sum(flips.values()), "aba_events": len(aba),
              "change_types": [{"split": key[0], "type": key[1], "count": value}
                               for key, value in sorted(flip_types.items())],
              "wrong_runs_3plus": len(wrong_runs),
              "by_split_and_label": [{"split": key[0], "label": key[1], **value}
                                     for key, value in sorted(aggregate.items())],
              "episodes_detail": episode_stats, "aba_examples": [{k: v for k, v in row.items()
                                                                    if k not in ("frame_ids_list", "prediction_list")}
                                                                   for row in aba[:20]],
              "longest_wrong_runs": wrong_runs[:20]}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("frames", "episodes", "eligible_pairs", "prediction_changes", "aba_events", "wrong_runs_3plus")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
