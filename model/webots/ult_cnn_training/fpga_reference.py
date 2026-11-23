"""Export a provisional full-CNN fixed-point reference and verification vectors.

This is a software contract for a future RTL implementation, not an RTL build.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from model import ULTCNN
from preprocess import preprocess_image
from train import CLASSES, DEFAULT_DATASET


ROOT = Path(__file__).resolve().parent
WEIGHT_SCALE = 1 << 14  # signed int16, Q2.14
ACT_SCALE = 1 << 10  # signed int16, Q6.10


def round_shift(values, bits):
    """Round to nearest with ties away from zero, including negative values."""
    magnitude = (np.abs(values) + (1 << (bits - 1))) >> bits
    return np.where(values < 0, -magnitude, magnitude)


def clamp16(values, stats, name):
    stats[name] = max(stats.get(name, 0), int(np.max(np.abs(values))))
    clipped = np.clip(values, -32768, 32767)
    stats[name + "_saturated"] = stats.get(name + "_saturated", 0) + int(np.count_nonzero(clipped != values))
    return clipped.astype(np.int64)


def conv(x, weight, bias, stats, name):
    # PyTorch Conv2d is cross-correlation; weight order is O,I,Y,X.
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    patches = sliding_window_view(padded, (3, 3), axis=(1, 2))
    mac = np.tensordot(weight, patches, axes=((1, 2, 3), (0, 3, 4)))
    stats[name + "_mac_abs_max"] = max(stats.get(name + "_mac_abs_max", 0), int(np.max(np.abs(mac))))
    values = round_shift(mac, 14) + bias[:, None, None]
    return clamp16(np.maximum(values, 0), stats, name)


def pool(x):
    channels, height, width = x.shape
    return x.reshape(channels, height // 2, 2, width // 2, 2).max(axis=(2, 4))


def dense(x, weight, bias, relu, stats, name):
    mac = weight @ x
    stats[name + "_mac_abs_max"] = max(stats.get(name + "_mac_abs_max", 0), int(np.max(np.abs(mac))))
    values = round_shift(mac, 14) + bias
    if relu:
        values = np.maximum(values, 0)
    return clamp16(values, stats, name)


def integer_inference(image_bytes, parameters, stats):
    # Exact conversion from 8-bit preprocessing output to Q6.10.
    x = ((image_bytes.astype(np.int64) * ACT_SCALE * 2 + 255) // (2 * 255))[None]
    c1 = conv(x, parameters["features.0.weight"], parameters["features.0.bias"], stats, "conv1")
    p1 = pool(c1)
    c2 = conv(p1, parameters["features.3.weight"], parameters["features.3.bias"], stats, "conv2")
    p2 = pool(c2)
    f1 = dense(p2.reshape(-1), parameters["classifier.1.weight"], parameters["classifier.1.bias"], True, stats, "fc1")
    logits = dense(f1, parameters["classifier.3.weight"], parameters["classifier.3.bias"], False, stats, "fc2")
    return {"input": x, "conv1": c1, "pool1": p1, "conv2": c2, "pool2": p2, "fc1": f1, "logits": logits}


def write_hex(path, array, width):
    mask = (1 << (4 * width)) - 1
    path.write_text("".join(f"{int(value) & mask:0{width}X}\n" for value in array.reshape(-1)), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "output_visible_pair_20260927" / "best_model.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "fpga_reference_output")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order differs from six-class contract")
    if checkpoint["preprocessing"]["mode"] != "binary":
        raise ValueError("This reference currently requires binary preprocessing")
    model = ULTCNN().eval()
    model.load_state_dict(checkpoint["model_state"])
    parameters = {}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    tensor_manifest = []
    for name, tensor in model.state_dict().items():
        scale = WEIGHT_SCALE if name.endswith("weight") else ACT_SCALE
        values = np.rint(tensor.numpy().astype(np.float64) * scale).astype(np.int64)
        if values.min() < -32768 or values.max() > 32767:
            raise ValueError(f"Parameter out of int16 range: {name}")
        parameters[name] = values
        filename = name.replace(".", "_") + ".hex"
        write_hex(output / filename, values, 4)
        tensor_manifest.append({"name": name, "shape": list(values.shape), "scale": scale, "file": filename,
                                "min": int(values.min()), "max": int(values.max())})

    with (args.dataset / "training_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = {row["frame_id"]: row for row in csv.DictReader(stream)}
    with (ROOT / "split_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        split_rows = list(csv.DictReader(stream))
    selected = [row for row in split_rows if row["split"] == "dev" or row["split"].startswith("test_")]
    stats = {}
    matched = 0
    mismatches = []
    examples = []
    saved_classes = set()
    for split in selected:
        row = rows[split["frame_id"]]
        image = preprocess_image(args.dataset / row["center_image"], mode="binary")
        image_bytes = np.rint(image * 255).astype(np.uint8)
        with torch.no_grad():
            float_logits = model(torch.from_numpy(image[None, None])).numpy()[0]
        result = integer_inference(image_bytes, parameters, stats)
        float_class = int(float_logits.argmax())
        integer_class = int(result["logits"].argmax())
        if float_class == integer_class:
            matched += 1
        else:
            mismatches.append({"frame_id": int(split["frame_id"]), "split": split["split"],
                               "float_class": CLASSES[float_class], "integer_class": CLASSES[integer_class],
                               "float_logits": [round(float(v), 5) for v in float_logits],
                               "integer_logits_q8_8": result["logits"].tolist()})
        truth = split["label_name"]
        if truth not in saved_classes and float_class == integer_class == CLASSES.index(truth):
            saved_classes.add(truth)
            stem = f"vector_{truth}"
            write_hex(output / f"{stem}_input.hex", image_bytes, 2)
            for layer, values in result.items():
                if layer != "input":
                    write_hex(output / f"{stem}_{layer}.hex", values, 4)
            examples.append({"frame_id": int(split["frame_id"]), "split": split["split"],
                             "truth": truth, "float_prediction": CLASSES[float_class],
                             "integer_prediction": CLASSES[integer_class], "prefix": stem})
    if saved_classes != set(CLASSES):
        raise RuntimeError(f"No correctly classified reference vector for: {sorted(set(CLASSES) - saved_classes)}")
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
              "status": "provisional_reference_not_deployment_ready", "class_order": list(CLASSES),
              "input": "16x16 uint8, channel/row/column order; byte-to-Q6.10 = round(byte*1024/255)",
              "arithmetic": "signed int16 Q6.10 activation and bias; signed int16 Q2.14 weight; int64 MAC; rounded right shift 14 (ties away from zero), add bias, ReLU if present, clamp int16; 2x2 max pool; argmax lowest index on tie",
              "tensor_order": "row-major flatten, Conv O/I/Y/X, Dense O/I", "tensors": tensor_manifest,
              "holdout_images": len(selected), "float_integer_class_matches": matched,
              "float_integer_agreement": matched / len(selected), "mismatches": mismatches,
              "max_abs_and_saturation": stats, "vectors": examples}
    (output / "reference_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(selected), "agreement": matched / len(selected),
                      "mismatches": len(mismatches), "saturation": {k: v for k, v in stats.items() if k.endswith("_saturated")},
                      "vectors": len(examples), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
