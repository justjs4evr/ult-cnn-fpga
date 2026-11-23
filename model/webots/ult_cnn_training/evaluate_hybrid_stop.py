"""Small labeled-image audit of the hybrid crosswalk stop gate."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import ULTCNN
from preprocess import preprocess_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "controllers" / "ult_cnn_hybrid"))
from lane_tracking import detect_lane  # noqa: E402


def main():
    torch.set_num_threads(1)
    saved = torch.load(ROOT / "output_current_six_binary" / "best_model.pt",
                       map_location="cpu", weights_only=False)
    model = ULTCNN()
    model.load_state_dict(saved["model_state"])
    model.eval()
    dataset = Path(r"C:\AICAR\dataset\classified_images")
    for name in ("crosswalk", "crossroads", "straight", "sharp_left", "sharp_right"):
        all_paths = sorted((dataset / name).glob("*.jpg"))
        paths = all_paths[::max(1, len(all_paths) // 100)][:100]
        geometry_hits = confirmed_stops = 0
        with torch.inference_mode():
            for path in paths:
                gray = np.asarray(Image.open(path).convert("L"))
                geometry = detect_lane(gray).crosswalk_bands >= 5
                geometry_hits += geometry
                if geometry:
                    tensor = torch.from_numpy(preprocess_image(path).copy()).reshape(1, 1, 16, 16)
                    stop_probability = float(model(tensor).softmax(1)[0, 5])
                    confirmed_stops += stop_probability >= 0.70
        print(f"{name}: images={len(paths)} geometry={geometry_hits} stop={confirmed_stops}", flush=True)


if __name__ == "__main__":
    main()
