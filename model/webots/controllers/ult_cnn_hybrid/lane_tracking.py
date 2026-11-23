"""Geometry-first lane tracking for the 160x120 Webots center camera."""

from dataclasses import dataclass

import numpy as np


ROWS = (0.08, 0.12, 0.16, 0.24, 0.34, 0.45, 0.57, 0.68, 0.79, 0.88)


@dataclass
class LaneResult:
    center: float
    heading: float
    confidence: float
    pair_rows: int
    near_pair_rows: int
    single_rows: int
    threshold: int
    inferred_side: str | None
    crosswalk_bands: int


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(a), int(b), (a + b - 1) / 2) for a, b in zip(starts, ends)]


def _expected_width(y_ratio, width):
    t = np.clip((y_ratio - 0.34) / 0.58, 0, 1)
    return width * (0.14 + 0.78 * t)


def _crosswalk_bands(binary):
    height, width = binary.shape
    # Webots zebra bars project as several short vertical strokes between
    # the lane edges, rather than as full-width horizontal white rows.
    rows = binary[int(height * 0.25):int(height * 0.82)]
    flags = []
    for row in rows:
        short_runs = sum(2 <= end - start <= width * 0.18
                         for start, end, _ in _runs(row))
        flags.append(short_runs >= 5)
    edges = np.diff(np.r_[False, flags, False].astype(np.int8))
    spans = [end - start for start, end in zip(np.flatnonzero(edges == 1),
                                               np.flatnonzero(edges == -1))]
    return max(spans, default=0)


def detect_lane(gray, previous_center=None, previous_side=None, cnn_hint=None):
    """Track actual boundary pairs; infer the missing boundary only at low confidence."""
    height, width = gray.shape
    camera_center = (width - 1) / 2
    previous_center = camera_center if previous_center is None else previous_center
    low, high = np.percentile(gray, (10, 99.5))
    threshold = int(max(65, min(190, low + 0.56 * (high - low))))
    binary = gray >= threshold if high >= 72 else np.zeros_like(gray, dtype=bool)
    samples = []
    pair_rows = near_pair_rows = single_rows = 0
    inferred_side = previous_side
    for ratio in ROWS:
        y = min(height - 1, int(ratio * height))
        expected_width = _expected_width(ratio, width)
        candidates = [(a, b, c) for a, b, c in _runs(binary[y])
                      if 2 <= b - a <= int(width * 0.23)]
        expected_center = previous_center if not samples else samples[-1][1]
        pairs = []
        for i, left in enumerate(candidates):
            for right in candidates[i + 1:]:
                span = right[2] - left[2]
                if not width * 0.12 <= span <= width * 0.98:
                    continue
                center = (left[2] + right[2]) / 2
                score = abs(center - expected_center) / width
                score += 0.25 * abs(span - expected_width) / width
                if left[2] <= expected_center <= right[2]:
                    score -= 0.12
                pairs.append((score, center))
        if pairs:
            _, center = min(pairs)
            pair_rows += 1
            if ratio >= 0.45:
                near_pair_rows += 1
            samples.append((y, center, 1.0))
            continue
        if not candidates:
            continue
        # One visible edge: choose its identity from position and recent evidence.
        run = min(candidates, key=lambda item: abs(item[2] - expected_center))
        x = run[2]
        left_cost = abs(x - (expected_center - expected_width / 2))
        right_cost = abs(x - (expected_center + expected_width / 2))
        if previous_side == "left":
            left_cost *= 0.85
        elif previous_side == "right":
            right_cost *= 0.85
        if abs(left_cost - right_cost) < width * 0.10:
            if cnn_hint in ("right", "sharp_right"):
                left_cost *= 0.8
            elif cnn_hint in ("left", "sharp_left"):
                right_cost *= 0.8
        side = "left" if left_cost <= right_cost else "right"
        inferred_side = side
        center = x + expected_width / 2 if side == "left" else x - expected_width / 2
        center = float(np.clip(center, -width * 0.35, width * 1.35))
        single_rows += 1
        samples.append((y, center, 0.45))

    if not samples:
        return LaneResult(float(previous_center), 0.0, 0.0, 0, 0, 0, threshold,
                          inferred_side, _crosswalk_bands(binary))
    ys = np.asarray([sample[0] for sample in samples], dtype=float)
    centers = np.asarray([sample[1] for sample in samples], dtype=float)
    weights = np.asarray([sample[2] * (0.6 + sample[0] / height) for sample in samples])
    center = float(np.average(centers, weights=weights))
    heading = 0.0
    if len(samples) >= 2 and np.ptp(ys) > 8:
        slope = np.polyfit(ys, centers, 1, w=weights)[0]
        heading = float(np.clip(-slope * 0.5, -0.9, 0.9))
    confidence = float((pair_rows + 0.4 * single_rows) / len(ROWS))
    return LaneResult(center, heading, confidence, pair_rows, near_pair_rows, single_rows,
                      threshold, inferred_side, _crosswalk_bands(binary))
