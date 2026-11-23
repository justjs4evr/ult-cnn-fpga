"""Filename-independent exploratory split when collection sessions are unknown."""
import random
from collections import defaultdict


def grouped_stratified_splits(hashes, labels, seed=42):
    groups = defaultdict(list)
    for i, digest in enumerate(hashes):
        groups[digest].append(i)
    strata = defaultdict(list)
    for digest, indices in sorted(groups.items()):
        strata[tuple(sorted({labels[i] for i in indices}))].append(indices)
    rng = random.Random(seed)
    result = ['train'] * len(labels)
    for key in sorted(strata):
        values = strata[key]
        rng.shuffle(values)
        # Keep identical/mirror-equivalent inputs together, including opposite labels.
        count = len(values)
        held = max(1, round(count * 0.1)) if count >= 3 else 0
        for position, indices in enumerate(values):
            split = 'dev_frame' if position < held else 'test_img' if position < 2 * held else 'train'
            for index in indices:
                result[index] = split
    return result
