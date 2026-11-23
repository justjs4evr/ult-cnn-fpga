"""Separate crossing fine-tuning with fixed baseline splits and rehearsal."""
import argparse
import copy
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from inference import CLASSES, load_model, preprocess_pil
from train_classified_images import score

ROOT = Path(__file__).resolve().parent
FLIP = torch.tensor([0, 2, 1, 4, 3, 5])
CROSSING = {'crossroads', 'crosswalk'}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def key(pixels):
    return hashlib.sha256(min(pixels.tobytes(), pixels[:, ::-1].tobytes())).hexdigest()


def evaluate(model, pixels, labels, rows, split, batch):
    idx = np.array([i for i, row in enumerate(rows) if row['split'] == split])
    x = torch.from_numpy(pixels[idx].astype(np.float32)[:, None] / 255)
    y = torch.tensor(labels[idx])
    metrics, pred = score(model, x, y, batch, CLASSES)
    crossing = np.array([j for j, i in enumerate(idx)
                         if rows[i]['source_class'] in CROSSING], dtype=np.int64)
    expected = labels[idx[crossing]]
    metrics['crossing'] = {
        'count': int(len(crossing)),
        'correct': int(np.sum(pred[crossing] == expected)),
        'recall': float(np.mean(pred[crossing] == expected)),
        'turn_errors': int(np.sum(np.isin(pred[crossing], [1, 2, 3, 4]))),
        'right_errors': int(np.sum(np.isin(pred[crossing], [2, 4]))),
        'predictions': dict(Counter(CLASSES[int(p)] for p in pred[crossing])),
    }
    directional = [metrics['f1'][c] for c in CLASSES[:5]]
    metrics['directional_macro_f1'] = float(np.mean(directional))
    return metrics, idx, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 4 or not 0 < args.learning_rate < 1:
        ap.error('invalid training settings')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(args.seed)
    baseline = args.baseline.resolve()
    checkpoint_path = baseline / 'best_model.pt' if baseline.is_dir() else baseline
    teacher, checkpoint = load_model(checkpoint_path)
    preprocessing = checkpoint['preprocessing']
    baseline_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    shutil.copy2(checkpoint_path, output / 'baseline_model.pt')
    manifest = checkpoint_path.parent / 'split_manifest.csv'
    shutil.copy2(manifest, output / 'original_split_manifest.csv')
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        original_rows = list(csv.DictReader(f))
    if any(row['split'] not in ('train', 'dev_frame', 'test_img') for row in original_rows):
        raise ValueError('Expected the original fixed baseline split')
    pixels = np.empty((len(original_rows), 16, 16), dtype=np.uint8)
    labels = np.array([CLASSES.index(row['training_class']) for row in original_rows])
    current_hashes = []
    for i, row in enumerate(original_rows):
        path = Path(row['path'])
        with Image.open(path) as image:
            pixels[i] = np.rint(preprocess_pil(image, preprocessing) * 255).astype(np.uint8)
        current_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if (i + 1) % 5000 == 0:
            print(f'Preprocessed {i + 1}/{len(original_rows)}', flush=True)
    train_idx = np.array([i for i, row in enumerate(original_rows) if row['split'] == 'train'])
    held_idx = np.array([i for i, row in enumerate(original_rows) if row['split'] != 'train'])
    crossing_idx = np.array([i for i in train_idx if original_rows[i]['source_class'] in CROSSING])
    held_keys = {key(pixels[i]) for i in held_idx}
    if any(key(pixels[i]) in held_keys for i in train_idx):
        raise ValueError('Baseline train/held-out exact or mirrored input leakage')
    if not len(crossing_idx):
        raise ValueError('No crossing training images')
    seen = {}
    for i in train_idx:
        p = pixels[i]; raw, mirrored = p.tobytes(), p[:, ::-1].tobytes()
        canonical_label = int(labels[i]) if raw <= mirrored else int(FLIP[labels[i]])
        digest = key(p)
        if digest in seen and seen[digest] != canonical_label:
            raise ValueError('Baseline mirrored input label conflict')
        seen[digest] = canonical_label
    generated_pixels, generated_rows = [], []
    rejected = Counter()
    # Photometric transforms precede thresholding; geometry and action stay fixed.
    for i in crossing_idx:
        row = original_rows[i]
        with Image.open(row['path']) as im:
            image = np.asarray(im.convert('RGB'), dtype=np.float32)
        for variant in range(2):
            brightness = float(rng.uniform(.85, 1.15))
            sigma = 2.0
            transformed = np.rint(np.clip(image * brightness +
                rng.normal(0, sigma, image.shape), 0, 255)).astype(np.uint8)
            im = Image.fromarray(transformed)
            p = np.rint(preprocess_pil(im, preprocessing) * 255).astype(np.uint8)
            digest = key(p)
            if digest in held_keys:
                rejected['matches_held_out'] += 1; continue
            if digest in seen:
                rejected['duplicate_input'] += 1; continue
            seen[digest] = 5
            destination = output / 'augmentation' / row['source_class'] / f'image_{i:06d}_{variant}.png'
            destination.parent.mkdir(parents=True, exist_ok=True)
            im.save(destination)
            generated_pixels.append(p)
            generated_rows.append({
                'path': str(destination), 'source_class': row['source_class'],
                'training_class': 'stop', 'split': 'train',
                'origin_path': row['path'], 'brightness': brightness,
                'noise_sigma': sigma, 'input_sha256': digest})
    rows = original_rows + [{k: r[k] for k in original_rows[0]} for r in generated_rows]
    with (output / 'split_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(original_rows[0])); w.writeheader(); w.writerows(rows)
    with (output / 'augmentation_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['path', 'source_class', 'training_class', 'split', 'origin_path',
                  'brightness', 'noise_sigma', 'input_sha256']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(generated_rows)
    with (output / 'original_image_hashes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['path', 'current_sha256', 'split'])
        w.writerows((r['path'], h, r['split']) for r, h in zip(original_rows, current_hashes))
    config = {
        'baseline': str(checkpoint_path), 'baseline_sha256': baseline_hash,
        'preprocessing': preprocessing, 'fixed_splits': dict(Counter(r['split'] for r in original_rows)),
        'original_crossing_train': len(crossing_idx), 'generated_augmentation': len(generated_rows),
        'rejected_augmentation': dict(rejected), 'epochs': args.epochs,
        'learning_rate': args.learning_rate, 'batch_size': args.batch_size, 'seed': args.seed,
        'batch_crossing_fraction': .25, 'rehearsal_fraction': .75,
        'loss': '.75 weighted replay CE + .25 crossing CE + .5 replay KL(T=2)',
        'selection': 'dev crossing recall, then fewer turn errors, then macro-F1; '
                     'dev macro-F1 and directional macro-F1 drop <= .005, accuracy drop <= .01',
        'new_collected_images': 0,
        'note': 'Existing reviewed crossing labels and train-only photometric variants; '
                'no screenshot crop or predicted-class pseudo labels used.'}
    dump(output / 'run_config.json', config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    model = copy.deepcopy(teacher)
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    teacher.eval()
    replay_x = torch.from_numpy(pixels[train_idx].astype(np.float32)[:, None] / 255)
    replay_y = torch.tensor(labels[train_idx])
    cross_pixels = pixels[crossing_idx]
    if generated_pixels:
        cross_pixels = np.concatenate([cross_pixels, np.stack(generated_pixels)])
    crossing_x = torch.from_numpy(cross_pixels.astype(np.float32)[:, None] / 255)
    replay_x = torch.cat([replay_x, replay_x.flip(3)])
    replay_y = torch.cat([replay_y, FLIP[replay_y]])
    crossing_x = torch.cat([crossing_x, crossing_x.flip(3)])
    counts = torch.bincount(replay_y, minlength=6)
    weights = len(replay_y) / (6 * counts.float())
    baseline_dev, _, _ = evaluate(teacher, pixels, labels, original_rows, 'dev_frame', args.batch_size)
    best_metrics = baseline_dev
    rank = lambda m: (m['crossing']['recall'], -m['crossing']['turn_errors'], m['macro_f1'], m['accuracy'])
    best_rank, best_epoch = rank(baseline_dev), 0
    best_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    crossing_batch = max(1, args.batch_size // 4)
    replay_batch = args.batch_size - crossing_batch
    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(replay_y)); total_loss = 0.; batches = 0
        for start in range(0, len(order), replay_batch):
            idx = order[start:start + replay_batch]
            rx, ry = replay_x[idx], replay_y[idx]
            cx = crossing_x[torch.randint(len(crossing_x), (crossing_batch,))]
            rlogits, clogits = model(rx), model(cx)
            with torch.no_grad(): tlogits = teacher(rx)
            kd = F.kl_div(F.log_softmax(rlogits / 2, dim=1),
                          F.softmax(tlogits / 2, dim=1), reduction='batchmean') * 4
            loss = (.75 * F.cross_entropy(rlogits, ry, weight=weights) +
                    .25 * F.cross_entropy(clogits, torch.full((len(cx),), 5, dtype=torch.long)) + .5 * kd)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); total_loss += float(loss.detach()); batches += 1
        dev, _, _ = evaluate(model, pixels, labels, original_rows, 'dev_frame', args.batch_size)
        acceptable = (dev['macro_f1'] >= baseline_dev['macro_f1'] - .005 and
                      dev['directional_macro_f1'] >= baseline_dev['directional_macro_f1'] - .005 and
                      dev['accuracy'] >= baseline_dev['accuracy'] - .01)
        selected = acceptable and rank(dev) > best_rank
        if selected:
            best_rank, best_epoch, best_metrics = rank(dev), epoch, dev
            best_state = copy.deepcopy(model.state_dict())
        entry = {'epoch': epoch, 'train_loss': total_loss / batches,
                 'development': dev, 'acceptable': acceptable, 'selected': selected}
        history.append(entry); dump(output / 'history.json', history)
        print(f'epoch {epoch:02d} loss={entry["train_loss"]:.4f} '
              f'dev={dev["accuracy"]:.4f} F1={dev["macro_f1"]:.4f} '
              f'cross={dev["crossing"]["correct"]}/{dev["crossing"]["count"]} '
              f'turns={dev["crossing"]["turn_errors"]} accepted={acceptable} best={best_epoch}', flush=True)
    model.load_state_dict(best_state); model.eval()
    new_checkpoint = {**checkpoint, 'model_state': best_state, 'epoch': best_epoch,
                      'init_checkpoint': str(checkpoint_path), 'training_kind': 'crossing_finetune',
                      'source_dataset': checkpoint.get('source_dataset'),
                      'split_manifest': str(output / 'split_manifest.csv'), 'run_config': config}
    torch.save(new_checkpoint, output / 'best_model.pt')
    # Test data is used only after the epoch has been selected using development data.
    baseline_test, _, _ = evaluate(teacher, pixels, labels, original_rows, 'test_img', args.batch_size)
    new_test, test_idx, test_pred = evaluate(model, pixels, labels, original_rows, 'test_img', args.batch_size)
    new_dev, dev_idx, dev_pred = evaluate(model, pixels, labels, original_rows, 'dev_frame', args.batch_size)
    comparison = {'baseline': {'development': baseline_dev, 'test': baseline_test},
                  'reinforced': {'development': new_dev, 'test': new_test},
                  'best_epoch': best_epoch, 'updated_weights': best_epoch != 0,
                  'config': config, 'history': history,
                  'limitations': ['No new manually collected crossing sequence added.',
                                  'Image metrics do not establish crossing driving success.',
                                  'Fixed historical splits do not prove independent maps or sessions.']}
    dump(output / 'comparison.json', comparison)
    dump(output / 'metrics.json', {'classes': CLASSES, 'development': new_dev, 'test': new_test,
                                 'best_epoch': best_epoch, 'preprocessing': preprocessing,
                                 'init_checkpoint': str(checkpoint_path), 'limitations': comparison['limitations']})
    with (output / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['path','split','source_class','true_class','predicted_class'])
        for split, indices, pred in [('dev_frame', dev_idx, dev_pred), ('test_img', test_idx, test_pred)]:
            w.writerows((original_rows[i]['path'], split, original_rows[i]['source_class'],
                         CLASSES[int(labels[i])], CLASSES[int(p)]) for i,p in zip(indices,pred))
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == baseline_hash
    dump(ROOT / 'latest_crossing_finetune.json', {'checkpoint': str(output / 'best_model.pt'),
                                                'output': str(output), 'baseline': str(checkpoint_path)})
    print(json.dumps({k:v for k,v in comparison.items() if k not in ('history','config')},
                     ensure_ascii=False, indent=2), flush=True)
    print('Output:', output, flush=True)


if __name__ == '__main__':
    main()
