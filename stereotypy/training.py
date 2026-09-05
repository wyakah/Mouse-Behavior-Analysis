"""Experimental video-learning primitives; never writes scored annotations.

Optional torch dependencies live in the training environment, not the web app.
Unknown labels are -1. A positive video bag does not label its member windows.
"""
import hashlib
import json
import math
from pathlib import Path

CLASSES = ('grooming', 'digging', 'gnawing_nonfood', 'rearing', 'jumping', 'circling')
RARE = (2, 4, 5)
STRONG = (0, 1, 3)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest):
    """Fail before fitting on leaks, invalid windows, or fabricated bag labels."""
    groups, ids, source_splits = {}, set(), {}
    for row in manifest['windows']:
        if row['id'] in ids:
            raise ValueError('Duplicate window ID')
        ids.add(row['id'])
        split = row['split']
        if split not in ('train', 'validation', 'test', 'adaptation'):
            raise ValueError('Invalid split')
        group = row['group']
        if group in groups and groups[group] != split:
            raise ValueError('A recording group crosses dataset splits')
        groups[group] = split
        source = manifest['sources'][row['source']]
        identity = source['sha256']
        if identity in source_splits and source_splits[identity] != split:
            raise ValueError('A source recording crosses dataset splits')
        source_splits[identity] = split
        if not all(math.isfinite(row[k]) for k in ('start_s','end_s')) or not 0 <= row['start_s'] < row['end_s']:
            raise ValueError('Invalid time interval')
        if len(row['labels']) != len(CLASSES) or any(x not in (-1, 0, 1) for x in row['labels']):
            raise ValueError('Invalid masked labels')
        if (split == 'adaptation' or row.get('bag')) and any(x != -1 for x in row['labels']):
            raise ValueError('Unlabeled adaptation / positive bags cannot supply window labels')
        if row.get('bag') and row['bag'] not in manifest['positive_bags']:
            raise ValueError('Window refers to an undeclared bag')
    for bag, label in manifest['positive_bags'].items():
        members = [r for r in manifest['windows'] if r.get('bag') == bag]
        if not members or label not in CLASSES or any(r['split'] != 'train' for r in members):
            raise ValueError('Invalid weak training bag')
    return manifest


def fitting_statistics(features, windows):
    """External held-out videos never influence normalization."""
    import numpy as np
    indices = [i for i,r in enumerate(windows) if r['split'] in ('train','adaptation')]
    if not indices:
        raise ValueError('No training or adaptation data')
    values = features[indices]
    return values.mean(axis=(0,1)), np.maximum(values.std(axis=(0,1)), .1)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_model(input_dim=960):
    import torch
    from torch import nn

    class TemporalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
            self.temporal = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
                                          nn.Conv1d(128, 128, 3, padding=1), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(.3), nn.Linear(256, len(CLASSES)))
            self.reconstruct = nn.Linear(128, input_dim)

        def forward(self, x):
            z = self.project(x)
            t = self.temporal(z.transpose(1, 2))
            logits = self.head(torch.cat((t.mean(2), t.amax(2)), 1))
            return logits, self.reconstruct(z)

    return TemporalModel()


def masked_loss(logits, labels, positive_weight, column_weight=None):
    import torch
    mask = labels >= 0
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels.clamp(min=0), pos_weight=positive_weight, reduction='none')
    if column_weight is not None:
        losses = losses * column_weight
    return (losses * mask).sum() / mask.sum().clamp(min=1)


def positive_bag_loss(logits, class_index):
    """At least one member can be positive; no per-window pseudo-ground-truth."""
    import torch
    return torch.nn.functional.softplus(-logits[:, class_index].amax())
