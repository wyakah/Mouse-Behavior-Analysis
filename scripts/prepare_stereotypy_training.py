"""Build a deterministic, source-grouped training manifest from acquired data."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pandas as pd
from stereotypy.training import CLASSES, RARE, save_json, sha256, validate_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'research/stereotypy-external')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/stereotypy-training/v1')
    args = parser.parse_args()
    rng = np.random.default_rng(20260905)
    sources, windows = {}, []

    def source(path, dataset, crop=None, **extra):
        cap = cv2.VideoCapture(str(path))
        width, height = int(cap.get(3)), int(cap.get(4))
        fps, frames = cap.get(5), int(cap.get(7))
        cap.release()
        if not fps > 0 or width < 1:
            raise ValueError(f'Cannot inspect {path}')
        key = f'{dataset}:{path.stem}'
        sources[key] = dict(path=str(path.resolve()), dataset=dataset, sha256=sha256(path),
                            width=width, height=height, fps=fps, frames_header=frames,
                            crop=crop or [0, 0, width, int(height * .95)], **extra)
        return key, fps, frames

    def add(key, group, split, start, end, labels, authority, **extra):
        windows.append(dict(id=f'{key}:{len(windows):06d}', source=key, group=group, split=split,
                            start_s=round(float(start), 9), end_s=round(float(end), 9),
                            labels=labels, authority=authority, **extra))

    # MIT bounds are one-based inclusive frame numbers. Restrict windows to a
    # single annotator-1 interval, with a 0.2 s margin at both boundaries.
    for path in sorted((args.data / 'mit/full_database').glob('*.mpg')):
        key, fps, _ = source(path, 'mit', [0, 35, 320, 222])
        group = 'mit:' + (path.stem[:8] if path.stem.startswith('200') else 'agouti20080229')
        split = ('test' if group[-8:] in ('20080424', '20080428') else
                 'validation' if group[-8:] in ('20080422', '20080229') else 'train')
        label_path = path.parent / 'Annotator_group_1' / (path.stem + '.txt')
        sources[key].update(annotation_path=str(label_path), annotation_sha256=sha256(label_path))
        candidates = {}
        for line in label_path.read_text().splitlines():
            match = re.fullmatch(r'frame:\s+(\d+)-(\d+)\s+(\w+)', line.strip())
            if not match:
                raise ValueError(f'Unrecognized MIT annotation {line!r}')
            first, last, label = match.groups()
            a, b = (int(first)-1)/fps+.2, int(last)/fps-.2
            if b-a < 2:
                continue
            # At most three separated windows from a long interval.
            for start in np.linspace(a, b-2, min(3, max(1, int((b-a)/2)))):
                candidates.setdefault(label, []).append(float(start))
        for label, starts in sorted(candidates.items()):
            selected = rng.choice(len(starts), min(24, len(starts)), replace=False)
            for i in sorted(selected):
                labels = [-1]*6
                labels[0], labels[3] = int(label == 'groom'), int(label == 'rear')
                authority = 'MIT annotator group 1; exclusive source ethogram'
                if label == 'rest':
                    # Explicit, auditable semantic negatives, NOT source six-class labels.
                    for k in RARE:
                        labels[k] = 0
                    authority += '; rare-class negatives derived from source rest label'
                add(key, group, split, starts[i], starts[i]+2, labels, authority, source_behavior=label)

    # OSF targets_inserted contains human labels; machine_results is NOT used.
    for path in sorted((args.data / 'osf').glob('*.mp4')):
        annotation = path.with_suffix('.csv')
        if not annotation.exists():
            continue
        key, fps, frames = source(path, 'osf')
        df = pd.read_csv(annotation)
        if len(df) != frames:
            raise ValueError(f'OSF video/label frame count differs: {path}')
        columns = [c for c in ('digging', 'diggingging') if c in df]
        if len(columns) != 1 or not df[columns[0]].isin([0, 1]).all():
            raise ValueError('Unrecognized digging labels')
        sources[key].update(annotation_path=str(annotation), annotation_sha256=sha256(annotation),
                            annotation_column=columns[0])
        group = 'osf:' + path.stem.split('_')[0]
        split = 'test' if group in ('osf:Mother17', 'osf:Mother19') else 'validation' if group == 'osf:Mother13' else 'train'
        y = df[columns[0]].to_numpy()
        for first in range(0, len(y)-round(2*fps)+1, round(2*fps)):
            selected = y[first:first+round(2*fps)]
            if not np.all(selected == selected[0]):
                continue  # transitions are not given a made-up majority label
            labels = [-1]*6; labels[1] = int(selected[0])
            add(key, group, split, first/fps, (first+len(selected))/fps, labels,
                'OSF targets_inserted human digging annotation')

    bags = {}
    for behavior in ('gnawing_nonfood', 'jumping', 'circling'):
        path = args.data / 'stanford' / (behavior+'.mp4')
        key, fps, frames = source(path, 'stanford')
        sources[key]['crop'] = [0, 0, sources[key]['width'], int(sources[key]['height']*.86)]
        group = 'stanford:' + behavior
        bags[group] = behavior
        for start in np.arange(0, frames/fps-2, 2):
            add(key, group, 'train', start, start+2, [-1]*6,
                'Stanford ethogram demonstration: positive VIDEO bag only', bag=group)

    for name, crop in [('745_stereotypy.MOV', [1200,245,3120,1090]),
                       ('729_stereotypy.mov', [300,340,3210,1590])]:
        path = ROOT / name
        key, fps, frames = source(path, 'local', crop)
        # Exact local PTS/time base is taken from the existing source index.
        index_path = ROOT / 'outputs/stereotypy-pilot' / name[:3] / 'source-index.json'
        sources[key]['source_index'] = str(index_path)
        for start in np.arange(0, frames/fps-2, 2):
            add(key, 'local:'+name[:3], 'adaptation', start, start+2, [-1]*6,
                'Unlabeled user recording; reconstruction loss only')
    manifest = dict(version=1, seed=20260905, classes=list(CLASSES), window_s=2,
                    samples_per_window=16, sources=sources, windows=windows, positive_bags=bags,
                    status='experimental_not_for_scored_totals',
                    split_notes='MIT grouped by recording date; animal identity unavailable. OSF grouped by mother ID. Local videos are adaptation data, not validation.',
                    data_terms='Local research copies only. MIT page links a research-only software agreement; data redistribution terms not separately established. OSF node license null. Stanford media terms not established. Do not bundle data or learned weights in the public repository.')
    validate_manifest(manifest)
    save_json(args.output / 'dataset.json', manifest)
    print(json.dumps(dict(sources=len(sources), windows=len(windows), splits={s:sum(r['split']==s for r in windows) for s in ['train','validation','test','adaptation']}), indent=2))


if __name__ == '__main__':
    main()
