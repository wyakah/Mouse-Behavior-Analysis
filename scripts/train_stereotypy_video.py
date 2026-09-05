"""Fit six experimental temporal heads, with masked and video-bag supervision.

Run using the DLC environment (torch/torchvision/sklearn). The image backbone is
frozen. Only the temporal encoder, reconstruction adapter and six heads are fitted.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TORCH_HOME'] = str(ROOT / '.cache/torch')
import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix
from stereotypy.training import (CLASSES, STRONG, RARE, make_model, masked_loss,
                                  positive_bag_loss, save_json, sha256, validate_manifest, fitting_statistics)

torch.set_num_threads(4)
IMPLEMENTATION = {str(p.relative_to(ROOT)):sha256(p) for p in
                  [Path(__file__), ROOT/'stereotypy/training.py']}


def input_image(frame, crop):
    x1, y1, x2, y2 = crop
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 224 / max(h, w)
    resized = cv2.resize(gray, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    result = np.full((224, 224), 127, np.uint8)
    a, b = (224-resized.shape[0])//2, (224-resized.shape[1])//2
    result[a:a+resized.shape[0], b:b+resized.shape[1]] = resized
    return np.repeat(result[None], 3, axis=0)


def extract(manifest, out, device):
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    net = mobilenet_v3_large(weights=weights).eval().to(device)
    net.classifier = torch.nn.Identity()
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    rows = manifest['windows']
    result = np.zeros((len(rows), 16, 960), np.float32)
    source_metadata = []
    for source_id, source in manifest['sources'].items():
        selected = [i for i, r in enumerate(rows) if r['source'] == source_id]
        # Cache is bound to both the source and the exact window/preprocessing recipe.
        cache_key = __import__('hashlib').sha256(json.dumps(dict(source=source, rows=[rows[i] for i in selected],
            recipe='gray-letterbox224-v1-mobilenet-v3-large-imagenet1k-v2'), sort_keys=True).encode()).hexdigest()
        cache = out/'features'/f'{cache_key}.npz'
        if cache.exists():
            data = np.load(cache, allow_pickle=False)
            result[selected] = data['features']
            source_metadata.append(json.loads(str(data['metadata'])))
            print('Cached', source_id, len(selected), flush=True)
            continue
        if sha256(source['path']) != source['sha256']:
            raise ValueError(f'Source changed: {source_id}')
        cap = cv2.VideoCapture(source['path'])
        fps = source['fps']
        frame_times = None
        if source.get('source_index'):
            index = json.loads(Path(source['source_index']).read_text())
            if index['source_sha256'] != source['sha256']:
                raise ValueError('Local source index hash mismatch')
            frame_times = np.array([f['start_s'] for f in index['frames']])
        requests = {}
        for i in selected:
            row = rows[i]
            times = np.linspace(row['start_s'], row['end_s'], 16, endpoint=False)
            indices = (np.searchsorted(frame_times, times, side='right')-1 if frame_times is not None
                       else np.floor(times*fps+1e-7).astype(int))
            for j, frame_id in enumerate(indices):
                requests.setdefault(int(frame_id), []).append((i, j))
        pending, destinations = [], []
        seen = set()

        def flush():
            if not pending:
                return
            x = torch.from_numpy(np.stack(pending)).to(device).float()/255
            with torch.inference_mode():
                features = net((x-mean)/std).cpu().numpy()
            for feature, targets in zip(features, destinations):
                for i, j in targets:
                    result[i, j] = feature
            pending.clear(); destinations.clear()

        frame_id = 0
        while cap.grab():
            if frame_id in requests:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f'Decode failed {source_id}:{frame_id}')
                pending.append(input_image(frame, source['crop']))
                destinations.append(requests[frame_id]); seen.add(frame_id)
                if len(pending) == 64:
                    flush()
            frame_id += 1
        cap.release(); flush()
        if set(requests) != seen:
            raise ValueError(f'Requested frames absent in {source_id}')
        if source['dataset'] == 'osf' and frame_id != source['frames_header']:
            raise ValueError('OSF decoded count does not match annotations')
        meta = dict(source=source_id, decoded_frames=frame_id, sampled_frames=len(seen), windows=len(selected),
                    index_policy='source PTS lookup' if frame_times is not None else 'sequential zero-based frame index',
                    cache_key=cache_key)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=result[selected], metadata=json.dumps(meta))
        source_metadata.append(meta)
        print('Extracted', source_id, len(selected), 'windows;', frame_id, 'frames decoded', flush=True)
    del net
    save_json(out/'extraction.json', dict(backbone=weights.name, backbone_frozen=True,
        preprocessing='grayscale replicated to RGB, aspect-preserving 224 letterbox, ImageNet normalization',
        samples_per_2s=16, sources=source_metadata))
    return result


def metrics(y, scores):
    result = {}
    for k in STRONG:
        mask = y[:, k] >= 0
        labels, p = y[mask, k], scores[mask, k]
        if len(np.unique(labels)) < 2:
            result[CLASSES[k]] = dict(status='insufficient_positive_or_negative_support', n=int(mask.sum()))
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(labels, p >= .5, average='binary', zero_division=0)
        result[CLASSES[k]] = dict(n=int(mask.sum()), positives=int(labels.sum()),
            average_precision=float(average_precision_score(labels, p)), precision=float(precision),
            recall=float(recall), f1=float(f1), threshold=.5, confusion_matrix=confusion_matrix(labels, p >= .5, labels=[0,1]).tolist())
    return result


def fit(features, manifest, out, epochs):
    # CPU for the small temporal network gives stable, reproducible local runs.
    torch.manual_seed(manifest['seed']); np.random.seed(manifest['seed'])
    rows = manifest['windows']
    y = np.array([r['labels'] for r in rows], np.float32)
    splits = np.array([r['split'] for r in rows])
    train = np.where((splits == 'train') & (y >= 0).any(1))[0]
    adaptation = np.where(splits == 'adaptation')[0]
    validation = np.where(splits == 'validation')[0]
    test = np.where(splits == 'test')[0]
    mean, scale = fitting_statistics(features, rows)
    x = torch.from_numpy(np.clip((features-mean)/scale, -10, 10))
    labels = torch.from_numpy(y)
    positive = np.maximum((y[train] == 1).sum(0), 1)
    negative = np.maximum((y[train] == 0).sum(0), 1)
    weight = torch.tensor(np.clip(negative/positive, .5, 12), dtype=torch.float32)
    col_weight = torch.ones(6); col_weight[list(RARE)] = 0
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0007, weight_decay=.01)
    history, best, best_ap = [], None, -1
    bag_indices = {b:np.array([i for i,r in enumerate(rows) if r.get('bag') == b]) for b in manifest['positive_bags']}
    rng = np.random.default_rng(manifest['seed'])
    for epoch in range(epochs):
        model.train(); losses=[]
        order = rng.permutation(train)
        for start in range(0, len(order), 64):
            ids = order[start:start+64]
            logits, reconstructed = model(x[ids] + torch.randn_like(x[ids])*.03)
            loss = masked_loss(logits, labels[ids], weight, col_weight)
            loss = loss + .02*torch.nn.functional.mse_loss(reconstructed, x[ids])
            # Both supplied mice affect learned adapter weights, without class labels.
            aids = rng.choice(adaptation, min(32, len(adaptation)), replace=False)
            _, local_reconstruction = model(x[aids] + torch.randn_like(x[aids])*.08)
            loss = loss + .08*torch.nn.functional.mse_loss(local_reconstruction, x[aids])
            # Whole-video bags are evaluated as whole bags; no frame labels are made.
            for bag, ids_bag in bag_indices.items():
                bag_logits, _ = model(x[ids_bag])
                k = CLASSES.index(manifest['positive_bags'][bag])
                negative_ids = train[y[train, k] == 0]
                negative_ids = rng.choice(negative_ids, min(32, len(negative_ids)), replace=False)
                negative_logits, _ = model(x[negative_ids])
                loss = loss + .03*(positive_bag_loss(bag_logits, k)
                    + torch.nn.functional.softplus(negative_logits[:, k]).mean())
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            probabilities = model(x[validation])[0].sigmoid().numpy()
        validation_metrics = metrics(y[validation], probabilities)
        ap = float(np.mean([m['average_precision'] for m in validation_metrics.values() if 'average_precision' in m]))
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), validation_macro_ap=ap))
        if ap > best_ap:
            best_ap, best = ap, copy.deepcopy(model.state_dict())
            best_epoch = epoch+1
        if epoch % 5 == 0 or epoch+1 == epochs:
            print('Epoch', epoch+1, 'loss', round(history[-1]['loss'],4), 'validation macro AP', round(ap,4), flush=True)
    model.load_state_dict(best); model.eval()
    with torch.inference_mode():
        all_scores = np.concatenate([model(batch)[0].sigmoid().numpy() for batch in x.split(128)])
    checkpoint = out/'temporal-model.pt'
    torch.save(dict(state_dict=best, mean=torch.from_numpy(mean), scale=torch.from_numpy(scale), classes=list(CLASSES),
                    dataset_sha256=sha256(out/'dataset.json'), trained_epoch=best_epoch,
                    production_ready=False, backbone='mobilenet_v3_large:IMAGENET1K_V2',
                    architecture='TemporalModel-v1', implementation_sha256=IMPLEMENTATION), checkpoint)
    report = dict(status='experimental_not_for_scored_totals', production_ready=False,
        best_epoch=best_epoch, epochs=epochs, history=history,
        checkpoint_sha256=sha256(checkpoint), dataset_sha256=sha256(out/'dataset.json'),
        implementation_sha256=IMPLEMENTATION,
        validation=metrics(y[validation], all_scores[validation]), test=metrics(y[test], all_scores[test]),
        local_accuracy=None, local_use='Both recordings used for unlabeled feature normalization and denoising reconstruction; not evaluation.',
        metrics_unit='Two-second windows entirely within a source label interval, not frame accuracy or bout accuracy.',
        weak_heads={CLASSES[k]:dict(status='trained_weak_video_bag_only', positive_recordings=1,
                    independent_positive_test_recordings=0, accuracy=None,
                    negatives='Derived from MIT rest labels; cross-dataset confounding is uncontrolled.') for k in RARE},
        split_notes=manifest['split_notes'],
        source_counts={s:len({r['source'] for r in rows if r['split']==s}) for s in set(splits)},
        class_window_support={CLASSES[k]:{s:dict(positive=int(((y[:,k]==1)&(splits==s)).sum()),
            negative=int(((y[:,k]==0)&(splits==s)).sum()), unknown=int(((y[:,k]<0)&(splits==s)).sum()))
            for s in ('train','validation','test','adaptation')} for k in range(6)})
    save_json(out/'training-report.json', report)
    np.savez_compressed(out/'window-predictions.npz', scores=all_scores, labels=y)
    for key, source in manifest['sources'].items():
        if source['dataset'] != 'local':
            continue
        selected = [i for i,r in enumerate(rows) if r['source']==key]
        predictions = [dict(start_s=rows[i]['start_s'], end_s=rows[i]['end_s'],
            scores={c:float(all_scores[i,k]) for k,c in enumerate(CLASSES)}) for i in selected]
        save_json(out/(Path(source['path']).stem+'-proposals.json'), dict(
            status='experimental_review_scores_not_behavior_annotations', source_sha256=source['sha256'],
            model_sha256=report['checkpoint_sha256'], scores_calibrated=False, behavior_totals=None,
            source_used_for_unlabeled_adaptation=True, windows=predictions))
    print(json.dumps(report['test'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/stereotypy-training/v1')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    manifest = validate_manifest(json.loads((args.run/'dataset.json').read_text()))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    features = extract(manifest, args.run, device)
    if not args.extract_only:
        fit(features, manifest, args.run, args.epochs)


if __name__ == '__main__':
    main()
