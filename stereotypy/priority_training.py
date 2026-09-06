"""Scientific guards and temporal features for experimental priority classifiers."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

CLASSES = ('grooming', 'digging', 'rearing')


def validate_catalog(rows):
    groups, hashes, ids = {}, {}, set()
    for row in rows:
        if row['id'] in ids:
            raise ValueError('Duplicate source ID')
        ids.add(row['id'])
        split = row['split']
        if split not in ('train', 'validation', 'test', 'legacy_test', 'development'):
            raise ValueError('Unknown split')
        for key, mapping in ((row['group'], groups), (row['sha256'], hashes)):
            if key in mapping and mapping[key] != split:
                raise ValueError('Source or mouse group crosses splits')
            mapping[key] = split
        if row['dataset'] == 'labgym' and split != 'train':
            raise ValueError('LabGym parent recording IDs are unknown: training only')
        if row['dataset'] == 'local' and split != 'development':
            raise ValueError('Unlabeled local videos cannot validate behavior accuracy')
    return rows


from .temporal import context_indices


def binary_metrics(labels, scores, threshold):
    mask = np.asarray(labels) >= 0
    y, p = np.asarray(labels)[mask], np.asarray(scores)[mask]
    if len(np.unique(y)) < 2:
        return dict(n=int(len(y)), positives=int((y == 1).sum()), status='insufficient_support')
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    precision = tp/max(tp+fp, 1); recall = tp/max(tp+fn, 1)
    specificity = tn/max(tn+fp, 1)
    accuracy = (tp+tn)/len(y)
    return dict(n=int(len(y)), positives=int(y.sum()), threshold=float(threshold),
        accuracy=float(accuracy), balanced_accuracy=float((recall+specificity)/2),
        precision=float(precision), recall=float(recall), specificity=float(specificity),
        f1=float(2*precision*recall/max(precision+recall, 1e-12)),
        average_precision=float(average_precision_score(y, p)),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        target_met=bool(min(accuracy, (recall+specificity)/2, precision, recall) > .90))


def select_threshold(labels, scores, datasets):
    """Select on validation only, weighting camera datasets equally."""
    best = None
    for threshold in np.linspace(.02, .98, 97):
        metrics = {d:binary_metrics(labels[datasets == d], scores[datasets == d], threshold)
                   for d in sorted(set(datasets))}
        supported = [v for v in metrics.values() if 'f1' in v]
        value = float(np.mean([v['f1'] for v in supported])) if supported else -1
        if best is None or value > best['selection_score']:
            best = dict(threshold=float(threshold), selection_score=value, by_dataset=metrics)
    return best


def labgym_labels(name):
    """Map only explicit source semantics; foraging never supplies digging labels."""
    labels = [-1, -1, -1]
    if name.startswith(('face grooming', 'body grooming')):
        labels[0] = 1
    elif name.startswith(('foraging', 'walking', 'rearing', 'standing', 'sniffing', 'sleeping', 'running', 'chewing', 'nest building')):
        labels[0] = 0
    if name.startswith(('rearing up', 'standing')) and 'wheel' not in name:
        labels[2] = 1
    elif name.startswith(('foraging', 'walking', 'sleeping', 'crawling')):
        labels[2] = 0
    return labels
