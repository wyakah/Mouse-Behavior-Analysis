"""Evaluate reviewed landmarks, counting missing rows and rejected predictions as misses."""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(successes, total):
    """Descriptive 95% binomial interval; video frames are not independent trials."""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [max(0., center-half), min(1., center+half)]


def summarize(rows, part, source=None):
    group = [r for r in rows if r['part'] == part and (source is None or r['source'] == source)]
    if not group:
        return None
    errors = [r['error_px'] for r in group if r['valid']]
    successes = sum(r['within_tolerance'] for r in group)
    reviewed_identity = [r for r in group if r['subject_correct'] is not None]
    result = dict(part=part, visible_labeled_frames=len(group), accepted_predictions=len(errors),
        coverage=len(errors)/len(group), missed_visible_frames=len(group)-len(errors),
        missing_prediction_rows=sum(not r['prediction_present'] for r in group),
        median_error_px=None if not errors else float(np.median(errors)),
        p95_error_px=None if not errors else float(np.percentile(errors, 95)),
        fraction_within_tolerance_including_misses=successes/len(group),
        correct_within_tolerance=successes,
        within_tolerance_wilson_95=wilson_interval(successes, len(group)),
        identity_reviewed_frames=len(reviewed_identity),
        wrong_subject_frames=sum(r['subject_correct'] is False for r in group),
        target_95_observed=successes/len(group) >= .95,
        target_99_observed=successes/len(group) >= .99,
        accuracy_target_validated=False)
    if source is not None:
        result['source'] = source
    return result


def evaluate(tracks, queue, annotations, cutoff=.6, tolerance_px=5):
    if not np.isfinite([cutoff, tolerance_px]).all() or not 0 <= cutoff <= 1 or tolerance_px <= 0:
        raise ValueError('Invalid cutoff or tolerance.')
    if not {'source', 'source_frame'}.issubset(tracks.columns):
        raise ValueError('Tracks require source and source_frame columns.')
    if tracks.duplicated(['source', 'source_frame']).any():
        raise ValueError('Duplicate source frame predictions; choose one run.')
    predictions = {(r['source'], int(r['source_frame'])): r for r in tracks.to_dict('records')}
    rows, seen = [], set()
    for e in queue:
        ann = annotations.get(e['id'], {})
        if ann.get('reviewed') is not True:
            continue
        key = (e['source'], int(e['source_frame']))
        if key in seen:
            raise ValueError('Duplicate reviewed source frame in queue.')
        seen.add(key)
        if ann.get('source', key[0]) != key[0] or ann.get('source_frame', key[1]) != key[1]:
            raise ValueError('Annotation provenance does not match its source frame.')
        r = predictions.get(key)
        # Optional human assessment of this prediction run, never inferred from confidence.
        review = ann.get('prediction_review', {})
        identity = review.get('subject_correct') if review.get('reviewed') is True else None
        if identity is not None and not isinstance(identity, bool):
            raise ValueError('Human subject_correct must be true, false, or absent.')
        for part in ['nose', 'center']:
            gt = ann.get('points', {}).get(part)
            if gt is None:
                continue
            if np.asarray(gt).shape != (2,) or not np.isfinite(gt).all():
                raise ValueError('Reviewed landmark must contain two finite coordinates.')
            values = [np.nan]*3 if r is None else [r.get(part+'_'+c, np.nan) for c in ['x', 'y', 'likelihood']]
            valid = bool(np.isfinite(values).all() and values[2] >= cutoff and values[2] <= 1)
            error = float(np.linalg.norm(np.asarray(values[:2])-gt)) if valid else None
            rows.append(dict(source=key[0], source_frame=key[1], part=part,
                prediction_present=r is not None, valid=valid, error_px=error,
                subject_correct=identity,
                within_tolerance=bool(valid and error <= tolerance_px and identity is not False)))
    if not rows:
        raise ValueError('No manually reviewed visible nose/center labels. Review frames first.')
    summary = [s for part in ['nose', 'center'] if (s := summarize(rows, part)) is not None]
    by_source = [s for source in sorted({r['source'] for r in rows}) for part in ['nose', 'center']
                 if (s := summarize(rows, part, source)) is not None]
    return {'cutoff': cutoff, 'tolerance_px': tolerance_px, 'summary': summary, 'frames': rows,
        'by_source': by_source, 'accuracy_validated': False,
        'targets': {'minimum': .95, 'stretch': .99, 'definition': 'Correct subject and visible landmark within chosen pixel tolerance, including misses.'},
        'interval_method': 'Two-sided 95% Wilson binomial interval; descriptive only. Video frames are correlated, not independent trials; these intervals cannot certify generalization.',
        'identity_note': 'Optional annotation prediction_review.reviewed=true and subject_correct=false counts as failure. Identity assessments must refer to the same prediction run.',
        'note': 'Errors conditional on accepted predictions; tolerance fraction includes missing rows and rejected predictions. Coverage alone is not accuracy. Pixels are not cm. This is development validation on inspected footage; final claims require an untouched animal/session test and a justified tolerance.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cutoff', type=float, default=.6); p.add_argument('--tolerance-px', type=float, default=5)
    p.add_argument('--subject', default='685', help='685 development validation; all audits all reviewed footage.')
    a = p.parse_args()
    path = ROOT/'labeling/annotations.json'
    if not path.exists(): p.error('No human labels yet. Open /labeling and review frames first.')
    tracks = pd.read_csv(a.tracks)
    queue = json.loads((ROOT/'labeling/queue.json').read_text())
    if a.subject != 'all':
        tracks = tracks[tracks.source.str.startswith(a.subject)]
        queue = [e for e in queue if e['source'].startswith(a.subject)]
    try:
        result = evaluate(tracks, queue, json.loads(path.read_text()), a.cutoff, a.tolerance_px)
    except ValueError as exc:
        p.error(str(exc))
    result['prediction_file'] = str(a.tracks.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['summary'], indent=2))

if __name__ == '__main__': main()
