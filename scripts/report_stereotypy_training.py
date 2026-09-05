"""Publish a local, read-only training report. Never accepts predictions as labels."""
import argparse
import csv
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stereotypy.training import CLASSES, STRONG, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/stereotypy-training/v1')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/stereotypy-training')
    args = parser.parse_args()
    manifest = json.loads((args.run/'dataset.json').read_text())
    report = json.loads((args.run/'training-report.json').read_text())
    sessions = {r['animal_id']:r for r in json.loads((ROOT/'reports/stereotypy-pilot/sessions.json').read_text())}
    args.output.mkdir(parents=True, exist_ok=True)
    videos = {}
    for animal in ('745','729'):
        path = args.run/(animal+'_stereotypy-proposals.json')
        proposal = json.loads(path.read_text())
        session = sessions[animal]
        source = json.loads((ROOT/'outputs/stereotypy-pilot'/animal/'source-index.json').read_text())
        if proposal['source_sha256'] != source['source_sha256']:
            raise ValueError('Review video and prediction source differ')
        videos[animal] = dict(**proposal, video=session['base']+'/review.mp4',
            poster=session['base']+'/poster.jpg', reviewer='/stereotypy?session='+session['session_id'],
            duration_s=source['duration_s'], source_gaps=[dict(start_s=g[0],end_s=g[1]) for g in source['source_gaps']])
        shutil.copy2(path, args.output/path.name)
        with (args.output/(animal+'-model-scores.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_id','start_s','end_s',*[c+'_uncalibrated_score' for c in CLASSES],'status'])
            for row in proposal['windows']:
                writer.writerow([animal,row['start_s'],row['end_s'],*[row['scores'][c] for c in CLASSES],
                                 'experimental_unreviewed_not_a_behavior_label'])
    shutil.copy2(args.run/'training-report.json', args.output/'training-report.json')
    cards = []
    names = dict(grooming='Grooming', digging='Digging', gnawing_nonfood='Nonfood gnawing',
                 rearing='Rearing', jumping='Jumping', circling='Circling')
    for k, behavior in enumerate(CLASSES):
        if k in STRONG:
            m = report['test'][behavior]
            detail = (f"<div class='number'>{m['f1']:.2f}<small>F1 at 0.5</small></div>"
                      f"<p>Precision {m['precision']:.2f} · recall {m['recall']:.2f}<br>"
                      f"{m['n']} test windows · {m['positives']} positive<br>Average precision {m['average_precision']:.2f}</p>")
            tag = 'External test measured'
        else:
            detail = "<div class='number quiet'>Not measured<small>independent accuracy</small></div><p>1 positive demonstration video<br>No independent positive test recording<br>Camera and background confounding unresolved</p>"
            tag = 'Weak supervision only'
        cards.append(f"<article class='metric'><span class='tag'>{tag}</span><h3>{names[behavior]}</h3>{detail}</article>")
    payload = json.dumps(dict(classes=list(CLASSES), names=names, videos=videos), allow_nan=False).replace('<','\\u003c')
    template = (ROOT/'static/stereotypy-training-report.html').read_text()
    page = template.replace('__CARDS__', ''.join(cards)).replace('__DATA__', payload)
    page = page.replace('__WINDOWS__', f"{len(manifest['windows']):,}")
    page = page.replace('__EPOCH__', str(report['best_epoch'])).replace('__SOURCES__', str(len(manifest['sources'])))
    (args.output/'index.html').write_text(page)
    print('http://127.0.0.1:8765/reports/stereotypy-training/index.html')


if __name__ == '__main__':
    main()
