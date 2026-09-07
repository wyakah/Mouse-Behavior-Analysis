"""Software correctness only: synthetic recordings cannot validate recognition."""
import csv
from io import BytesIO, StringIO
import json
import zipfile

import pytest
from flask import Flask, jsonify

from stereotypy.core import measure, validate_annotations
from stereotypy.api import register_stereotypy
from stereotypy.video import index_video
from test_flow import tiny_video


def row(a, b, label="present", behavior="grooming"):
    return dict(start_s=a, end_s=b, label=label, behavior=behavior)


def grooming(rows, end, gaps=(), merge=0):
    summary, bouts = measure(rows, 0, end, gaps, merge)
    return summary[0], [b for b in bouts if b["behavior"] == "grooming"]


def test_250_frames_at_25_fps_are_ten_seconds():
    summary, bouts = grooming([row(i / 25, (i+1) / 25) for i in range(250)], 10)
    assert summary["active_seconds"] == pytest.approx(10)
    assert summary["coverage_percent"] == 100
    assert len(bouts) == 1
    assert bouts[0]["left_censored"] and bouts[0]["right_censored"]


def test_variable_duration_intervals_use_elapsed_time():
    summary, _ = grooming([row(0, .03), row(.03, .10, "absent"), row(.10, .31)], .31)
    assert summary["active_seconds"] == pytest.approx(.24)
    assert summary["scored_seconds"] == pytest.approx(.31)


def test_known_absence_and_unknown_are_different():
    summaries, _ = measure([row(0, 10, "absent")], 0, 10)
    assert summaries[0]["active_seconds"] == 0
    assert summaries[0]["observed_segments"] == 0
    assert summaries[0]["review_status"] == "complete"
    for untouched in summaries[1:]:
        assert untouched["active_seconds"] is None
        assert untouched["observed_segments"] is None
        assert untouched["coverage_percent"] == 0
        assert untouched["review_status"] == "unreviewed"
    summary, _ = grooming([row(0, 10, "unobservable")], 10)
    assert summary["active_seconds"] is None
    assert summary["unknown_seconds"] == 10


def test_merge_negative_pause_preserves_active_and_span_times():
    summary, bouts = grooming([row(0, 2), row(2, 2.2, "absent"), row(2.2, 4.2)], 4.2, merge=.2)
    assert summary["active_seconds"] == pytest.approx(4)
    assert len(bouts) == 1
    assert bouts[0]["span_seconds"] == 4.2
    assert bouts[0]["active_seconds"] == 4


@pytest.mark.parametrize("label", ["unreviewed", "ambiguous", "unobservable"])
def test_unknown_pause_splits_and_censors_even_when_merging(label):
    _, bouts = grooming([row(0, 2), row(2, 2.2, label), row(2.2, 4.2)], 4.2, merge=10)
    assert len(bouts) == 2
    assert bouts[0]["right_censored"] and bouts[1]["left_censored"]


def test_source_gap_overrides_positive_annotation():
    summary, bouts = grooming([row(0, 4.2)], 4.2, gaps=[(2, 2.2)], merge=10)
    assert summary["active_seconds"] == 4
    assert summary["source_gap_seconds"] == pytest.approx(.2)
    assert len(bouts) == 2


def test_observed_onsets_and_offsets_are_not_censored():
    _, bouts = grooming([row(0, 1, "absent"), row(1, 2), row(2, 3, "absent")], 3)
    assert not bouts[0]["left_censored"] and not bouts[0]["right_censored"]


def test_labels_can_overlap_across_behaviors_only():
    assert len(validate_annotations([row(0, 2), row(0, 2, behavior="rearing")], 0, 2)) == 2
    with pytest.raises(ValueError, match="overlap"):
        validate_annotations([row(0, 2), row(1, 2)], 0, 2)


@pytest.mark.parametrize("bad", [row(-1, 1), row(0, 0), row(0, 3), row(0, float("nan")), row(0, float("inf")), row(0, 1, "guessed"), row(0, 1, behavior="sniffing")])
def test_reject_invalid_annotation(bad):
    with pytest.raises(ValueError):
        validate_annotations([bad], 0, 2)


def test_index_strict_decoding_and_source_identity(tmp_path):
    video = tmp_path / "tiny.mp4"
    tiny_video(video)
    info = index_video(video)
    assert info["frame_count"] == 4
    assert info["duration_s"] == pytest.approx(.4)
    assert info["frames"][1]["start_s"] == pytest.approx(.1)
    assert info["source_gaps"] == []
    assert len(info["source_sha256"]) == 64
    assert info["variable_frame_rate"] is False


def test_real_encoded_nonzero_pts_and_gap_are_preserved(tmp_path):
    from fractions import Fraction
    import av
    import numpy as np
    video = tmp_path / "irregular.mp4"
    with av.open(str(video), "w") as out:
        stream = out.add_stream("libx264", rate=25)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        for pts in (1000, 1040, 1120, 1160):
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    info = index_video(video)
    assert info["source_origin_s"] == 1
    assert [r["start_s"] for r in info["frames"]] == pytest.approx([0, .04, .12, .16])
    assert len(info["source_gaps"]) == 1
    assert info["source_gaps"][0] == pytest.approx([.08, .12])
    summary, _ = grooming([row(0, info["duration_s"])], info["duration_s"], gaps=info["source_gaps"])
    assert summary["unknown_seconds"] == pytest.approx(.04)
    assert summary["active_seconds"] == pytest.approx(sum(f["end_s"]-f["start_s"] for f in info["frames"]))


class ImmediatePool:
    def submit(self, work):
        work()


@pytest.fixture
def client(tmp_path):
    tiny_video(tmp_path / "tiny.mp4")
    app = Flask(__name__)
    app.testing = True
    app.register_error_handler(ValueError, lambda e: (jsonify(error=str(e)), 400))
    register_stereotypy(app, lambda: tmp_path, ImmediatePool(), {})
    with app.test_client() as client:
        yield client, tmp_path


def create(client):
    response = client.post("/api/stereotypy/sessions", json=dict(video="tiny.mp4", animal_id="A", session_id="day1", apparatus_id="side1", view_confirmed=True))
    assert response.status_code == 202
    assert response.json["status"] == "complete", response.json
    return "/api/stereotypy/sessions/" + response.json["session_id"]


def test_revision_export_and_exact_frame_end_to_end(client):
    c, root = client
    url = create(c)
    initial = c.get(url).json
    assert initial["detector_status"] == "model_required"
    assert initial["summary"][0]["active_seconds"] is None
    assert c.get(url + "/frame/2").mimetype == "image/jpeg"
    assert c.get(url + "/frame/4").status_code == 400
    payload = dict(revision=0, annotator_id="WW", start_s=0, end_s=.4, merge_gap_s=0,
                   annotations=[row(0, .2), row(.2, .4, "absent")])
    saved = c.post(url, json=payload)
    assert saved.status_code == 200
    assert saved.json["summary"][0]["active_seconds"] == pytest.approx(.2)
    assert saved.json["revision"] == 1
    assert c.post(url, json=payload).status_code == 409
    first_export = c.post(url + "/export")
    assert first_export.status_code == 200
    payload.update(revision=1, annotations=[row(0, .4, "absent")])
    assert c.post(url, json=payload).json["summary"][0]["active_seconds"] == 0
    second_export = c.post(url + "/export")
    with zipfile.ZipFile(BytesIO(first_export.data)) as archive:
        summary = list(csv.DictReader(StringIO(archive.read("summary.csv").decode())))
        assert float(summary[0]["active_seconds"]) == pytest.approx(.2)
        assert summary[1]["active_seconds"] == ""
        manifest = json.loads(archive.read("run_manifest.json"))
        assert manifest["predictions_available"] is False
        assert set(manifest["implementation_sha256"]) == {"core.py", "video.py", "api.py", "cage.py", "pilot.py", "window.py"}
        assert manifest["package_versions"]["av"]
        assert len(manifest["session"]["revisions"]) == 2
    with zipfile.ZipFile(BytesIO(second_export.data)) as archive:
        manifest = json.loads(archive.read("run_manifest.json"))
        assert len(manifest["session"]["revisions"]) == 3
        assert manifest["session"]["revisions"][1]["annotations"][0]["label"] == "present"
    assert len(list((root / "outputs/stereotypy").rglob("*.zip"))) == 2
    assert c.get("/api/stereotypy/sessions").json[0]["animal_id"] == "A"


def test_path_view_and_source_change_protection(client):
    c, root = client
    args = dict(video="../outside.mp4", animal_id="A", session_id="D", apparatus_id="C", view_confirmed=True)
    assert c.post("/api/stereotypy/sessions", json=args).status_code == 400
    args.update(video="tiny.mp4", view_confirmed=False)
    assert c.post("/api/stereotypy/sessions", json=args).status_code == 400
    url = create(c)
    (root / "tiny.mp4").write_bytes(b"changed")
    assert c.get(url).status_code == 400
    assert c.post(url + "/export").status_code == 400


def test_window_cannot_silently_discard_annotations(client):
    c, _ = client
    url = create(c)
    response = c.post(url, json=dict(revision=0, annotator_id="WW", start_s=.1, end_s=.4, annotations=[row(0, .2)]))
    assert response.status_code == 400
    assert c.get(url).json["revision"] == 0


def test_sessions_enforce_first_twenty_minutes(client, monkeypatch):
    import stereotypy.api as api_module
    original = api_module.index_video
    def long_source(path):
        return dict(original(path), duration_s=1250)
    monkeypatch.setattr(api_module, 'index_video', long_source)
    c, _ = client
    url = create(c)
    record = c.get(url).json
    assert record['start_s'] == 0 and record['end_s'] == 1200
    assert c.post(url, json=dict(revision=0, annotator_id='reviewer', start_s=0,
                                 end_s=1250, annotations=[])).status_code == 400
    assert c.post(url, json=dict(revision=0, annotator_id='reviewer', start_s=10,
                                 end_s=1200, annotations=[])).status_code == 400


def test_queue_limit_metadata_defaults_and_export(client):
    c,root=client
    for i in range(150):(root/f'clip{i}.mp4').write_bytes((root/'tiny.mp4').read_bytes())
    rows=[dict(id=f'M{i}',video=f'clip{i}.mp4',sex='female',genotype='WT') for i in range(150)]
    assert c.post('/api/stereotypy/draft',json={'entries':rows}).status_code==200
    assert len(c.get('/api/stereotypy/draft').json['entries'])==150
    assert c.post('/api/stereotypy/draft',json={'entries':rows+[rows[0]]}).status_code==400
    assert c.post('/api/stereotypy/setup',json={'entries':rows}).status_code==200
    response=c.post('/api/stereotypy/sessions',json=dict(video='tiny.mp4',animal_id='M01',sex='female',genotype='WT',view_confirmed=True))
    assert response.status_code==202
    sid=response.json['session_id'];url='/api/stereotypy/sessions/'+sid
    record=c.get(url).json
    assert record['sex']=='female' and record['genotype']=='WT'
    assert record['apparatus_id']=='Not recorded'
    assert c.post(url+'/metadata',json={'animal_id':'M01','sex':'female','genotype':'KO'}).status_code==200
    record=c.get(url).json
    assert record['genotype']=='KO' and record['revision']==0
    assert [r['genotype'] for r in record['metadata_history']]==['WT','KO']
    with zipfile.ZipFile(BytesIO(c.post(url+'/export').data)) as archive:
        row=next(csv.DictReader(StringIO(archive.read('summary.csv').decode())))
        assert row['animal_id']=='M01' and row['sex']=='female' and row['genotype']=='KO'
    assert c.post('/api/stereotypy/sessions',json=dict(video='tiny.mp4',animal_id='A',sex='invalid',view_confirmed=True)).status_code==400
