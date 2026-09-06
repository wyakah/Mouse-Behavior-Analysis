"""Local manual-scoring sessions with atomic revisions and immutable exports."""
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from threading import RLock
import csv
import json
import os
import re
import uuid
import zipfile
from importlib.metadata import version

from flask import jsonify, request, send_file, abort
from threechamber.core import sha256
from threechamber.statistics import sample_metadata
from threechamber.recordings import validate_recordings
from threechamber.batch import save_json
from .core import BEHAVIORS, LABELS, ETHOGRAM_VERSION, number, validate_annotations, measure
from .video import index_video
from .window import analysis_end
from .cage import validate_mapping, crop_frame
from .pilot import run_pilot

# Capture the implementation loaded by this process, even if files change later.
IMPLEMENTATION_HASHES = {name: sha256(Path(__file__).with_name(name))
                         for name in ("core.py", "video.py", "api.py", "cage.py", "pilot.py", "window.py")}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def csv_bytes(rows, fields):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        # Preserve free text without allowing spreadsheet formula evaluation.
        writer.writerow({k: ("'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v)
                         for k, v in row.items()})
    return stream.getvalue().encode("utf-8")


def register_stereotypy(app, root_getter, pool, jobs):
    lock = RLock()

    def root():
        return Path(root_getter()).resolve()

    def folder():
        return root() / "labeling" / "stereotypy"

    def source(name):
        if not isinstance(name, str):
            raise ValueError("Choose a local recording.")
        path = (root() / name).resolve()
        if not path.is_relative_to(root()) or not path.is_file() or path.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv", ".m4v"):
            raise ValueError("Choose a video inside this workspace.")
        return path

    def load(sid):
        if not re.fullmatch(r"[0-9a-f]{32}", sid):
            abort(404)
        path = folder() / f"{sid}.json"
        if not path.is_file():
            abort(404)
        record = json.loads(path.read_text())
        if record["ethogram_version"] == 'side-view-draft-1':
            # Add requested classes without relabeling or changing any prior interval.
            # Previous immutable exports retain their original four-class definitions.
            record.setdefault('ethogram_history',[]).append(dict(
                version=record['ethogram_version'],definitions=record.get('definitions',{}),
                upgraded_at=utc_now(),reason='Add jumping and circling; existing intervals unchanged.'))
            record['ethogram_version']=ETHOGRAM_VERSION
            record['definitions']=dict(BEHAVIORS)
            save(record)
        if record["ethogram_version"] != ETHOGRAM_VERSION:
            raise ValueError("This session uses a different ethogram version; an explicit migration is required.")
        return record

    def save(record):
        folder().mkdir(parents=True, exist_ok=True)
        path = folder() / f"{record['id']}.json"
        temp = path.with_suffix(".tmp")
        with temp.open("w") as handle:
            json.dump(record, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)

    def payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("Send a JSON object.")
        return data

    def identifiers(data):
        result = {}
        for key in ("animal_id", "session_id", "apparatus_id"):
            value = data.get(key)
            if key == "session_id" and not value: value = "session-" + uuid.uuid4().hex[:8]
            if key == "apparatus_id" and not value: value = "Not recorded"
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100:
                raise ValueError("Enter a mouse ID (1–100 characters).")
            result[key] = value.strip()
        return result

    def check_source(record, full_hash=False):
        path = source(record["video"])
        info, stat = record["video_manifest"], path.stat()
        if stat.st_size != info["source_size"] or stat.st_mtime_ns != info["source_mtime_ns"]:
            raise ValueError("Source changed since indexing. Create a new session for the changed recording.")
        if full_hash and sha256(path) != info["source_sha256"]:
            raise ValueError("Source content hash no longer matches this session.")
        return path

    def public(record):
        current = record["revisions"][-1]
        summary, bouts = measure(current["annotations"], record["start_s"], record["end_s"],
                                 record["video_manifest"]["source_gaps"], record["merge_gap_s"])
        return {k: v for k, v in record.items() if k != "revisions"} | dict(
            revision=current["revision"], annotations=current["annotations"], summary=summary,
            bouts=bouts, detector_status="model_required", ethogram=BEHAVIORS, labels=LABELS)

    @app.get("/stereotypy")
    def stereotypy_page():
        return app.send_static_file("stereotypy.html")

    @app.route("/api/stereotypy/draft", methods=["GET", "POST"])
    def recording_draft():
        path=root()/"batches/stereotypy-draft.json"
        if request.method=="POST":
            data=payload();validate_recordings(data.get("entries"));save_json(path,data)
            return jsonify(saved=True)
        return jsonify(json.loads(path.read_text()) if path.exists() else dict(entries=[]))

    @app.post("/api/stereotypy/setup")
    def validate_setup():
        data=payload();validate_recordings(data.get("entries"),require_ids=True)
        for entry in data['entries']:source(entry['video'])
        return jsonify(valid=True)

    @app.post("/api/stereotypy/sessions/<sid>/metadata")
    def update_metadata(sid):
        data=payload()
        with lock:
            record=load(sid);check_source(record)
            values=identifiers(dict(record,animal_id=data.get('animal_id'))) | sample_metadata(data)
            before={key:record.get(key) for key in values}
            if before!=values:
                history=record.setdefault('metadata_history',[dict(changed_at=record['created_at'],**before)])
                history.append(dict(changed_at=utc_now(),**values));record.update(values);save(record)
        return jsonify(saved=True)

    @app.get("/api/stereotypy/sessions")
    def sessions():
        records = []
        for path in sorted(folder().glob("*.json")):
            r = json.loads(path.read_text())
            records.append({k: r.get(k) for k in ("id", "animal_id", "sex", "genotype", "session_id", "apparatus_id", "video", "created_at")})
        return jsonify(records)

    @app.post("/api/stereotypy/sessions")
    def create_session():
        data = payload()
        ids = identifiers(data) | sample_metadata(data)
        path = source(data.get("video"))
        workspace = root()
        if data.get("view_confirmed") is not True:
            raise ValueError("Confirm that this is a single-mouse side-view recording.")
        sid, jobid = uuid.uuid4().hex, "stereotypy-" + uuid.uuid4().hex[:12]
        jobs[jobid] = dict(id=jobid, status="queued", message="Source timing inspection queued")

        def work():
            try:
                jobs[jobid].update(status="running", message="Checking source timestamps and decoding the recording")
                info = index_video(path)
                # Stereotypy uses the first twenty minutes of source time.
                end = analysis_end(info["duration_s"])
                record = dict(id=sid, video=str(path.relative_to(workspace)), **ids,
                              view="side", created_at=utc_now(), video_manifest=info,
                              metadata_history=[dict(changed_at=utc_now(), **ids)],
                              start_s=0, end_s=end, merge_gap_s=0,
                              ethogram_version=ETHOGRAM_VERSION, ethogram_status="draft",
                              definitions=dict(BEHAVIORS),
                              revisions=[dict(revision=0, created_at=utc_now(), annotator_id=None,
                                              annotations=[], start_s=0, end_s=end, merge_gap_s=0)])
                with lock:
                    save(record)
                jobs[jobid].update(status="complete", message="Manual scoring ready", session_id=sid)
            except Exception as exc:
                jobs[jobid].update(status="failed", message=str(exc))

        pool.submit(work)
        return jsonify(jobs[jobid]), 202

    @app.get("/api/stereotypy/sessions/<sid>")
    def get_session(sid):
        with lock:
            record = load(sid)
        check_source(record)
        return jsonify(public(record))

    @app.post("/api/stereotypy/sessions/<sid>")
    def update_session(sid):
        data = payload()
        with lock:
            record = load(sid)
            check_source(record)
            if data.get("revision") != record["revisions"][-1]["revision"]:
                return jsonify(error="This session changed in another tab. Reload before saving."), 409
            annotator = data.get("annotator_id")
            if not isinstance(annotator, str) or not annotator.strip() or len(annotator) > 100:
                raise ValueError("Enter an annotator ID before saving.")
            start, end = number(data.get("start_s")), number(data.get("end_s"))
            merge = number(data.get("merge_gap_s", 0))
            if start != 0 or abs(end - analysis_end(record["video_manifest"]["duration_s"])) > 1e-7 or merge < 0:
                raise ValueError("Stereotypy uses the first 20 minutes (or the whole source if shorter). Choose a nonnegative merge gap.")
            annotations = validate_annotations(data.get("annotations"), start, end)
            revision = dict(revision=data["revision"] + 1, created_at=utc_now(),
                            annotator_id=annotator.strip(), annotations=annotations,
                            start_s=start, end_s=end, merge_gap_s=merge)
            record.update(start_s=start, end_s=end, merge_gap_s=merge)
            record["revisions"].append(revision)
            save(record)
        return jsonify(public(record))

    @app.post("/api/stereotypy/sessions/<sid>/cage")
    def save_cage(sid):
        data=payload()
        with lock:
            record=load(sid);check_source(record)
            info=record['video_manifest']
            if info['rotation_degrees'] != 0:
                raise ValueError('Rotated footage needs orientation verification before cage mapping.')
            if data.get('revision') != record.get('cage_revision',0):
                return jsonify(error='Cage mapping changed in another tab. Reload before saving.'),409
            mapping=validate_mapping(data.get('mapping'),info['width'],info['height'])
            if mapping==record.get('cage_mapping'):
                return jsonify(mapping=mapping,revision=record['cage_revision'])
            record['cage_revision']=record.get('cage_revision',0)+1
            record['cage_mapping']=mapping
            record.setdefault('cage_history',[]).append(dict(revision=record['cage_revision'],changed_at=utc_now(),mapping=mapping))
            save(record)
        return jsonify(mapping=mapping,revision=record['cage_revision'])

    @app.post("/api/stereotypy/sessions/<sid>/cage/analyze")
    def analyze_cage(sid):
        with lock:
            record=load(sid);path=check_source(record)
            if not record.get('cage_mapping'):raise ValueError('Save the cage mapping first.')
            active=next((j for j in jobs.values() if j.get('cage_session')==sid and j['status'] in ('queued','running')),None)
            if active:return jsonify(active),202
            runid=uuid.uuid4().hex
            jobid='cage-'+runid[:12]
            jobs[jobid]=dict(id=jobid,status='queued',message='Cage review queued',cage_session=sid)
        destination=root()/'outputs/stereotypy'/sid/runid
        def work():
            try:
                jobs[jobid].update(status='running',message='Building cage background')
                result=run_pilot(path,record['cage_mapping'],destination,record['video_manifest'],
                    lambda n,total:jobs[jobid].update(message=f'Cage review: {n:,} / {total:,} frames',completed_frames=n,total_frames=total))
                run=dict(id=runid,created_at=utc_now(),cage_revision=record.get('cage_revision',0),
                         directory=str(destination.relative_to(root())),result=result)
                with lock:
                    latest=load(sid);latest.setdefault('cage_runs',[]).append(run);save(latest)
                jobs[jobid].update(status='complete',message='Cage review ready; behavior scores require review.',run=run)
            except Exception as exc:jobs[jobid].update(status='failed',message=str(exc))
        pool.submit(work)
        return jsonify(jobs[jobid]),202

    @app.get("/api/stereotypy/sessions/<sid>/cage/<runid>/<filename>")
    def cage_artifact(sid,runid,filename):
        if filename not in ('review.mp4','poster.jpg','background.jpg','manifest.json','frame-features.csv','review-windows.json'):
            abort(404)
        with lock:record=load(sid)
        check_source(record)
        run=next((r for r in record.get('cage_runs',[]) if r['id']==runid),None)
        if run is None:abort(404)
        path=(root()/run['directory']/filename).resolve()
        if not path.is_relative_to(root()/'outputs') or not path.is_file():abort(404)
        return send_file(path,conditional=True,as_attachment=request.args.get('download')=='1')

    @app.get("/api/stereotypy/sessions/<sid>/frame/<int:frame_id>")
    def exact_frame(sid, frame_id):
        import av
        with lock:
            record = load(sid)
        path = check_source(record)
        frames = record["video_manifest"]["frames"]
        if not 0 <= frame_id < len(frames):
            raise ValueError("Frame is outside the recording.")
        target = frames[frame_id]["pts"]
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(target, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts == target:
                    import cv2
                    image=frame.to_ndarray(format="bgr24")
                    if request.args.get('crop')=='cage' and record.get('cage_mapping'):
                        image=crop_frame(image,record['cage_mapping'])
                    ok, encoded = cv2.imencode(".jpg", image)
                    if not ok:
                        raise ValueError("Could not encode review frame.")
                    return send_file(BytesIO(encoded.tobytes()), mimetype="image/jpeg")
                if frame.pts is not None and frame.pts > target:
                    break
        raise ValueError("Exact source frame could not be decoded.")

    @app.post("/api/stereotypy/sessions/<sid>/export")
    def export_session(sid):
        with lock:
            record = load(sid)
        check_source(record, full_hash=True)
        current = record["revisions"][-1]
        summaries, bouts = measure(current["annotations"], record["start_s"], record["end_s"],
                                   record["video_manifest"]["source_gaps"], record["merge_gap_s"])
        run_id = uuid.uuid4().hex
        provenance = dict(video_id=sid, animal_id=record["animal_id"], session_id=record["session_id"],
                          sex=record.get("sex", "unknown"), genotype=record.get("genotype", ""),
                          annotation_revision=current["revision"], annotator_id=current["annotator_id"],
                          ethogram_version=record["ethogram_version"], run_id=run_id, model_id=None)
        summaries = [provenance | r for r in summaries]
        bouts = [provenance | r for r in bouts]
        annotations = [provenance | r for r in current["annotations"]]
        manifest = dict(schema_version="stereotypy-manual-1", run_id=run_id, created_at=utc_now(),
                        detector_status="model_required", result_source="manual", session=record,
                        definitions=record.get("definitions", BEHAVIORS), predictions_available=False,
                        implementation_sha256=IMPLEMENTATION_HASHES,
                        package_versions={name: version(name) for name in ("av", "Flask")},
                        timing_note="Original PTS normalized to first decoded frame; source gaps remain unknown.",
                        segmentation_note="No minimum bout filter; merge only explicitly absent gaps.")
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("summary.csv", csv_bytes(summaries, list(summaries[0])))
            bundle.writestr("bouts.csv", csv_bytes(bouts, list(provenance) + ["behavior", "start_s", "end_s", "active_seconds", "left_censored", "right_censored", "span_seconds"]))
            bundle.writestr("annotations.csv", csv_bytes(annotations, list(provenance) + ["behavior", "label", "start_s", "end_s", "note"]))
            bundle.writestr("run_manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        dest = root() / "outputs" / "stereotypy" / sid
        dest.mkdir(parents=True, exist_ok=True)
        output = dest / f"{run_id}.zip"
        output.write_bytes(archive.getvalue())
        return send_file(output, as_attachment=True, download_name=f"stereotypy-{sid[:8]}-r{current['revision']}.zip")
