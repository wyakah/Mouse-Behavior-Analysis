"""Strict source timing index, using declared frame durations to expose gaps."""
import av
from threechamber.core import sha256


def index_video(path):
    signature = path.stat()
    digest = sha256(path)
    frames, gaps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        info = dict(width=stream.width, height=stream.height, time_base=str(stream.time_base),
                    nominal_fps=float(stream.average_rate) if stream.average_rate else None,
                    codec=stream.codec_context.name, audio_present=bool(container.streams.audio))
        origin = None
        try:
            for frame in container.decode(stream):
                if frame.pts is None or not frame.duration or frame.duration <= 0:
                    raise ValueError("Source lacks exact frame timestamps/durations. Timing repair and verification are required before scoring.")
                pts = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = pts
                    info["rotation_degrees"] = int(frame.rotation)
                start = pts - origin
                end = start + float(frame.duration * frame.time_base)
                if frames:
                    previous = frames[-1]
                    if start <= previous["start_s"] or start < previous["end_s"] - 1e-7:
                        raise ValueError("Source frame intervals overlap or timestamps are not increasing.")
                    if start > previous["end_s"] + 1e-7:
                        gaps.append([previous["end_s"], start])
                frames.append(dict(frame_id=len(frames), start_s=start, end_s=end, pts=frame.pts))
        except av.error.FFmpegError as exc:
            raise ValueError("Source decoding failed. No partial scoring session was created.") from exc
    if not frames:
        raise ValueError("Video has no decoded frames.")
    if (signature.st_size, signature.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError("Source changed during indexing. Import a stable recording.")
    durations = {round(r["end_s"] - r["start_s"], 7) for r in frames}
    info.update(source_sha256=digest, source_size=signature.st_size,
                source_mtime_ns=signature.st_mtime_ns, source_origin_s=origin,
                duration_s=frames[-1]["end_s"], frame_count=len(frames),
                variable_frame_rate=len(durations) > 1 or bool(gaps),
                frames=frames, source_gaps=gaps, timing_version="declared-frame-duration-v1")
    return info
