"""Shared assay window: first twenty minutes, never extend a short source."""
import math
MAX_SECONDS = 1200.0


def analysis_end(duration):
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Source duration must be positive and finite.')
    return min(duration, MAX_SECONDS)


def clip_manifest(manifest):
    end = analysis_end(manifest['duration_s'])
    frames = [dict(row, end_s=min(row['end_s'], end))
              for row in manifest['frames'] if row['start_s'] < end]
    return dict(manifest, frames=frames, frame_count=len(frames), duration_s=end,
                original_duration_s=manifest['duration_s'],
                source_gaps=[[a,min(b,end)] for a,b in manifest['source_gaps'] if a < end],
                analysis_window_policy='first-1200-seconds-v1')
