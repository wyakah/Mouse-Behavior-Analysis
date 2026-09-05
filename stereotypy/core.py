"""Explicit interval labels on a source-video clock. No automatic detector."""
from collections import defaultdict
import math

ETHOGRAM_VERSION = "side-view-draft-1"
BEHAVIORS = {
    "grooming": "Self-directed paw washing, face/head sweeps, or licking/nibbling own fur. Exclude feeding, drinking, and isolated scratching.",
    "digging": "Repeated forepaw scraping of bedding with substrate displacement. Exclude walking, sniffing, and nest carrying.",
    "gnawing_nonfood": "Repeated visible oral movements contacting an identified nonfood object. Proximity alone is insufficient; exclude food chewing and drinking.",
    "rearing": "Forequarters raised with forepaws off the substrate; may overlap another behavior.",
}
LABELS = ("present", "absent", "unobservable", "ambiguous", "unreviewed")
KNOWN = {"present", "absent"}


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Times must be finite numbers.") from exc
    if not math.isfinite(result):
        raise ValueError("Times must be finite numbers.")
    return result


def validate_annotations(rows, start, end):
    """Labels are independent by behavior, disjoint within each behavior."""
    if not isinstance(rows, list):
        raise ValueError("Annotations must be a list.")
    clean = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each annotation must be an object.")
        behavior, label = row.get("behavior"), row.get("label")
        if not isinstance(behavior, str) or not isinstance(label, str) or behavior not in BEHAVIORS or label not in LABELS:
            raise ValueError("Choose a supported behavior and label.")
        a, b = number(row.get("start_s")), number(row.get("end_s"))
        if not start <= a < b <= end:
            raise ValueError("Intervals must have positive duration inside the analysis window.")
        clean.append(dict(behavior=behavior, label=label, start_s=a, end_s=b,
                          note=str(row.get("note", ""))[:2000]))
    clean.sort(key=lambda r: (r["behavior"], r["start_s"], r["end_s"]))
    previous = {}
    for row in clean:
        if row["start_s"] < previous.get(row["behavior"], start):
            raise ValueError("Intervals for the same behavior overlap. Edit or remove the earlier interval.")
        previous[row["behavior"]] = row["end_s"]
    return clean


def resolve_intervals(rows, behavior, start, end, gaps=()):
    """Fill unlabeled time as unreviewed; source gaps override human labels."""
    relevant = [r for r in rows if r["behavior"] == behavior]
    boundaries = {start, end}
    for r in relevant:
        boundaries.update((r["start_s"], r["end_s"]))
    clipped_gaps = [(max(start, a), min(end, b)) for a, b in gaps if a < end and b > start]
    for a, b in clipped_gaps:
        boundaries.update((a, b))
    boundaries = sorted(boundaries)
    result, cursor, gap_cursor = [], 0, 0
    clipped_gaps.sort()
    for a, b in zip(boundaries, boundaries[1:]):
        while cursor < len(relevant) and relevant[cursor]["end_s"] <= a:
            cursor += 1
        while gap_cursor < len(clipped_gaps) and clipped_gaps[gap_cursor][1] <= a:
            gap_cursor += 1
        label = "unreviewed"
        if cursor < len(relevant) and relevant[cursor]["start_s"] <= a:
            label = relevant[cursor]["label"]
        if gap_cursor < len(clipped_gaps) and clipped_gaps[gap_cursor][0] <= a:
            label = "source_gap"
        if result and result[-1]["label"] == label:
            result[-1]["end_s"] = b
        else:
            result.append(dict(start_s=a, end_s=b, label=label))
    return result


def measure(rows, start, end, gaps=(), merge_gap_s=0):
    """Duration is the sum of active time. Merge only known-negative gaps.

    Event counts are observed segments, not estimates of whole-session bouts.
    Unknown time and window edges censor boundaries. No minimum duration filter.
    """
    start, end, merge_gap_s = number(start), number(end), number(merge_gap_s)
    if end <= start or merge_gap_s < 0:
        raise ValueError("Invalid analysis window or gap-merging duration.")
    rows = validate_annotations(rows, start, end)
    summaries, bouts = [], []
    for behavior in BEHAVIORS:
        intervals = resolve_intervals(rows, behavior, start, end, gaps)
        durations = defaultdict(float)
        for r in intervals:
            durations[r["label"]] += r["end_s"] - r["start_s"]
        scored = durations["present"] + durations["absent"]
        events = []
        for i, r in enumerate(intervals):
            if r["label"] != "present":
                continue
            left = i == 0 or intervals[i-1]["label"] not in KNOWN
            right = i == len(intervals)-1 or intervals[i+1]["label"] not in KNOWN
            active = r["end_s"] - r["start_s"]
            can_merge = (events and i >= 2 and intervals[i-1]["label"] == "absent"
                         and intervals[i-2]["label"] == "present"
                         and r["start_s"] - events[-1]["end_s"] <= merge_gap_s + 1e-9)
            if can_merge:
                events[-1].update(end_s=r["end_s"], right_censored=right)
                events[-1]["active_seconds"] += active
            else:
                events.append(dict(behavior=behavior, start_s=r["start_s"], end_s=r["end_s"],
                                   active_seconds=active, left_censored=left, right_censored=right))
        for event in events:
            event["span_seconds"] = event["end_s"] - event["start_s"]
        unknown = end - start - scored
        summaries.append(dict(
            behavior=behavior, active_seconds=durations["present"] if scored else None,
            scored_seconds=scored, unknown_seconds=unknown,
            percent_scored=100 * durations["present"] / scored if scored else None,
            coverage_percent=100 * scored / (end-start),
            observed_segments=len(events) if scored else None,
            censored_segments=sum(e["left_censored"] or e["right_censored"] for e in events) if scored else None,
            unreviewed_seconds=durations["unreviewed"], unobservable_seconds=durations["unobservable"],
            ambiguous_seconds=durations["ambiguous"], source_gap_seconds=durations["source_gap"],
            review_status="unreviewed" if not rows else ("complete" if unknown < 1e-8 else "partial"),
            result_source="manual", detector_status="model_required",
        ))
        # Reviewing another behavior must not mark this behavior as reviewed.
        if not any(r["behavior"] == behavior for r in rows):
            summaries[-1]["review_status"] = "unreviewed"
        bouts.extend(events)
    return summaries, bouts
