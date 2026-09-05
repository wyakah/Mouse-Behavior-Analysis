"""Shared recording setup contract for behavioral tests."""
from threechamber.statistics import sample_metadata

MAX_VIDEOS = 20


def validate_recordings(entries, require_ids=False):
    if not isinstance(entries, list) or len(entries) > MAX_VIDEOS:
        raise ValueError('Set up at most 20 videos at a time.')
    ids=set();videos=set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('video'), str) or not entry['video']:
            raise ValueError('Each row needs a video.')
        identifier=entry.get('id','')
        if not isinstance(identifier,str) or len(identifier.strip())>100:
            raise ValueError('Mouse IDs must be text up to 100 characters.')
        if require_ids and (not identifier.strip() or identifier.strip() in ids):
            raise ValueError('Each video needs a unique, nonempty mouse ID.')
        if require_ids and entry['video'] in videos:
            raise ValueError('The same video appears more than once.')
        videos.add(entry['video'])
        ids.add(identifier.strip())
        sample_metadata(entry)
    if require_ids and not entries:
        raise ValueError('Upload at least one video.')
    return entries
