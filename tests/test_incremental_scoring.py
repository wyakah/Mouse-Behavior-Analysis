import numpy as np
from stereotypy.incremental import sections
from stereotypy.temporal import context_indices

def test_publication_preserves_full_temporal_context_at_every_boundary():
    times=np.arange(0,40,.5);cuts=np.zeros(len(times),bool);cuts[[0,15,37]]=True
    full=context_indices(cuts);covered=[]
    for first,stop,available,end in sections(times,40):
        partial=context_indices(cuts[:available])
        np.testing.assert_array_equal(partial[first:stop],full[first:stop])
        covered.extend(range(first,stop))
    assert covered==list(range(len(times)))
    first=next(sections(times,40))
    assert times[first[2]-1]<10 and first[1]<len(times)

def test_short_source_and_large_gaps_do_not_duplicate_samples():
    for times,duration in [(np.array([0,.5]),.8),(np.array([0,.5,20,20.5]),21)]:
        plans=list(sections(times,duration))
        assert [i for a,b,_,_ in plans for i in range(a,b)]==list(range(len(times)))

def test_index_cache_requires_matching_content_not_just_filename_or_size(tmp_path):
    import json
    from stereotypy.video import cached_index
    from threechamber.core import sha256
    source=tmp_path/'mouse.mov';source.write_bytes(b'original')
    folder=tmp_path/'outputs/stereo-example/work/001';folder.mkdir(parents=True)
    record=dict(source_size=source.stat().st_size,source_sha256=sha256(source),source_mtime_ns=1,timing_version='declared-duration-bounded-cap-v3',frames=[{'pts':0}])
    (folder/'source-index.json').write_text(json.dumps(record))
    (folder/'input.json').write_text(json.dumps(dict(source_index_sha256=sha256(folder/'source-index.json'))))
    cached=cached_index(tmp_path,source)
    assert cached['source_mtime_ns']==source.stat().st_mtime_ns and cached['frames']==record['frames']
    source.write_bytes(b'changed!')
    assert cached_index(tmp_path,source) is None
