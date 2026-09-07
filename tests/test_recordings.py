from copy import deepcopy
import pytest
from threechamber.recordings import validate_recordings
from threechamber.batch import validate_batch


def entries(n):return [dict(id=f'M{i:02}',video=f'video{i}.mp4',sex='female',genotype='WT') for i in range(n)]


def test_150_recordings_allowed_and_151_rejected_before_video_work(tmp_path):
    assert len(validate_recordings(entries(150),True))==150
    with pytest.raises(ValueError,match='150 videos'):validate_recordings(entries(151))
    with pytest.raises(ValueError,match='150 videos'):validate_batch(tmp_path,{'entries':entries(151)})


def test_drafts_allow_incomplete_ids_but_continue_requires_unique_ids():
    data=entries(2);data[0]['id']=''
    validate_recordings(data)
    with pytest.raises(ValueError,match='mouse ID'):validate_recordings(data,True)
    data[0]['id']=data[1]['id']
    with pytest.raises(ValueError,match='mouse ID'):validate_recordings(data,True)


def test_shared_validation_preserves_metadata_and_refuses_invalid_sex():
    data=entries(150);original=deepcopy(data);validate_recordings(data,True);assert data==original
    data[0]['sex']='invalid'
    with pytest.raises(ValueError,match='Sex'):validate_recordings(data)
