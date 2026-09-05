"""Scientific data integrity and supervision semantics, not model accuracy tests."""
import copy
import numpy as np
import pytest
from stereotypy.training import validate_manifest, fitting_statistics


def dataset():
    return dict(sources={'a':{'sha256':'aaa'},'b':{'sha256':'bbb'}}, positive_bags={}, windows=[
        dict(id='a:1',source='a',group='mouse-a',split='train',start_s=0,end_s=2,labels=[1,-1,-1,0,-1,-1]),
        dict(id='b:1',source='b',group='mouse-b',split='test',start_s=0,end_s=2,labels=[0,-1,-1,1,-1,-1])])


def test_same_mouse_cannot_cross_splits_even_with_different_files():
    data = dataset(); data['windows'][1]['group']='mouse-a'
    with pytest.raises(ValueError, match='group crosses'):
        validate_manifest(data)


def test_renaming_a_source_cannot_hide_leakage():
    data = dataset(); data['sources']['b']['sha256']='aaa'
    with pytest.raises(ValueError, match='recording crosses'):
        validate_manifest(data)


@pytest.mark.parametrize('kind',['adaptation','bag'])
def test_unlabeled_data_cannot_silently_supply_negative_labels(kind):
    data = dataset(); row=data['windows'][0]
    if kind=='adaptation':row['split']='adaptation'
    else:row['bag']='example';data['positive_bags']['example']='jumping'
    with pytest.raises(ValueError, match='cannot supply window labels'):
        validate_manifest(data)
    row['labels']=[-1]*6
    validate_manifest(data)


def test_nonfinite_window_is_rejected():
    data=dataset();data['windows'][0]['end_s']=float('inf')
    with pytest.raises(ValueError, match='time interval'):validate_manifest(data)


def test_held_out_extreme_values_do_not_change_fitted_normalization():
    rows=[{'split':s} for s in ['train','adaptation','validation','test']]
    values=np.array([[[1.,2.]],[[3.,4.]],[[999.,999.]],[[999.,999.]]])
    mean, scale=fitting_statistics(values, rows)
    np.testing.assert_allclose(mean,[2,3]);np.testing.assert_allclose(scale,[1,1])
    values[2:]=1e9
    for a,b in zip((mean,scale),fitting_statistics(values,rows)):np.testing.assert_array_equal(a,b)


def test_unknown_class_labels_produce_no_classification_gradient():
    torch=pytest.importorskip('torch')
    from stereotypy.training import masked_loss
    logits=torch.tensor([[.4,-.2,1.,-.8,.3,.1]],requires_grad=True)
    labels=torch.tensor([[1.,-1.,-1.,0.,-1.,-1.]])
    masked_loss(logits,labels,torch.ones(6)).backward()
    assert logits.grad[0,0]<0 and logits.grad[0,3]>0
    assert torch.equal(logits.grad[0,[1,2,4,5]],torch.zeros(4))


def test_positive_video_bag_does_not_label_every_window_positive():
    torch=pytest.importorskip('torch')
    from stereotypy.training import positive_bag_loss
    logits=torch.tensor([[0.,1.],[0.,-1.],[0.,-2.]],requires_grad=True)
    positive_bag_loss(logits,1).backward()
    assert logits.grad[0,1]<0
    assert torch.equal(logits.grad[1:],torch.zeros((2,2)))
    assert torch.equal(logits.grad[:,0],torch.zeros(3))
