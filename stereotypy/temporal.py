"""Temporal context shared by training and incremental inference."""
import numpy as np

def context_indices(cuts, radius=3):
    """Edge-pad within each recording segment; never bridge sparse gaps/cuts."""
    cuts = np.asarray(cuts, bool).copy()
    if not len(cuts):
        return np.empty((0, radius*2+1), int)
    cuts[0] = True
    starts = np.flatnonzero(cuts)
    ends = np.r_[starts[1:], len(cuts)]
    result = np.empty((len(cuts), radius*2+1), int)
    for start, end in zip(starts, ends):
        result[start:end] = np.clip(np.arange(start, end)[:, None] + np.arange(-radius, radius+1), start, end-1)
    return result
