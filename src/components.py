"""Finite-grid component paths. Thresholds must descend; masks grow with index."""
import numpy as np
from scipy import ndimage as ndi


def component_masks(scores, thresholds, connectivity=2, min_area=64):
    structure=ndi.generate_binary_structure(2,connectivity)
    anchor=np.unravel_index(np.argmax(scores),scores.shape)
    history=np.zeros_like(scores,dtype=bool)
    for threshold in thresholds:
        raw=scores>=threshold
        labels,n=ndi.label(raw,structure)
        counts=np.bincount(labels.ravel(),minlength=n+1); counts[0]=0
        winner=int(np.argmax(counts)) if n else 0
        largest=(labels==winner) if n else np.zeros_like(raw)
        history |= largest
        anchor_label=labels[anchor]
        anchored=(labels==anchor_label) if anchor_label else np.zeros_like(raw)
        opened=raw & (counts[labels]>=min_area)
        yield {'raw':raw,'lcc':largest,'prefix':history.copy(),'anchor':anchored,'opening':opened}


def crc_index(losses, alpha):
    """First passing grid index; standard CRC only justified for monotone loss."""
    losses=np.asarray(losses)
    if losses.ndim!=2 or len(losses)==0 or not 0<=alpha<=1:
        raise ValueError('Nonempty image-by-grid loss matrix and alpha in [0,1] required')
    if not np.isfinite(losses).all() or np.any((losses<0)|(losses>1)):
        raise ValueError('Losses must be finite and bounded in [0,1]')
    if not np.all(losses[:,-1]==0):
        raise ValueError('A known zero-loss terminal action is required')
    n=len(losses)
    ok=np.flatnonzero((losses.sum(axis=0)+1)/(n+1)<=alpha)
    if len(ok)==0:
        # Full-frame terminal action has identically zero risk; used for tiny n.
        return losses.shape[1]-1
    return int(ok[0])


def suffix_envelope(losses):
    return np.maximum.accumulate(losses[...,::-1],axis=-1)[...,::-1]
