"""Shared actor ranking metrics, with exact expected hits under random tie order."""
import numpy as np


def expected_topk(scores, actor, ks=(1, 3)):
    """For g strictly higher candidates and m tied candidates (including actor),
    expected top-k hit is clip(k-g, 0, m)/m. No random sampling is needed.
    Scores use -inf for padding; actor=-1 denotes an unreachable target.
    """
    scores, actor = np.asarray(scores), np.asarray(actor)
    if scores.ndim != 2 or actor.shape != (len(scores),):
        raise ValueError('ranking scores and labels have inconsistent shapes')
    if (actor < -1).any() or (actor >= scores.shape[1]).any() or any(k < 1 for k in ks):
        raise ValueError('invalid actor slot or top-k')
    reach = actor >= 0
    own = scores[np.arange(len(scores)), np.maximum(actor, 0)][:, None]
    if not np.isfinite(own[reach]).all():
        raise ValueError('reachable actor must have a finite score')
    above, tied = (scores > own).sum(1), (scores == own).sum(1)
    return {k: np.where(reach, np.clip(k - above, 0, tied) / np.maximum(tied, 1), 0.0) for k in ks}
