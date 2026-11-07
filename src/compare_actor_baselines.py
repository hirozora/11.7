"""Paired per-match bootstrap; a positive gain favours --candidate over --reference."""
import argparse
import numpy as np

from actor_benchmark_metrics import check_alignment, paired_comparison, score_archive
from common import ROOT, load
from train_actor_baseline import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, help='run directory')
    p.add_argument('--candidate', required=True, help='run directory')
    p.add_argument('--draws', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.draws < 100:
        raise ValueError('use at least 100 bootstrap draws')
    first, second = ROOT / a.candidate, ROOT / a.reference
    runs = [load(d / 'run.json') for d in (first, second)]
    if any(r['mode'] != 'train' for r in runs):
        raise ValueError('smoke outputs cannot enter research comparisons')
    if (load(first / 'config.json') != load(second / 'config.json')
            or runs[0]['seed'] != runs[1]['seed'] or runs[0]['optimizer_steps'] != runs[1]['optimizer_steps']):
        raise ValueError('compare matching protocols, seeds and training budgets')
    with np.load(first / 'training_rows.npz') as p, np.load(second / 'training_rows.npz') as q:
        if p.files != q.files or any(not np.array_equal(p[k], q[k]) for k in p.files):
            raise ValueError('training target rows differ')
    x, y = dict(np.load(first / 'predictions.npz')), dict(np.load(second / 'predictions.npz'))
    check_alignment(x, y)
    _, tx = score_archive(x)
    _, ty = score_archive(y)
    out = ROOT / a.out
    if out.exists():
        raise FileExistsError(out)
    write_json(out, {'candidate': a.candidate, 'reference': a.reference, 'draws': a.draws, 'seed': a.seed,
                     'positive_gain_favours': 'candidate',
                     'interpretation': 'development-set intervals after model selection, not independent test evidence',
                     'metrics': paired_comparison(tx, ty, a.draws, a.seed)})


if __name__ == '__main__':
    main()
