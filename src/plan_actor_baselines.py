"""Print explicit search/confirmation commands; never launch jobs implicitly.

Confirmation chooses one LR per family from the complete common search grid using
validation NLL. All searches must have matching rows, seeds, configuration and budget.
"""
import argparse
import shlex
import numpy as np

from actor_baselines import FAMILIES
from actor_benchmark_metrics import check_alignment
from common import ROOT, load
from train_actor_baseline import write_json


def run_path(root, family, seed, lr):
    return root / family / f'seed_{seed}_lr_{lr:g}'


def select(cfg, root):
    selected, anchor = {}, None
    for family in FAMILIES:
        choices = []
        for lr in cfg['learning_rates']:
            d = run_path(root, family, cfg['search_seed'], lr)
            run = load(d / 'run.json')
            if (load(d / 'config.json') != cfg or run['mode'] != 'train' or run['family'] != family
                    or run['seed'] != cfg['search_seed'] or run['lr'] != lr
                    or run['epochs'] != cfg['training']['epochs']):
                raise ValueError(f'{d}: does not implement the specified search protocol')
            with np.load(d / 'predictions.npz') as p:
                archive = dict(p)
            with np.load(d / 'training_rows.npz') as p:
                rows = dict(p)
            if anchor is None:
                anchor = (archive, rows, run['optimizer_steps'])
            else:
                check_alignment(archive, anchor[0])
                if run['optimizer_steps'] != anchor[2] or any(not np.array_equal(rows[k], anchor[1][k]) for k in rows):
                    raise ValueError('search runs differ in training rows or optimizer budget')
            choices.append((run['best_validation_nll'], lr, str(d)))
        nll, lr, path = min(choices)  # Smaller LR breaks exact NLL ties.
        selected[family] = {'lr': lr, 'validation_nll': nll, 'search_run': path}
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', required=True, choices=['search', 'confirm'])
    p.add_argument('--python', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--runs-root', required=True)
    p.add_argument('--selection', help='required output JSON for confirmation stage')
    a = p.parse_args()
    cfg = load(ROOT / a.config)
    root = ROOT / a.runs_root
    if a.stage == 'confirm':
        if not a.selection:
            p.error('--selection is required for confirmation')
        chosen = select(cfg, root)
        dest = ROOT / a.selection
        if dest.exists():
            raise FileExistsError(dest)
        write_json(dest, chosen)
    for family in FAMILIES:
        seeds = [cfg['search_seed']] if a.stage == 'search' else cfg['confirmation_seeds']
        rates = cfg['learning_rates'] if a.stage == 'search' else [chosen[family]['lr']]
        for seed in seeds:
            for lr in rates:
                cmd = [a.python, str(ROOT / 'src/train_actor_baseline.py'), '--config', a.config,
                       '--family', family, '--seed', str(seed), '--lr', str(lr), '--mode', 'train',
                       '--device', a.device, '--out', str(run_path(root, family, seed, lr))]
                print(shlex.join(cmd))


if __name__ == '__main__':
    main()
