"""Fit match-cross-fitted population offsets for one explicitly selected head."""
import argparse

import numpy as np

from common import ROOT, build, crossfit_g, load, team_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--head', choices=['who', 'event'], required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    A, meta = build(cfg)
    X, cat = team_features(A)
    if args.head == 'who':
        with np.load(ROOT / cfg['who_path']) as W:
            role = W['role_prior']
            cand = np.column_stack([np.where(role >= 0, role, len(meta['roles'])),
                                    W['mins_on'], W['starter'], W['cum'], W['since']])
            Xw = np.column_stack([X[W['act']], cand])
            g = crossfit_g(Xw, W['y'], cat + [True, False, False, False, False], W['match'], cfg)
            out = {'rows': np.array(len(W['y'])), 'who_prior': g}
        path = ROOT / cfg['offsets_path']
    else:
        out = {f'team_{t}': crossfit_g(X, A[f't_{t}'].to_numpy().astype(float), cat,
                                      A['match_id'].to_numpy(), cfg) for t in cfg['team_targets']}
        path = ROOT / cfg['event_offsets_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print('written', path, flush=True)


if __name__ == '__main__':
    main()
