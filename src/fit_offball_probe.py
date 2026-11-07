"""Probe of the off-ball position estimates (src/offball_features.py) on a frozen who-acts offset (MODEL_PLAN.md, note 21).

score = offset (cross-fitted, e.g. M1 or M1-) + sum over the chosen estimates e in {d0, d2} and idle bins b of
  [a_eb * d_e / 10 + q_eb * (d_e / 10)^2] * 1{idle in b}, plus one indicator per estimate for "no estimate".
Idle bins (seconds since the candidate's last earlier action): 0-10, 10-30, 30-60, 60-180, 180+. A candidate without an
earlier action has no estimate. The coefficients are fitted on the reachable training targets with src/cond_logit.py
(ridge cfg["ridge"] on the summed NLL). Writes <out_dir>/<name>.npz (scores (N, C), -inf padding) and probe.json.
"""
import argparse

import numpy as np
from scipy import sparse

from common import ROOT, load
from cond_logit import Problem, fit
from model_data import load_meta, scope_dirs, write_json

BINS = [0, 10, 30, 60, 180, np.inf]


def design(F, rows_t, slots, estimates):
    idle = F["idle"][rows_t, slots]
    cols, names = [], []
    for e in estimates:
        d = F[e][rows_t, slots]
        has = d >= 0
        u = np.where(has, d, 0.0) / 10
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            inb = has & (idle >= lo) & (idle < hi)
            cols += [u * inb, u * u * inb]
            names += [f"{e}_lin_{lo}-{hi}", f"{e}_sq_{lo}-{hi}"]
        cols.append((~has).astype(float))
        names.append(f"{e}_missing")
    return np.stack(cols, 1), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    Y, meta = load_meta(cfg["scopes"])
    F = {k: np.concatenate([np.load(d / "offball.npz")[k] for d in scope_dirs(cfg["scopes"])]) for k in ("idle", "d0", "d2")}
    O = np.load(ROOT / cfg["offsets"])["scores"]
    mask = np.isfinite(O)
    rows_t, slots = np.nonzero(mask)
    actor = Y["actor_slot"]
    block = meta["block"].to_numpy()
    fit_rows = np.flatnonzero((block[rows_t] == 0) & (actor[rows_t] >= 0))
    y = (slots == actor[rows_t]).astype(float)
    info = {"offsets": cfg["offsets"], "probes": {}}
    for name, estimates in cfg["probes"].items():
        X, names = design(F, rows_t, slots, estimates)
        r = fit_rows
        prob = Problem(sparse.csr_matrix(X[r]), O[rows_t[r], slots[r]].astype(np.float64), y[r],
                       np.searchsorted(np.unique(rows_t[r]), rows_t[r]), cfg["ridge"])
        theta, res = fit(prob)
        S = np.full(O.shape, -np.inf, np.float32)
        S[rows_t, slots] = O[rows_t, slots] + X @ theta
        np.savez(out / f"{name}.npz", scores=S)
        info["probes"][name] = {"estimates": estimates, "coefficients": dict(zip(names, theta.tolist())), **res}
        print(name, {k: round(v, 4) for k, v in zip(names, theta)}, flush=True)
    write_json(out / "probe.json", info)


if __name__ == "__main__":
    main()
