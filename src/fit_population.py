"""Population models without entity identity (docs/MODEL_PLAN.md, section 3, first stage): R1, R2 and M0.

  R1  most recent actor: score = beta * (-log1p(own-team actions since the candidate's last action))
  R2  nearest prematch position: score = beta * (-distance / 50 m); unknown distance = median known training distance
  M0  binary gradient boosting on candidate rows (settings of common.crossfit_g), inputs = context summary and the
      first cfg["m0_candidate_columns"] columns of cand_num: 10 = candidate state (no window details, no own-action
      pooling; the M0 of the plan), 36 = state and own-action pooling (a diagnostic that splits M1 - M0 into the
      pooled in-match information and the rest); cross-fitted over the M1 folds on training targets,
      fitted on all training targets for validation; then, on the conditional logit with that offset, the confirmed
      urgency x position-group term (7 coefficients, CM the reference group)
beta and the urgency term are fitted on reachable training targets with src/cond_logit.py. Writes <out_dir>/r1.npz,
r2.npz, m0.npz (scores (N, C), -inf padding) and population.json.
"""
import argparse

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier

from common import ROOT, load
from cond_logit import Problem, fit
from model_data import folds, load_scopes, write_json



def flat_problem(rows_t, slots, actor, x_cols, offset):
    """Conditional-logit problem over the candidate rows of the given (reachable) targets."""
    group = np.searchsorted(np.unique(rows_t), rows_t)
    y = (slots == actor[rows_t]).astype(float)
    return Problem(sparse.csr_matrix(x_cols), offset, y, group, lam=0.0)


def scores_matrix(N, C, rows_t, slots, values):
    out = np.full((N, C), -np.inf, np.float32)
    out[rows_t, slots] = values
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    X, Y, meta, shared = load_scopes(cfg["scopes"])
    N, C = X["cand_mask"].shape
    actor = Y["actor_slot"]
    block = meta["block"].to_numpy()
    fold = folds(meta, cfg["folds"], cfg["fold_seed"])
    rows_t, slots = np.nonzero(X["cand_mask"])
    y = (slots == actor[rows_t]).astype(np.int8)
    reach_t = actor >= 0
    fit_rows = (block[rows_t] == 0) & reach_t[rows_t]
    info = {}

    # R1, R2: one coefficient each
    since = X["cand_num"][rows_t, slots, 3] * 200
    dist = X["cand_num"][rows_t, slots, 8] * 50
    known = X["cand_num"][rows_t, slots, 9] > 0
    med = float(np.median(dist[known & (block[rows_t] == 0)]))
    feats = {"r1": -np.log1p(since), "r2": -np.where(known, dist, med) / 50}
    for name, x in feats.items():
        prob = flat_problem(rows_t[fit_rows], slots[fit_rows], actor, x[fit_rows, None], np.zeros(fit_rows.sum()))
        beta, res = fit(prob)
        np.savez(out / f"{name}.npz", scores=scores_matrix(N, C, rows_t, slots, beta[0] * x))
        info[name] = {"beta": float(beta[0]), **res}
        print(name, info[name], flush=True)
    info["r2"]["median_distance_m"] = med

    # M0: gradient boosting on candidate rows
    F = np.concatenate([X["ctx_cat"][rows_t].astype(np.float32), X["ctx_num"][rows_t],
                        X["cand_cat"][rows_t, slots].astype(np.float32), X["cand_num"][rows_t, slots, :cfg["m0_candidate_columns"]]], 1)
    cat = ([True] * X["ctx_cat"].shape[1] + [False] * X["ctx_num"].shape[1] + [True, True, True, True]
           + [False] * cfg["m0_candidate_columns"])
    g = np.full(len(rows_t), np.nan)
    fold_r = fold[rows_t]
    params = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                  categorical_features=cat, early_stopping=False, random_state=cfg["fold_seed"])
    for f in list(range(cfg["folds"])) + ["full"]:
        train = (block[rows_t] == 0) & (fold_r != f) if f != "full" else (block[rows_t] == 0)
        target = (fold_r == f) if f != "full" else (block[rows_t] == 1)
        model = HistGradientBoostingClassifier(**params).fit(F[train], y[train])
        p = np.clip(model.predict_proba(F[target])[:, 1], 1e-6, 1 - 1e-6)
        g[target] = np.log(p / (1 - p))
        print("M0 gradient boosting", f, "fitted on", int(train.sum()), "rows", flush=True)
    if np.isnan(g).any():
        raise ValueError("candidate rows without an M0 offset")
    u = X["urgency"][rows_t]
    u = (u - X["urgency"][block == 0].mean()) / X["urgency"][block == 0].std()
    grp = X["cand_cat"][rows_t, slots, 2].astype(int)
    U = np.stack([u * (grp == j) for j in range(1, len(shared["groups"]))], 1)
    prob = Problem(sparse.csr_matrix(U[fit_rows]), g[fit_rows], y[fit_rows].astype(float),
                   np.searchsorted(np.unique(rows_t[fit_rows]), rows_t[fit_rows]), lam=1e-6)
    w, res = fit(prob)
    np.savez(out / "m0.npz", scores=scores_matrix(N, C, rows_t, slots, g + U @ w))
    info["m0"] = {"urgency_by_group": dict(zip(shared["groups"][1:], w.tolist())), **res,
                  "gbdt": {k: v for k, v in params.items() if k != "categorical_features"}}
    write_json(out / "population.json", info)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
