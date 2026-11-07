"""B0: the hierarchical baseline of the who-acts head (docs/MODEL_PLAN.md, section 3, second stage).

Score = frozen offset (cross-fitted M1 or M0 scores) + independent parameters per entity x cell for the restricted terms:
  slot x zone, player x zone; slot x steps, player x steps; player x phase; slot and player x opponent event (steps 0-1);
  types: team style type x prematch role x steps; left foot, height and weight x zone (missing height or weight = 0)
Each term is a parameter block with its own ridge strength. The strengths are chosen term by term, in this order, by
2-fold cross-validation on the training targets (matches alternate between the folds in kickoff order); each step keeps
the strengths already chosen. No validation target is used. The final fit uses all reachable training targets.
Fitting: src/cond_logit.py. Writes <out>/b0.npz (scores and the entity part, (N, C)), b0_theta.npz, b0.json.
"""
import argparse
from multiprocessing import Pool

import numpy as np
from scipy import sparse

from common import ROOT, load
from cond_logit import Problem, fit
from model_data import load_meta, write_json

TERMS = ["slot_zone", "player_zone", "slot_steps", "player_steps", "player_phase", "slot_opp", "player_opp",
         "type_steps", "physique_zone"]
STATE = {}


def blocks(E, rows_t, slots):
    """Sparse design blocks over the candidate rows (rows_t, slots), one per term."""
    n = len(rows_t)
    zone, steps, phase, opp = (E[k][rows_t] for k in ("zone", "steps", "phase", "opp5"))
    player, slot, role, ttype = (E[k][rows_t, slots] for k in ("player", "slot", "role", "team_type"))
    P, S, R, TT = len(E["player_ids"]), len(E["slot_ids"]), int(E["role"].max()) + 1, int(E["team_type"].max()) + 1

    def onehot(col, width, keep=None):
        keep = np.ones(n, bool) if keep is None else keep
        r = np.flatnonzero(keep)
        return sparse.csr_matrix((np.ones(len(r)), (r, col[keep])), shape=(n, width))

    on = opp >= 0
    foot = E["left_foot"][player]
    num = np.stack([foot, E["height_z"][player], E["weight_z"][player]], 1)
    phys = sparse.hstack([sparse.csr_matrix((num[:, j], (np.arange(n), zone)), shape=(n, 9)) for j in range(3)])
    return {"slot_zone": onehot(slot * 9 + zone, S * 9), "player_zone": onehot(player * 9 + zone, P * 9),
            "slot_steps": onehot(slot * 5 + steps, S * 5), "player_steps": onehot(player * 5 + steps, P * 5),
            "player_phase": onehot(player * 3 + phase, P * 3),
            "slot_opp": onehot(slot * 5 + np.maximum(opp, 0), S * 5, on),
            "player_opp": onehot(player * 5 + np.maximum(opp, 0), P * 5, on),
            "type_steps": onehot((ttype * R + role) * 5 + steps, TT * R * 5), "physique_zone": phys.tocsr()}


def problem(terms, lams, rows):
    X = sparse.hstack([STATE["B"][t] for t in terms]).tocsr()[rows]
    lam = np.concatenate([np.full(STATE["B"][t].shape[1], l) for t, l in zip(terms, lams)])
    grp = np.searchsorted(np.unique(STATE["rows_t"][rows]), STATE["rows_t"][rows])
    return Problem(X, STATE["offset"][rows], STATE["y"][rows], grp, lam)


def cv_task(job):
    terms, lams, fold = job
    fit_rows = STATE["fit"] & (STATE["fold"] == 1 - fold)
    held = STATE["fit"] & (STATE["fold"] == fold)
    theta, info = fit(problem(terms, lams, np.flatnonzero(fit_rows)))
    held_prob = problem(terms, lams, np.flatnonzero(held))
    return terms[-1], lams[-1], fold, float(held_prob.loglik_per_target(theta).sum()), info["iterations"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    Y, meta = load_meta(cfg["scopes"])
    E = dict(np.load(ROOT / cfg["terms"]))
    O = np.load(ROOT / cfg["offsets"])["scores"]
    mask = np.isfinite(O)
    if not np.array_equal(mask, E["player"] >= 0):
        raise ValueError("offsets and entity terms cover different candidate slots")
    rows_t, slots = np.nonzero(mask)
    actor = Y["actor_slot"]
    block = meta["block"].to_numpy()
    train_matches = np.unique(meta.loc[block == 0, "match_id"])
    kickoff = {}
    for path in cfg["scopes"]:
        mcfg = load(ROOT / path)
        for r in load(ROOT / mcfg["train_manifest"]):
            kickoff[r["match_id"]] = r["kickoff"]
    order = sorted(train_matches, key=lambda m: (kickoff[m], m))
    fold_of = {m: i % 2 for i, m in enumerate(order)}
    fold_t = meta["match_id"].map(fold_of).fillna(-1).to_numpy().astype(int)
    STATE.update({"B": blocks(E, rows_t, slots), "rows_t": rows_t, "offset": O[rows_t, slots].astype(np.float64),
                  "y": (slots == actor[rows_t]).astype(np.float64), "fold": fold_t[rows_t],
                  "fit": (block[rows_t] == 0) & (actor[rows_t] >= 0)})

    chosen, cv = [], {}
    with Pool(min(cfg["workers"], 2 * len(cfg["lambda_grid"]))) as pool:
        for t in TERMS:
            jobs = [([*TERMS[:len(chosen)], t], [*chosen, lam], f) for lam in cfg["lambda_grid"] for f in (0, 1)]
            held = {}
            for term, lam, f, ll, it in pool.imap_unordered(cv_task, jobs):
                held[lam] = held.get(lam, 0.0) + ll
            best = max(cfg["lambda_grid"], key=lambda lam: held[lam])
            chosen.append(best)
            cv[t] = {"held_out_loglik": {str(k): v for k, v in sorted(held.items())}, "lambda": best}
            print(t, "lambda", best, {k: round(v, 1) for k, v in sorted(held.items())}, flush=True)

    rows_fit = np.flatnonzero(STATE["fit"])
    theta, info = fit(problem(TERMS, chosen, rows_fit))
    X_all = sparse.hstack([STATE["B"][t] for t in TERMS]).tocsr()
    ent = X_all @ theta
    N, C = O.shape
    scores = np.full((N, C), -np.inf, np.float32)
    part = np.zeros((N, C), np.float32)
    scores[rows_t, slots] = STATE["offset"] + ent
    part[rows_t, slots] = ent
    np.savez(out / "b0.npz", scores=scores, entity_part=part)
    widths = [STATE["B"][t].shape[1] for t in TERMS]
    np.savez(out / "b0_theta.npz", theta=theta, widths=np.array(widths), lambdas=np.array(chosen, float))
    write_json(out / "b0.json", {"offsets": cfg["offsets"], "terms": TERMS, "lambdas": dict(zip(TERMS, chosen)),
                                 "cross_validation": cv, "final_fit": info, "parameters": int(len(theta))})
    print("written", out, info, flush=True)


if __name__ == "__main__":
    main()
