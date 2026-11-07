"""Common actor metrics and paired match bootstrap, including empty cold-start groups.

NLL and candidate BCE use reachable targets; top-k and top-1 ECE include unreachable
targets as misses. Ties use exact expected hits under random tie order, shared with
the main development evaluator. Candidate BCE is evaluated from logits in float64, without probability
clipping. Cold groups use only candidate-side information available at prediction.
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from actor_benchmark_data import KEYS
from ranking_metrics import expected_topk

PAIR_FIELDS = (*KEYS, "league", "candidate_player", "candidate_mask", "actor_slot", "prior_matches",
               "known_in_training", "recent_substitute", "minute", "steps")


def check_alignment(a, b):
    for k in PAIR_FIELDS:
        if not np.array_equal(a[k], b[k]):
            raise ValueError(f"predictions are not paired on {k}")


def reliability(prob, truth, bins=15):
    bucket = np.minimum((prob * bins).astype(int), bins - 1)
    result = []
    for b in range(bins):
        sel = bucket == b
        result.append({"bin": b, "count": int(sel.sum()),
                       "mean_probability": float(prob[sel].mean()) if sel.any() else None,
                       "frequency": float(truth[sel].mean()) if sel.any() else None})
    return result


def score_archive(a):
    s = np.asarray(a["scores"], dtype=np.float64)
    mask = a["candidate_mask"].astype(bool)
    actor = a["actor_slot"].astype(int)
    if s.shape != mask.shape or not np.array_equal(np.isfinite(s), mask) or not np.all(np.isneginf(s[~mask])):
        raise ValueError("scores must be finite for real candidates and -inf for padding")
    if not mask.any(1).all() or (actor < -1).any() or (actor >= s.shape[1]).any():
        raise ValueError("invalid actors or empty candidate sets")
    reach = actor >= 0
    if not mask[np.flatnonzero(reach), actor[reach]].all():
        raise ValueError("actor outside candidate set")
    if len(set(zip(a["match_id"].tolist(), a["seq"].tolist()))) != len(s):
        raise ValueError("duplicate prediction keys")
    z = logsumexp(s, axis=1)
    lp = s - z[:, None]
    p = np.exp(lp)
    nll = np.where(reach, -lp[np.arange(len(s)), np.maximum(actor, 0)], 0.0)
    hit = expected_topk(s, actor)
    hit1, hit3 = hit[1], hit[3]
    y = np.arange(s.shape[1])[None] == actor[:, None]
    bce = np.zeros_like(s)
    for c in range(s.shape[1]):
        others = s.copy()
        others[:, c] = -np.inf
        neg = z - logsumexp(others, axis=1)
        bce[:, c] = np.where(y[:, c], -lp[:, c], neg)
    num = a["prior_matches"]
    groups = {"all_candidates": mask, "history_zero": mask & (num == 0),
              "history_1_4": mask & (num >= 1) & (num < 5), "history_lt5": mask & (num < 5),
              "unseen_in_training": mask & ~a["known_in_training"],
              "recent_substitutes": mask & a["recent_substitute"]}
    subsets = {"all": np.ones(len(s), bool), "first_15_minutes": a["minute"] < 15,
               "steps_0_2": a["steps"] <= 2, "cold_candidate_on_pitch": groups["history_lt5"].any(1),
               "unseen_candidate_on_pitch": groups["unseen_in_training"].any(1)}
    actor_prior = num[np.arange(len(s)), np.maximum(actor, 0)]
    actor_known = a["known_in_training"][np.arange(len(s)), np.maximum(actor, 0)]
    subsets.update({"reachable_actor_history_zero": reach & (actor_prior == 0),
                    "reachable_actor_history_1_4": reach & (actor_prior >= 1) & (actor_prior < 5),
                    "reachable_actor_history_lt5": reach & (actor_prior < 5),
                    "reachable_actor_unseen": reach & ~actor_known})
    for league in np.unique(a["league"]):
        subsets[f"league_{league}"] = a["league"] == league
    rows, metrics = [], {"actions": {}, "candidates": {}}
    mids, inverse = np.unique(a["match_id"], return_inverse=True)
    league_by_match = np.array([np.unique(a["league"][a["match_id"] == mid]).item() for mid in mids])

    def accumulate(group, values, counts):
        sums = np.bincount(inverse, weights=values, minlength=len(mids))
        ns = np.bincount(inverse, weights=counts, minlength=len(mids))
        rows.extend({"match_id": int(mid), "league": str(lg), "metric": group, "sum": float(v), "count": int(n)}
                    for mid, lg, v, n in zip(mids, league_by_match, sums, ns))

    for name, selected in subsets.items():
        r = selected & reach
        top = reliability(p.max(1)[selected], hit1[selected])
        ece = sum(b["count"] * abs(b["mean_probability"] - b["frequency"]) for b in top if b["count"])
        metrics["actions"][name] = {"targets": int(selected.sum()), "reachable": int(r.sum()),
                                    "nll": float(nll[r].mean()) if r.any() else None,
                                    "top1": float(hit1[selected].mean()) if selected.any() else None,
                                    "top3": float(hit3[selected].mean()) if selected.any() else None,
                                    "ece": float(ece / selected.sum()) if selected.any() else None,
                                    "reliability": top}
        accumulate(f"{name}/nll", np.where(r, nll, 0), r)
        accumulate(f"{name}/top1", np.where(selected, hit1, 0.0), selected)
        accumulate(f"{name}/top3", np.where(selected, hit3, 0.0), selected)
    for name, group in groups.items():
        selected = group & reach[:, None]
        if not np.isfinite(bce[selected]).all():
            raise ValueError("non-finite candidate loss")
        metrics["candidates"][name] = {"rows": int(selected.sum()), "positives": int(y[selected].sum()),
                                       "binary_log_loss": float(bce[selected].mean()) if selected.any() else None,
                                       "reliability": reliability(p[selected], y[selected])}
        accumulate(f"{name}/bce", np.where(selected, bce, 0).sum(1), selected.sum(1))
    return metrics, pd.DataFrame(rows)


def paired_comparison(first, second, draws, seed):
    key = ["metric", "league", "match_id"]
    a, b = first.sort_values(key).reset_index(drop=True), second.sort_values(key).reset_index(drop=True)
    if not a[key + ["count"]].equals(b[key + ["count"]]):
        raise ValueError("per-match denominators differ")
    rng = np.random.default_rng(seed)
    base = a[a.metric == "all/nll"]
    picks = {lg: rng.integers(0, int((base.league == lg).sum()), (draws, int((base.league == lg).sum())))
             for lg in base.league.unique()}
    out = {}
    for metric, part in a.groupby("metric", sort=True):
        other = b.loc[part.index]
        sign = -1 if metric.endswith(("/nll", "/bce")) else 1
        diff = sign * (part["sum"].to_numpy() - other["sum"].to_numpy())
        count = part["count"].to_numpy()
        bs, bn = np.zeros(draws), np.zeros(draws)
        for lg, pick in picks.items():
            selected = part.league.to_numpy() == lg
            bs += diff[selected][pick].sum(1)
            bn += count[selected][pick].sum(1)
        valid = bn > 0
        eligible = int((count > 0).sum())
        out[metric] = {"gain": float(diff.sum() / count.sum()) if count.sum() else None,
                       "ci95": np.quantile(bs[valid] / bn[valid], [.025, .975]).tolist() if eligible >= 2 and valid.any() else None,
                       "count": int(count.sum()), "contributing_matches": eligible, "nonempty_draws": int(valid.sum())}
    return out
