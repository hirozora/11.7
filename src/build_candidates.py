"""On-pitch candidates and rolling prematch position priors for the selected non-test matches."""
import argparse
import json
import numpy as np
import pandas as pd
from event_schema import DEPTH, LATERAL, PERIOD_ORDER
from common import ROOT, build, load


DEPTH_UP = np.array([u for u, _ in DEPTH])
LATERAL_UP = np.array([u for u, _ in LATERAL])

def role_index(dx, y, roles):
    """Fine-role index of depth dx and lateral y (fixed event_schema bands), vectorized."""
    names = np.array([f"{d}/{l}" for _, d in DEPTH for _, l in LATERAL])
    code = np.searchsorted(DEPTH_UP, dx, side="right") * len(LATERAL) + np.searchsorted(LATERAL_UP, y, side="right")
    lookup = np.array([roles.index(n) for n in names])
    return lookup[code]


def prior_sums(A, player_match_path):
    """Per (match, player): event count, sum of n * dx and sum of n * mean y over the player's previous matches with a
    defined depth (outfield, at least 10 events)."""
    pm = pd.read_parquet(player_match_path)
    kick = A.groupby("match_id")["kickoff"].first()
    pm = pm[pm["match_id"].isin(kick.index)].assign(kickoff=lambda d: d["match_id"].map(kick))
    valid = pm["dx"].notna().to_numpy()
    pm = pm.assign(w=np.where(valid, pm["n_events"], 0.0), w_dx=np.where(valid, pm["dx"] * pm["n_events"], 0.0),
                   w_y=np.where(valid, pm["mean_y"] * pm["n_events"], 0.0))
    hist = {p: g.sort_values("kickoff") for p, g in pm.groupby("player_id")}
    return hist, kick


def expand_candidates(A, onpitch, tol, cap):
    spans = {k: g for k, g in onpitch.groupby(["match_id", "team_id"])}
    out = {k: [] for k in ("act", "cand", "mins_on", "starter", "cum", "since")}
    minute_all, actor_all = A["minute"].to_numpy(), A["player_id"].to_numpy()
    per_all, sec_all = A["period"].map(PERIOD_ORDER).to_numpy(), A["event_sec"].to_numpy()
    counts = {"actions": len(A), "actions_without_actor_in_set": 0, "rows_on_by_tolerance_only": 0}
    for (m, t), idx in A.groupby(["match_id", "team_id"]).indices.items():
        sp = spans[(m, t)]
        minute, actor, per, sec = minute_all[idx], actor_all[idx], per_all[idx], sec_all[idx]
        start, sub_end, cand = sp["start"].to_numpy(), sp["sub_end"].to_numpy(), sp["player_id"].to_numpy()
        cper, csec = sp["card_period"].to_numpy(), sp["card_sec"].to_numpy()
        before_card = ((cper[None, :] < 0) | (per[:, None] < cper[None, :])
                       | ((per[:, None] == cper[None, :]) & (sec[:, None] < csec[None, :])))
        on = (minute[:, None] >= start[None, :] - tol) & (minute[:, None] < sub_end[None, :] + tol) & before_card
        strict = (minute[:, None] >= start[None, :]) & (minute[:, None] < sub_end[None, :]) & before_card
        is_actor = actor[:, None] == cand[None, :]
        keep = (on & is_actor).any(1)
        counts["actions_without_actor_in_set"] += int((~keep).sum())
        on &= keep[:, None]
        counts["rows_on_by_tolerance_only"] += int((on & ~strict).sum())
        cum = np.cumsum(is_actor, axis=0) - is_actor
        n = len(idx)
        pos = np.where(is_actor, np.arange(n)[:, None], -1)
        last = np.vstack([np.full((1, len(cand)), -1), np.maximum.accumulate(pos, axis=0)[:-1]])
        since = np.where(last >= 0, np.arange(n)[:, None] - last, cap)
        i, j = np.nonzero(on)
        out["act"].append(idx[i])
        out["cand"].append(cand[j])
        out["mins_on"].append(minute[i] - start[j])
        out["starter"].append((start[j] == 0).astype(float))
        out["cum"].append(cum[i, j].astype(float))
        out["since"].append(np.minimum(since[i, j], cap).astype(float))
    return {k: np.concatenate(v) for k, v in out.items()}, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    A, meta = build(cfg)
    roles = meta["roles"]
    gk = roles.index("GK")
    onpitch = pd.read_parquet(ROOT / cfg["onpitch_path"])
    W, counts = expand_candidates(A, onpitch, cfg["onpitch_tolerance_minutes"], cfg["since_cap"])
    W["y"] = (A["player_id"].to_numpy()[W["act"]] == W["cand"]).astype(float)
    W["match"] = A["match_id"].to_numpy()[W["act"]]
    players = {p["wyId"]: p for p in load((ROOT / cfg["raw_dir"]) / "entities" / "players.json")}
    is_gk = np.array([players[c]["role"]["code2"] == "GK" for c in W["cand"]])

    # Prematch position uses only strictly earlier kickoffs.
    hist, kick = prior_sums(A, ROOT / cfg["player_match_path"])
    keys = pd.DataFrame({"match": W["match"], "cand": W["cand"]}).drop_duplicates()
    prior = {}
    for m, c in zip(keys["match"], keys["cand"]):
        h = hist.get(c)
        if h is None:
            prior[(m, c)] = (0.0, 0.0, 0.0)
            continue
        before = h[h["kickoff"] < kick[m]]
        prior[(m, c)] = (float(before["w"].sum()), float(before["w_dx"].sum()), float(before["w_y"].sum()))
    pw, pdx, py = (np.array([prior[(m, c)][i] for m, c in zip(W["match"], W["cand"])]) for i in range(3))
    W["role_prior"] = np.where(is_gk, gk, -1)
    has = (~is_gk) & (pw > 0)
    W["role_prior"][has] = role_index(pdx[has] / pw[has], py[has] / pw[has], roles)

    W["weight_prior"] = np.where(is_gk, np.nan, pw)
    np.savez(ROOT / cfg["who_path"], **W)
    counts.update({"rows": int(len(W["y"])), "actions_in_who_rows": int(len(np.unique(W["act"]))),
                   "prior_role_defined": float(np.mean(W["role_prior"] >= 0))})
    with open(ROOT / cfg["causal_counts_path"], "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
