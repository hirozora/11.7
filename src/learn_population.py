"""Exploratory population-feature comparisons; metrics are binary nats per candidate row."""
import argparse
import json

import numpy as np
import pandas as pd

import hypothesis_statistics as ci
from pooling import forward_pool
from features import CAT_CTX, NUM_CTX
from numeric import loglik
from common import GROUP_OF, GROUPS, ROOT, build, crossfit_g, load, team_features

FEATURE_GROUPS = {
    "steps_phase": ["phase", "steps_code"],
    "opponent_event": ["opp_last_event", "prev_event"],
    "game_state": ["score_state", "score_diff", "minute", "time_bin", "minutes_since_goal", "state_duration"],
    "possession_start": ["possession_start", "poss_start_x"],
    "minutes_on": ["minutes_on"],
}
WHO_GROUPS = ["steps_phase", "opponent_event", "game_state", "minutes_on"]
EVENT_GROUPS = ["steps_phase", "game_state", "possession_start"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    rng = np.random.default_rng(cfg["seed"])
    A, meta = build(cfg)
    roles, nR = meta["roles"], len(meta["roles"])
    X, cat = team_features(A)
    names = CAT_CTX + NUM_CTX
    W = dict(np.load(ROOT / cfg["who_path"]))
    g_who = np.load(ROOT / cfg["offsets_path"])["who_prior"]
    if len(g_who) != len(W["y"]):
        raise ValueError("who offsets and causal rows differ in length")
    out_dir = ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, summary = {}, {}

    def record(name, new, old, y, match, evl):
        pm = ci.per_match((loglik(new, y) - loglik(old, y))[evl], match[evl])
        for key in ("match", "sum", "n"):
            arrays[f"{name}|{key}"] = pm[key]
        r = forward_pool([{"sum": pm["sum"], "n": pm["n"]}], cfg["bootstrap"], rng)
        summary[name] = {"gain": r["gain"], "ci": r["ci"], "rows": int(evl.sum()), "matches": int(len(pm["sum"]))}
        print(name, json.dumps(summary[name]), flush=True)

    # ---------------- who acts ----------------
    role = W["role_prior"]
    keep = role >= 0
    k_w = A["k"].to_numpy()[W["act"]]
    evl_w, est_w = keep & (k_w >= cfg["eval_from"]), keep & (k_w < cfg["eval_from"])
    cand = np.column_stack([np.where(keep, role, nR), W["mins_on"], W["starter"], W["cum"], W["since"]])
    Xw, catw = np.column_stack([X[W["act"]], cand]), cat + [True, False, False, False, False]
    all_names = names + ["role", "minutes_on", "starter", "cum", "since"]
    g_nogs = None
    for grp in WHO_GROUPS:
        cols = [j for j in range(Xw.shape[1]) if all_names[j] not in FEATURE_GROUPS[grp]]
        g = crossfit_g(Xw[:, cols], W["y"], [catw[j] for j in cols], W["match"], cfg)
        record(f"who|population|{grp}", g_who, g, W["y"], W["match"], evl_w)
        if grp == "game_state":
            g_nogs = g
    # confirmed form of the game state on top of g without the game-state group
    U = A["U"].to_numpy()[W["act"]].astype(float)
    U = (U - U[est_w].mean()) / U[est_w].std()
    grp_of = np.full(len(role), -1)
    grp_of[keep] = [GROUPS.index(GROUP_OF[roles[r]]) for r in role[keep]]
    Xu = np.column_stack([np.ones(len(U)), U] + [U * (grp_of == g) for g in range(1, len(GROUPS))])
    res = ci.glm_fit(W["y"][est_w], Xu[est_w], g_nogs[est_w], pd.factorize(W["match"][est_w])[0])
    beta = np.asarray(res.params)
    form_beta = dict(zip(["const", "U"] + [f"U x {GROUPS[g]}" for g in range(1, len(GROUPS))], beta.tolist()))
    print("game-state form beta", json.dumps(form_beta), flush=True)
    record("who|game_state_form", g_nogs + Xu @ beta, g_nogs, W["y"], W["match"], evl_w)

    # ---------------- event head ----------------
    actor = W["y"] == 1
    a_prior = np.full(len(A), -1)
    a_prior[W["act"][actor]] = role[actor]
    match_a = A["match_id"].to_numpy()
    evl_a = (A["k"].to_numpy() >= cfg["eval_from"]) & (a_prior >= 0)
    event_g = {}
    for target in cfg["team_targets"]:
        y = A[f"t_{target}"].to_numpy().astype(float)
        event_g[f"team_{target}"] = crossfit_g(X, y, cat, match_a, cfg)
        for grp in EVENT_GROUPS:
            cols = [j for j in range(X.shape[1]) if names[j] not in FEATURE_GROUPS[grp]]
            g = crossfit_g(X[:, cols], y, [cat[j] for j in cols], match_a, cfg)
            record(f"event|population|{target}|{grp}", event_g[f"team_{target}"], g, y, match_a, evl_a)
    np.savez(ROOT / cfg["event_offsets_path"], **event_g)
    np.savez(out_dir / "population.npz", **arrays)
    with open(out_dir / "population.json", "w", encoding="utf-8") as f:
        json.dump({"league": cfg["league"], "comparisons": summary, "game_state_form_beta": form_beta}, f, indent=1)
    print("written", out_dir / "population.npz", flush=True)


if __name__ == "__main__":
    main()
