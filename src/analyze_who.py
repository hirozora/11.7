"""Compute the registered actor-side items using prematch roles and exact candidate sets."""
import argparse
import json
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

import hypothesis_statistics as ci
import entity_types as r8
from numeric import loglik, pop_offset
from entity_units import make, units
from common import GROUP_OF, GROUPS, ROOT, Z, build, load, permute_units, standardize

STATE = {}
STEPS_L = 5


def vc_job(job):
    kind, name, b = job
    cfg = STATE["cfg"]
    spec = (STATE["est_specs"] if kind.startswith("est") else STATE["specs"])[name]
    labels = spec["unit_label"]
    if kind.endswith("perm"):
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(f"sens|{cfg['league']}|{kind}|{name}|{b}".encode())])
        labels = permute_units(labels, spec["strata"], rng)
    T, V, kept, _, _ = ci.vc_stat(spec, labels)
    return kind, name, b, T, V, kept


def glm_who(y, role, roles, zone, steps, match, U, F, M, opp, cfg):
    """R1-Pa, R4-P and R3-P with prematch roles for the offset and position groups."""
    grp = np.array([GROUPS.index(GROUP_OF[roles[r]]) for r in role])
    gW = pd.factorize(match)[0]
    offw = pop_offset(y, role, zone, steps, np.ones(len(y), bool), cfg["reference_pseudocount"])
    dummies = [(grp == g).astype(float) for g in range(1, len(GROUPS))]
    out = {}
    X = np.column_stack([np.ones(len(y)), U] + [U * d for d in dummies])
    out["R1-Pa"] = ci.glm_item(y, X, ["const", "U"] + [f"U x {GROUPS[g]}" for g in range(1, len(GROUPS))], offw, gW,
                               list(range(2, 2 + len(dummies))))
    X = np.column_stack([np.ones(len(y)), F] + [F * d for d in dummies] + [M] + [M * d for d in dummies])
    names = (["const", "F"] + [f"F x {GROUPS[g]}" for g in range(1, len(GROUPS))] + ["minute"]
             + [f"minute x {GROUPS[g]}" for g in range(1, len(GROUPS))])
    out["R4-P"] = ci.glm_item(y, X, names, offw, gW, list(range(2, 2 + len(dummies))))
    s01 = steps <= 1
    levels = ["duel", "interception", "clearance", "other"]
    od = [(opp[s01] == o).astype(float) for o in levels]
    gd = [(grp[s01] == g).astype(float) for g in range(1, len(GROUPS))]
    cols = [np.ones(s01.sum())] + od + [(steps[s01] == 1).astype(float)] + [o * d for o in od for d in gd]
    names = ["const"] + levels + ["steps1"] + [f"{o} x {GROUPS[g]}" for o in levels for g in range(1, len(GROUPS))]
    out["R3-P"] = ci.glm_item(y[s01], np.column_stack(cols), names, offw[s01], gW[s01], list(range(6, 6 + len(od) * len(gd))))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    league, lam, alpha = cfg["league"], cfg["type_ridge"], cfg["alpha_beta"]
    ci.STATE["cfg"] = cfg
    STATE["cfg"] = cfg
    A, meta = build(cfg)
    roles, nR = meta["roles"], len(meta["roles"])
    Wc = dict(np.load(ROOT / cfg["who_path"]))
    keep = Wc["role_prior"] >= 0
    act = Wc["act"][keep]
    rc = {k: v[keep] for k, v in Wc.items()}
    role, y = rc["role_prior"], rc["y"]
    match, team_i = A["match_id"].to_numpy()[act], A["team_i"].to_numpy()[act]
    zone, steps, k = A["zone"].to_numpy()[act], A["steps_code"].to_numpy()[act], A["k"].to_numpy()[act]
    phase = A["phase"].map({"first": 0, "early": 1, "settled": 2}).to_numpy()[act]
    opp5 = A["opp5"].to_numpy()[act]
    opp = A["opp5"].map({"pass": 0, "duel": 1, "interception": 2, "clearance": 3, "other": 4}).to_numpy()[act]
    U, F, M = standardize(A["U"].to_numpy()[act]), standardize(rc["mins_on"]), standardize(A["minute"].to_numpy()[act])
    items = {"league": league, "setting": "on-pitch candidates, prematch role, cross-fitted population offset",
             "rows": {"candidates_all": int(len(Wc["y"])), "with_prior_role": int(keep.sum())}}
    items["glm"] = glm_who(y, role, roles, zone, steps, match, U, F, M, opp5, cfg)
    items["p1"] = ci.p1_counts({"role": role, "steps": steps, "y": y}, roles)
    if cfg["mode"] == "reference":
        prof = np.array(items["p1"]["acts"]) / np.array(items["p1"]["opportunities"])
        ref = {"glm_beta": {n: r["beta"] for n, r in items["glm"].items()}, "p1_roles": roles,
               "p1_profile": np.log(prof[:, 4] / prof[:, 0]).tolist()}
        (ROOT / cfg["reference_json"]).parent.mkdir(parents=True, exist_ok=True)
        with open(ROOT / cfg["reference_json"], "w", encoding="utf-8") as f:
            json.dump(ref, f, indent=1)
        print(league, "reference written", cfg["reference_json"], flush=True)
        return
    if cfg["mode"] != "full":
        raise ValueError(f"mode must be reference or full, got {cfg['mode']}")
    print(league, "GLM done", flush=True)

    off = np.load(ROOT / cfg["offsets_path"])
    if int(off["rows"]) != len(Wc["y"]):
        raise ValueError("offsets do not match the rows")
    g = off["who_prior"][keep]
    era, oppc = A["era"].to_numpy()[act], (A["opponent_class"] == "big_six").to_numpy()[act].astype(int)
    slot_u, player_u = units(match, team_i, rc["cand"], role, era, oppc, nR)
    s01 = steps <= 1
    specs = {
        "R1-Ea": make(y, g, zone, k, match, slot_u, "slope", role, nR, x=U),
        "R1-Eb": make(y, g, zone, k, match, player_u, "slope", x=U),
        "R3-Ea": ci.subset(make(y, g, zone, k, match, slot_u, "cells", role, nR, level=opp, L=5), s01),
        "R3-Eb": ci.subset(make(y, g, zone, k, match, player_u, "cells", level=opp, L=5), s01),
        "R4-Ea": make(y, g, zone, k, match, player_u, "slope", x=F),
        "R4-Eb": make(y, g, zone, k, match, slot_u, "slope", role, nR, x=F),
        "T1": make(y, g, zone, k, match, slot_u, "cells", role, nR, level=steps, L=STEPS_L),
        "I1a": make(y, g, zone, k, match, player_u, "cells", level=steps, L=STEPS_L),
        "I1b": make(y, g, zone, k, match, player_u, "cells", level=phase, L=3),
    }
    x = A["x"].to_numpy()[act]
    band = (x > 37.5) & (x <= 63.5)
    cf = np.array(roles)[role] == "front/centre"
    specs["C2"] = ci.subset({**make(y, g, zone, k, match, player_u, "cells", level=(band & (phase == 1)).astype(int), L=2),
                             "style_cell": zone * 2 + band, "n_style": 2 * Z}, cf)
    # types: prior role (the type role), left foot, height and weight (z-scores over this league's candidates with both)
    attrs = {p["wyId"]: p for p in load(ROOT / cfg["players_json"])}
    valid = np.array([attrs[c]["height"] > 0 and attrs[c]["weight"] > 0 for c in rc["cand"]])
    cands = np.unique(rc["cand"][valid])
    hw = pd.DataFrame({"h": [attrs[c]["height"] for c in cands], "w": [attrs[c]["weight"] for c in cands]}, index=cands)
    zs = (hw - hw.mean()) / hw.std()
    hz, wz = zs["h"].reindex(rc["cand"]).to_numpy(), zs["w"].reindex(rc["cand"]).to_numpy()
    left = np.array([attrs[c]["foot"] == "left" for c in rc["cand"]], dtype=int)
    r1 = s01 & valid
    sp = ci.subset(make(y, g, zone, k, match, player_u, "cells", level=opp, L=5), r1)
    lv, rl = opp[r1], role[r1]
    ty1_cells = [(rl * 5 + lv, nR * 5), (left[r1] * 5 + lv, 10)]
    ty1_slopes = [(hz[r1], lv, 5), (wz[r1], lv, 5)]
    specs["TY1|none"] = sp
    specs["TY1|type"] = {**sp, "g": sp["g"] + r8.fit_terms(sp["g"], sp["y"], np.ones(len(sp["y"]), bool), ty1_cells, ty1_slopes, lam)}
    K = cfg["team_type_K"]
    feats, ttypes = r8.team_types(A, [K], cfg["seed"])
    ttype = ttypes[K]
    ty_team = pd.Series(meta["teams"]).map(ttype).to_numpy()[team_i]
    s4 = ci.subset(specs["T1"], valid)
    est4 = s4["k"] <= 19
    st4, rl4 = steps[valid], role[valid]
    ty4_cells = [((ty_team[valid] * nR + rl4) * STEPS_L + st4, K * nR * STEPS_L), (rl4 * STEPS_L + st4, nR * STEPS_L),
                 (left[valid] * STEPS_L + st4, 2 * STEPS_L)]
    ty4_slopes = [(hz[valid], st4, STEPS_L), (wz[valid], st4, STEPS_L)]
    s4 = {**s4, "g": s4["g"] + r8.fit_terms(s4["g"], s4["y"], est4, ty4_cells, ty4_slopes, lam)}
    est_specs = {"TY1|none": ci.subset(sp, sp["k"] <= 19), "TY4": ci.subset(s4, est4)}
    mc, mi_ = pd.factorize(sp["match"])
    STATE.update({"specs": specs, "est_specs": est_specs})
    ci.STATE.update({"cfg": cfg, "ty1": sp, "ty1_match_i": mc, "ty1_n_matches": len(mi_), "ty1_cells": ty1_cells, "ty1_slopes": ty1_slopes})
    items["types"] = {"ty1_rows": int(len(sp["y"])), "team_types": {str(int(t)): int(v) for t, v in ttype.items()}}
    print(league, "specs:", {n: len(s["y"]) for n, s in specs.items()}, flush=True)

    B, Bf = cfg["permutations"], cfg["forward_permutations"]
    jobs = [("obs", n, 0) for n in specs] + [("perm", n, b) for n in specs for b in range(B)]
    jobs += [("est_obs", n, 0) for n in est_specs] + [("est_perm", n, b) for n in est_specs for b in range(Bf)]
    jobs.sort(key=lambda j: -len((est_specs if j[0].startswith("est") else specs)[j[1]]["y"]))
    obs, perm, est_obs, est_perm = {}, {n: np.full(B, np.nan) for n in specs}, {}, {n: [] for n in est_specs}
    with Pool(cfg["workers"]) as pool:
        for i, (kind, n, b, T, V, kept) in enumerate(pool.imap_unordered(vc_job, jobs, chunksize=4)):
            if kind == "obs":
                obs[n] = (T, V, kept)
            elif kind == "perm":
                perm[n][b] = T
            elif kind == "est_obs":
                est_obs[n] = (T, V, kept)
            else:
                est_perm[n].append(T)
            if (i + 1) % 5000 == 0 or i + 1 == len(jobs):
                print(f"{league}: {i + 1}/{len(jobs)} fits", flush=True)
        ratio = sorted(pool.imap_unordered(ci.ratio_task, range(cfg["ratio_bootstrap"]), chunksize=1))
    arrays = {}
    items["vc"] = {}
    for n in specs:
        if np.isnan(perm[n]).any():
            raise RuntimeError(f"missing permutations for {n}")
        T, V, kept = obs[n]
        arrays[f"perm|{n}"] = perm[n]
        items["vc"][n] = {"T": T, "V": V, "kept": kept, "perm_mean": float(perm[n].mean()), "perm_sd": float(perm[n].std(ddof=1)),
                          "rows": int(len(specs[n]["y"]))}
    hyper = {}
    for n in est_specs:
        T, V, kept = est_obs[n]
        pm = float(np.mean(est_perm[n]))
        hyper[n] = {"tau2": max(0.0, (T - pm) / V), "phi": pm / kept}
    items["hyper"] = hyper
    r = np.array(ratio)
    for j, n in enumerate(("none", "type")):
        arrays[f"ratio|{n}|T"], arrays[f"ratio|{n}|null"], arrays[f"ratio|{n}|V"] = r[:, 1 + 3 * j], r[:, 2 + 3 * j], r[:, 3 + 3 * j]

    fwd = {}
    est, evl = sp["k"] <= 19, sp["k"] >= cfg["eval_from"]
    m0 = ci.style_fit(sp, sp["g"], est, alpha)
    variants = {"full": (ty1_cells, ty1_slopes), "role": (ty1_cells[:1], []), "physical": (ty1_cells[1:], ty1_slopes)}
    mt = {v: ci.style_fit(sp, sp["g"] + r8.fit_terms(sp["g"], sp["y"], est, c, s, lam), est, alpha) for v, (c, s) in variants.items()}
    mi = m0 + r8.eb_effect(sp, m0, est, hyper["TY1|none"]["tau2"], hyper["TY1|none"]["phi"], cfg)
    yy, mm = sp["y"][evl], sp["match"][evl]
    ll = {v: loglik(eta[evl], yy) for v, eta in {"base": m0, "individual": mi, **mt}.items()}
    fwd["TY1a"] = ci.per_match(ll["full"] - ll["base"], mm)
    fwd["TY1b"] = ci.per_match(ll["full"] - ll["individual"], mm)
    fwd["TY1_individual"] = ci.per_match(ll["individual"] - ll["base"], mm)
    fwd["TY2"] = ci.per_match(ll["role"] - ll["physical"], mm)
    fwd["TY2_role"] = ci.per_match(ll["role"] - ll["base"], mm)
    fwd["TY2_physical"] = ci.per_match(ll["physical"] - ll["base"], mm)
    v4 = s4["k"] >= cfg["eval_from"]
    p4 = ci.style_fit(s4, s4["g"], est4, alpha)
    p4s = p4 + r8.eb_effect(s4, p4, est4, hyper["TY4"]["tau2"], hyper["TY4"]["phi"], cfg)
    fwd["TY4"] = ci.per_match(loglik(p4s[v4], s4["y"][v4]) - loglik(p4[v4], s4["y"][v4]), s4["match"][v4])
    items["forward"] = {}
    for n, d in fwd.items():
        for kk, vv in d.items():
            arrays[f"fwd|{n}|{kk}"] = vv
        items["forward"][n] = {"gain": float(1000 * d["sum"].sum() / d["n"].sum()), "matches": int(len(d["match"]))}
    out_dir = ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "who.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1)
    np.savez(out_dir / "who.npz", **arrays)
    print(league, "written", out_dir, flush=True)


if __name__ == "__main__":
    main()
