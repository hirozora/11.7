"""Exploratory main-effect, interaction and type comparisons with prematch roles."""
import argparse
import json
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

import hypothesis_statistics as ci
import entity_types as r8
from pooling import forward_pool
from numeric import cv_style
from numeric import loglik
from entity_units import make, units
from common import ROOT, Z, build, load, permute_units, standardize

STATE = {}
STEPS_L = 5
TARGETS = (("long_pass", "T2a", "I2a"), ("back_pass", "T2c", "I2c"), ("forward", "T2d", "I2b"), ("cross", "T2b", None))


def style_index(spec):
    n_ent = (int(spec["unit_label"].max()) + 1) * spec["n_attr"]
    return (spec["unit_label"][spec["unit"]] * spec["n_attr"] + spec["attr"]) * Z + spec["zone"], n_ent * Z


def style_task(name):
    """Tuned style of one spec on top of the base offset stored in the spec's g."""
    spec, cfg = STATE["styles"][name], STATE["cfg"]
    idx, n = style_index(spec)
    return name, cv_style(spec["g"], spec["y"], idx, n, spec["k"] < cfg["eval_from"], spec["match"], STATE["kick"],
                          cfg["style_alpha_grid"])


def vc_job(job):
    name, b = job
    spec = STATE["est_specs"][name]
    labels = spec["unit_label"]
    if b >= 0:
        rng = np.random.default_rng([STATE["cfg"]["seed"], zlib.crc32(f"learn5|{name}|{b}".encode())])
        labels = permute_units(labels, spec["strata"], rng)
    T, V, kept, _, _ = ci.vc_stat(spec, labels)
    return name, b, T, V, kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    rng = np.random.default_rng(cfg["seed"])
    lam, ev_from = cfg["type_ridge"], cfg["eval_from"]
    ci.STATE["cfg"] = cfg
    A, meta = build(cfg)
    nR = len(meta["roles"])
    kick = A.groupby("match_id")["kickoff"].first().to_dict()
    Wc = dict(np.load(ROOT / cfg["who_path"]))
    g_who_all = np.load(ROOT / cfg["offsets_path"])["who_prior"]
    event_g = dict(np.load(ROOT / cfg["event_offsets_path"]))
    if len(g_who_all) != len(Wc["y"]) or any(len(event_g[f"team_{t}"]) != len(A) for t, _, _ in TARGETS):
        raise ValueError("offsets and rows differ in length")
    phase_a = A["phase"].map({"first": 0, "early": 1, "settled": 2}).to_numpy()
    opp_a = A["opp5"].map({"pass": 0, "duel": 1, "interception": 2, "clearance": 3, "other": 4}).to_numpy()
    U_a = standardize(A["U"].to_numpy())
    era_a, oppc_a = A["era"].to_numpy(), (A["opponent_class"] == "big_six").to_numpy().astype(int)
    out_dir = ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, summary = {}, {"styles": {}, "interactions": {}}

    def record(part, name, new, old, yy, mm, kk, extra=None):
        evl = kk >= ev_from
        pm = ci.per_match((loglik(new, yy) - loglik(old, yy))[evl], mm[evl])
        for key in ("match", "sum", "n"):
            arrays[f"{name}|{key}"] = pm[key]
        r = forward_pool([{"sum": pm["sum"], "n": pm["n"]}], cfg["bootstrap"], rng)
        summary[part][name] = {**(extra or {}), "gain": r["gain"], "ci": r["ci"], "rows": int(evl.sum())}
        print(f"{name:50s} gain {r['gain']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
              + (f" (tau_est {extra['tau_est']:.3f})" if extra else ""), flush=True)

    # who rows: causal candidates with a prior role
    keep = Wc["role_prior"] >= 0
    rc = {key: v[keep] for key, v in Wc.items()}
    act, role, y = rc["act"], rc["role_prior"], rc["y"]
    g_who = g_who_all[keep]
    match, team_i = A["match_id"].to_numpy()[act], A["team_i"].to_numpy()[act]
    zone, steps, k = A["zone"].to_numpy()[act], A["steps_code"].to_numpy()[act], A["k"].to_numpy()[act]
    slot_u, player_u = units(match, team_i, rc["cand"], role, era_a[act], oppc_a[act], nR)
    zeros = np.zeros(len(y), int)
    styles = {"who|slot": make(y, g_who, zone, k, match, slot_u, "cells", role, nR, level=zeros, L=1),
              "who|player": make(y, g_who, zone, k, match, player_u, "cells", level=zeros, L=1)}

    # event rows: actions whose actor has a prior role; actor type = the actor's prior role
    actor_row = y == 1
    a_prior = np.full(len(A), -1)
    a_prior[act[actor_row]] = role[actor_row]
    ra = np.flatnonzero(a_prior >= 0)
    a_role = a_prior[ra]
    m_a, t_a = A["match_id"].to_numpy()[ra], A["team_i"].to_numpy()[ra]
    team_u, actor_u = units(m_a, t_a, A["player_id"].to_numpy()[ra], a_role, era_a[ra], oppc_a[ra], nR)
    z_a, s_a, k_a, za = A["zone"].to_numpy()[ra], A["steps_code"].to_numpy()[ra], A["k"].to_numpy()[ra], np.zeros(len(ra), int)
    event_base = {}
    for target, _, _ in TARGETS:
        yt, gt = A[f"t_{target}"].to_numpy().astype(float)[ra], event_g[f"team_{target}"][ra]
        gt_type = gt + r8.fit_terms(gt, yt, k_a < ev_from, [(a_role * Z + z_a, nR * Z)], [], lam)
        event_base[target] = (yt, gt, gt_type)
        styles[f"{target}|team"] = make(yt, gt_type, z_a, k_a, m_a, team_u, "cells", level=za, L=1)
    STATE.update({"cfg": cfg, "styles": styles, "kick": kick})
    with Pool(cfg["workers"]) as pool:
        fitted = dict(pool.imap_unordered(style_task, list(styles), chunksize=1))
    # second-level styles: player over slot style; actor over team style
    styles2 = {"who|player_over_slot": {**styles["who|player"], "g": fitted["who|slot"]}}
    for target, _, _ in TARGETS:
        styles2[f"{target}|actor_over_team"] = make(event_base[target][0], fitted[f"{target}|team"], z_a, k_a, m_a, actor_u,
                                                    "cells", level=za, L=1)
    STATE["styles"] = styles2
    with Pool(cfg["workers"]) as pool:
        fitted.update(dict(pool.imap_unordered(style_task, list(styles2), chunksize=1)))

    record("styles", "who|slot over g", fitted["who|slot"], g_who, y, match, k)
    record("styles", "who|player over slot", fitted["who|player_over_slot"], fitted["who|slot"], y, match, k)
    for target, _, _ in TARGETS:
        yt, gt, gt_type = event_base[target]
        record("styles", f"{target}|actor type over g", gt_type, gt, yt, m_a, k_a)
        record("styles", f"{target}|team over actor type", fitted[f"{target}|team"], gt_type, yt, m_a, k_a)
        record("styles", f"{target}|actor over team", fitted[f"{target}|actor_over_team"], fitted[f"{target}|team"], yt, m_a, k_a)

    # interactions on top of the tuned main effects (who: slot + player style; event: actor type + team + actor style)
    base_who = fitted["who|player_over_slot"]
    F, Uw = standardize(rc["mins_on"]), U_a[act]
    s01 = steps <= 1
    inter = {
        "T1 slot x steps": make(y, base_who, zone, k, match, slot_u, "cells", role, nR, level=steps, L=STEPS_L),
        "I1a player x steps": make(y, base_who, zone, k, match, player_u, "cells", level=steps, L=STEPS_L),
        "I1b player x phase": make(y, base_who, zone, k, match, player_u, "cells", level=phase_a[act], L=3),
        "R3-Ea slot x opponent event": ci.subset(make(y, base_who, zone, k, match, slot_u, "cells", role, nR, level=opp_a[act], L=5), s01),
        "R3-Eb player x opponent event": ci.subset(make(y, base_who, zone, k, match, player_u, "cells", level=opp_a[act], L=5), s01),
        "R4-Eb slot x fatigue": make(y, base_who, zone, k, match, slot_u, "slope", role, nR, x=F),
        "R4-Ea player x fatigue": make(y, base_who, zone, k, match, player_u, "slope", x=F),
        "R1-Ea slot x urgency": make(y, base_who, zone, k, match, slot_u, "slope", role, nR, x=Uw),
        "R1-Eb player x urgency": make(y, base_who, zone, k, match, player_u, "slope", x=Uw),
    }
    for target, t2, i2 in TARGETS:
        yt = event_base[target][0]
        base_ev = fitted[f"{target}|actor_over_team"]
        inter[f"{t2} team x steps ({target})"] = make(yt, base_ev, z_a, k_a, m_a, team_u, "cells", level=s_a, L=STEPS_L)
        if i2:
            inter[f"{i2} actor x steps ({target})"] = make(yt, base_ev, z_a, k_a, m_a, actor_u, "cells", level=s_a, L=STEPS_L)

    # types on top of the tuned main effects (rows with recorded height and weight): player type x level (prior role and
    # left foot as cells, height and weight as slopes); TY4 adds team style type x slot role x steps; TY3a team type x steps
    attrs = {p["wyId"]: p for p in load(cfg["players_json"])}
    valid = np.array([attrs[c]["height"] > 0 and attrs[c]["weight"] > 0 for c in rc["cand"]])
    cands = np.unique(rc["cand"][valid])
    hw = pd.DataFrame({"h": [attrs[c]["height"] for c in cands], "w": [attrs[c]["weight"] for c in cands]}, index=cands)
    zs = (hw - hw.mean()) / hw.std()
    hz, wz = zs["h"].reindex(rc["cand"]).to_numpy(), zs["w"].reindex(rc["cand"]).to_numpy()
    left = np.array([attrs[c]["foot"] == "left" for c in rc["cand"]], dtype=int)
    K = cfg["team_type_K"]
    _, ttypes = r8.team_types(A, [K], cfg["seed"])
    ty_team = pd.Series(meta["teams"]).map(ttypes[K]).to_numpy()
    est_w = k < ev_from
    for label, rows, level, L, team_term in (("TY1", s01 & valid, opp_a[act], 5, False), ("TY4", valid, steps, STEPS_L, True)):
        lv, rl = level[rows], role[rows]
        cells = [(rl * L + lv, nR * L), (left[rows] * L + lv, 2 * L)]
        if team_term:
            cells.append(((ty_team[team_i[rows]] * nR + rl) * L + lv, K * nR * L))
        g_t = base_who[rows] + r8.fit_terms(base_who[rows], y[rows], est_w[rows], cells, [(hz[rows], lv, L), (wz[rows], lv, L)], lam)
        record("styles", f"{label} types over main effects", g_t, base_who[rows], y[rows], match[rows], k[rows])
        if label == "TY1":
            spec = ci.subset(make(y, base_who, zone, k, match, player_u, "cells", level=level, L=L), rows)
            inter["TY1 player x opponent event beyond types"] = {**spec, "g": g_t}
        else:
            spec = ci.subset(make(y, base_who, zone, k, match, slot_u, "cells", role, nR, level=level, L=L), rows)
            inter["TY4 slot x steps beyond types"] = {**spec, "g": g_t}
    yt = event_base["long_pass"][0]
    base_ev = fitted["long_pass|actor_over_team"]
    g3 = base_ev + r8.fit_terms(base_ev, yt, k_a < ev_from, [(ty_team[t_a] * STEPS_L + s_a, K * STEPS_L)], [], lam)
    record("styles", "TY3a team type x steps over main effects (long_pass)", g3, base_ev, yt, m_a, k_a)

    est_specs = {n: ci.subset(s, s["k"] < ev_from) for n, s in inter.items()}
    STATE["est_specs"] = est_specs
    jobs = [(n, b) for n in est_specs for b in range(-1, cfg["forward_permutations"])]
    jobs.sort(key=lambda j: -len(est_specs[j[0]]["y"]))
    res = {}
    with Pool(cfg["workers"]) as pool:
        for n, b, T, V, kept in pool.imap_unordered(vc_job, jobs, chunksize=4):
            res.setdefault(n, {"perm": {}})
            if b < 0:
                res[n].update({"T": T, "V": V, "kept": kept})
            else:
                res[n]["perm"][b] = T
    for n, s in inter.items():
        r = res[n]
        perm = np.array([r["perm"][b] for b in range(cfg["forward_permutations"])])
        tau2, phi = max(0.0, (r["T"] - perm.mean()) / r["V"]), perm.mean() / r["kept"]
        full = s["g"] + r8.eb_effect(s, s["g"], s["k"] < ev_from, tau2, phi, cfg)
        record("interactions", n, full, s["g"], s["y"], s["match"], s["k"], {"tau_est": float(np.sqrt(tau2))})
    np.savez(out_dir / "ladder.npz", **arrays)
    with open(out_dir / "ladder.json", "w", encoding="utf-8") as f:
        json.dump({"league": cfg["league"], **summary}, f, indent=1)
    print("written", out_dir / "ladder.npz", flush=True)


if __name__ == "__main__":
    main()
