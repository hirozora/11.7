"""Compute all registered event-side items and possession outcomes in one run."""
import argparse
import json
from multiprocessing import Pool

import numpy as np

import hypothesis_statistics as ci
import entity_types as r8
from event_hypotheses import prepare, population
from numeric import loglik
from common import ROOT, load

EVENT_VC = ["R1-Ec", "R1-Ed", "R2-Ea", "R2-Eb", "R5-E", "T2a", "T2b", "T2c", "T2d", "T3", "I2a", "I2b", "I2c", "C1", "TY3b"]
EVENT_GLM = ["R1-Pb", "R1-Pc", "R2-Pa", "R2-Pb", "R5-P", "P2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    league, lam, alpha = cfg["league"], cfg["type_ridge"], cfg["alpha_beta"]
    run = prepare(cfg)
    A = run["A"]
    glm = population(A, cfg)
    if cfg["mode"] == "reference":
        (ROOT / cfg["event_reference_json"]).parent.mkdir(parents=True, exist_ok=True)
        with open(ROOT / cfg["event_reference_json"], "w", encoding="utf-8") as f:
            json.dump({"glm_beta": {n: glm[n]["beta"] for n in EVENT_GLM}}, f, indent=1)
        print(league, "event-side reference written", cfg["event_reference_json"], flush=True)
        return
    if cfg["mode"] != "full":
        raise ValueError(f"mode must be reference or full, got {cfg['mode']}")
    out_dir = ROOT / cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    items = {"league": league, "matches": int(A["match_id"].nunique()), "teams": len(run["meta"]["teams"]),
             "rows": {"team": int(len(A))}, "glm": {n: glm[n] for n in EVENT_GLM}}
    arrays = {}

    # P3', T4' and B on the possession-level acquisition mode
    A_acq = A.assign(possession_start=A["acquisition_mode"])
    specs = {n: run["specs"][n] for n in EVENT_VC}
    items["p3prime"], arrays["mh_tables"], specs["T4prime"] = ci.p3_items(A_acq, cfg)
    s0 = A["steps_code"].to_numpy() == 0
    high = A["acquisition_mode"].isin(["interception", "duel_won"]).to_numpy() & (A["poss_start_x"].to_numpy() >= cfg["high_regain_x"])
    f = A["t_forward"].to_numpy()
    items["descriptive_B"] = {"high_regain": [int(f[s0 & high].sum()), int((s0 & high).sum())],
                              "other_first_actions": [int(f[s0 & ~high].sum()), int((s0 & ~high).sum())]}
    print(league, "population layer done; specs:", {n: len(s["y"]) for n, s in specs.items()}, flush=True)

    ci.STATE.update({"cfg": cfg, "specs": specs, "est_specs": {}})
    B = cfg["permutations"]
    jobs = [("obs", n, 0) for n in specs] + [("perm", n, b) for n in specs for b in range(B)]
    jobs.sort(key=lambda j: -len(specs[j[1]]["y"]))
    obs, perm = {}, {n: np.full(B, np.nan) for n in specs}
    level_sum = {n: 0.0 for n in specs}
    with Pool(cfg["workers"]) as pool:
        for i, (kind, n, b, T, V, kept, Tl, Vl) in enumerate(pool.imap_unordered(ci.vc_task, jobs, chunksize=4)):
            if kind == "obs":
                obs[n] = (T, V, kept, Tl, Vl)
            else:
                perm[n][b] = T
                if Tl is not None:
                    level_sum[n] = level_sum[n] + Tl
            if (i + 1) % 5000 == 0 or i + 1 == len(jobs):
                print(f"{league}: {i + 1}/{len(jobs)} fits", flush=True)
    items["vc"] = {}
    for n in specs:
        T, V, kept, Tl, Vl = obs[n]
        if np.isnan(perm[n]).any():
            raise RuntimeError(f"missing permutations for {n}")
        arrays[f"perm|{n}"] = perm[n]
        items["vc"][n] = {"T": T, "V": V, "kept": kept, "perm_mean": float(perm[n].mean()), "perm_sd": float(perm[n].std(ddof=1)),
                          "T_level": None if Tl is None else Tl.tolist(), "V_level": None if Vl is None else Vl.tolist(),
                          "perm_level_mean": None if Tl is None else (level_sum[n] / B).tolist(), "rows": int(len(specs[n]["y"]))}

    s3 = specs["T2a"]
    e3, v3 = s3["k"] <= 19, s3["k"] >= cfg["eval_from"]
    b3 = ci.style_fit(s3, s3["g"], e3, alpha)
    t3 = ci.style_fit(s3, s3["g"] + r8.fit_terms(s3["g"], s3["y"], e3, run["ty3_cells"], [], lam), e3, alpha)
    d = ci.per_match(loglik(t3[v3], s3["y"][v3]) - loglik(b3[v3], s3["y"][v3]), s3["match"][v3])
    for k, v in d.items():
        arrays[f"fwd|TY3a|{k}"] = v
    items["forward"] = {"TY3a": {"gain": float(1000 * d["sum"].sum() / d["n"].sum()), "matches": int(len(d["match"]))}}
    items["types"] = run["types"]
    with open(out_dir / "event.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1)
    np.savez(out_dir / "event.npz", **arrays)
    print(league, "written", out_dir, flush=True)


if __name__ == "__main__":
    main()
