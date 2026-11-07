"""Pool current actor and event statistics; apply family-wise Holm and the confirmation eligibility ceiling."""
import argparse
import json
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.contingency_tables import StratifiedTable
from common import ROOT, load
from numeric import holm
from pooling import glm_pool, vc_pool, forward_pool, ratio_of, p1_profile, fmt_est

GLM_ITEMS = ["R1-Pa", "R1-Pb", "R1-Pc", "R2-Pa", "R2-Pb", "R3-P", "R4-P", "R5-P", "P2"]
VC_ITEMS = ["R1-Ea", "R1-Eb", "R1-Ec", "R1-Ed", "R2-Ea", "R2-Eb", "R3-Ea", "R3-Eb", "R4-Ea", "R4-Eb", "R5-E",
            "T1", "T2a", "T2b", "T2c", "T2d", "T3", "T4prime", "I1a", "I1b", "I2a", "I2b", "I2c", "TY3b", "C1", "C2"]
FWD_ITEMS = ["TY3a", "TY4"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    rng = np.random.default_rng(cfg["seed"])
    fam = load(ROOT / cfg["family_json"])
    ref = load(ROOT / cfg["reference_json"])
    eligible = load(ROOT / cfg["eligibility_json"])["eligible"]
    leagues = cfg["leagues"]
    I, R = [], []
    for lg in leagues:
        path = ROOT / cfg["results_dirs"][lg]
        who, event = load(path / "who.json"), load(path / "event.json")
        if who["league"] != lg or event["league"] != lg:
            raise ValueError(f"Unexpected league in {path}")
        I.append({**event, "rows": {**event["rows"], **who["rows"]},
                  "glm": {**event["glm"], **who["glm"]}, "vc": {**event["vc"], **who["vc"]}, "p1": who["p1"]})
        with np.load(path / "who.npz") as w, np.load(path / "event.npz") as e:
            if set(w.files) & set(e.files):
                raise ValueError("Actor and event arrays have duplicate keys")
            R.append({**dict(w), **dict(e)})
    def perms(i, n):
        return R[i][f"perm|{n}"]
    G = I
    res = {}

    for n in GLM_ITEMS:
        res[n] = {"kind": "glm", **glm_pool([g["glm"][n] for g in G], ref["glm_beta"][n])}
    res["P3prime"] = {"kind": "glm", **glm_pool([g["p3prime"]["glm"] for g in G], [1.0])}
    for n in VC_ITEMS:
        res[n] = {"kind": "vc", **vc_pool([it["vc"][n] for it in I], [perms(i, n) for i in range(len(I))])}

    def fwd(n):
        return forward_pool([{k: r[f"fwd|{n}|{k}"] for k in ("sum", "n")} for r in R], cfg["bootstrap"], rng)

    for n in FWD_ITEMS:
        res[n] = {"kind": "forward", **fwd(n)}

    # TY1: intersection-union
    c1, c2 = fwd("TY1a"), fwd("TY1b")
    t2, t2b, per = {}, {}, []
    for k in ("none", "type"):
        vs = [it["vc"][f"TY1|{k}"] for it in I]
        t2[k] = sum(v["T"] - perms(i, f"TY1|{k}").mean() for i, v in enumerate(vs)) / sum(v["V"] for v in vs)
        t2b[k] = sum(r[f"ratio|{k}|T"] - r[f"ratio|{k}|null"] for r in R) / sum(r[f"ratio|{k}|V"] for r in R)
    ratio = float(ratio_of(np.array(t2["type"]), np.array(t2["none"])))
    ratio_b = ratio_of(t2b["type"], t2b["none"])
    p3 = float((1 + (ratio_b >= 1).sum()) / (1 + len(ratio_b)))
    for i, (it, r) in enumerate(zip(I, R)):
        tl = {k: (it["vc"][f"TY1|{k}"]["T"] - perms(i, f"TY1|{k}").mean()) / it["vc"][f"TY1|{k}"]["V"] for k in ("none", "type")}
        rb = ratio_of((r["ratio|type|T"] - r["ratio|type|null"]) / r["ratio|type|V"], (r["ratio|none|T"] - r["ratio|none|null"]) / r["ratio|none|V"])
        per.append({"ratio": float(ratio_of(np.array(tl["type"]), np.array(tl["none"]))),
                    "ratio_ci": np.quantile(rb, [0.025, 0.975]).tolist(), "tau_none": float(np.sqrt(max(tl["none"], 0.0))),
                    "tau_type": float(np.sqrt(max(tl["type"], 0.0)))})
    agree = [l1["agrees"] and l2["agrees"] and lr["ratio"] < 1 for l1, l2, lr in zip(c1["leagues"], c2["leagues"], per)]
    res["TY1"] = {"kind": "iut", "p": max(c1["p"], c2["p"], p3), "p_conditions": [c1["p"], c2["p"], p3],
                  "type_gain": c1, "type_minus_individual": c2, "ratio": ratio, "ratio_ci": np.quantile(ratio_b, [0.025, 0.975]).tolist(),
                  "tau_none": float(np.sqrt(max(t2["none"], 0.0))), "tau_type": float(np.sqrt(max(t2["type"], 0.0))),
                  "direction_ok": ratio <= cfg["ty1_ratio_max"], "leagues": per, "consistent": int(sum(agree))}

    # P1: profile of fine-role involvement at steps 6+ vs 0
    roles = I[0]["p1"]["roles"]
    if any(it["p1"]["roles"] != roles for it in I) or roles != ref["p1_roles"]:
        raise ValueError("fine roles differ from the England reference")
    refp = np.array(ref["p1_profile"])
    prof = [p1_profile(it["p1"]["acts"], it["p1"]["opportunities"]) for it in I]
    pooled = p1_profile(sum(np.array(it["p1"]["acts"]) for it in I), sum(np.array(it["p1"]["opportunities"]) for it in I))
    rho = float(spearmanr(pooled, refp)[0])
    perm_rho = np.array([spearmanr(rng.permutation(pooled), refp)[0] for _ in range(cfg["p1_permutations"])])
    rho_l = [float(spearmanr(p, refp)[0]) for p in prof]
    res["P1"] = {"kind": "profile", "rho": rho, "p": float((1 + (perm_rho >= rho).sum()) / (1 + len(perm_rho))),
                 "direction_ok": rho >= cfg["p1_threshold"], "profile": dict(zip(roles, pooled.tolist())),
                 "leagues": [{"rho": r} for r in rho_l], "consistent": int(sum(r >= cfg["p1_threshold"] for r in rho_l))}

    # P3' sensitivity analyses and the separately reported box starts
    sens = glm_pool([g["p3prime"]["glm_control_opp_pass_lost"] for g in G], [1.0])
    st = StratifiedTable([t.reshape(2, 2) for r in R for t in r["mh_tables"]])
    res["P3prime"]["sensitivity"] = {
        "control_opp_pass_lost": {k: sens[k] for k in ("or", "or_ci", "p", "consistent")},
        "mantel_haenszel": {"or": float(st.oddsratio_pooled), "or_ci": [float(x) for x in st.oddsratio_pooled_confint()],
                            "p": float(st.test_null_odds().pvalue)},
        "box_starts_shot_rate": {g: [sum(it["p3prime"]["box_shot"][g][0] for it in I), sum(it["p3prime"]["box_shot"][g][1] for it in I)]
                                 for g in ("active", "control")},
        "counts": [it["p3prime"]["counts"] for it in I]}

    # ---------- families: Holm within each, criteria, consistency ----------
    for role, members in fam["families"].items():
        reject = holm(np.array([res[n]["p"] for n in members]), fam["family_alpha"])
        for n, rj in zip(members, reject):
            r = res[n]
            r["role"] = role
            r["holm_reject"] = bool(rj)
            r["meets_criteria"] = bool(rj and r["direction_ok"] and r["consistent"] >= fam["min_consistent"])
            r["confirmed"] = bool(eligible[n] and r["meets_criteria"])
    for n in res:
        if "role" not in res[n]:
            res[n]["role"] = "secondary"
    # ---------- secondary ----------
    sec = {"TY2": fwd("TY2"), "TY2_role": fwd("TY2_role"), "TY2_physical": fwd("TY2_physical"),
           "TY1_individual": fwd("TY1_individual")}
    t5 = {}
    for n in ("T2a", "T2b", "T2c", "T2d"):
        ex = [np.array(it["vc"][n]["T_level"]) - np.array(it["vc"][n]["perm_level_mean"]) for it in I]
        vl = [np.array(it["vc"][n]["V_level"]) for it in I]
        share = [float(e[:3].sum() / e.sum()) for e in ex]
        vshare = [float(v[:3].sum() / v.sum()) for v in vl]
        t5[n] = {"excess_share_steps_0_2": float(sum(e[:3].sum() for e in ex) / sum(e.sum() for e in ex)),
                 "V_share_steps_0_2": float(sum(v[:3].sum() for v in vl) / sum(v.sum() for v in vl)),
                 "leagues": [{"excess_share": a, "V_share": b} for a, b in zip(share, vshare)]}
    sec["T5"] = t5
    hb = [it["descriptive_B"]["high_regain"] for it in I]
    ob = [it["descriptive_B"]["other_first_actions"] for it in I]
    sec["B"] = {"high_regain_forward_rate": sum(h[0] for h in hb) / sum(h[1] for h in hb),
                "other_first_actions_forward_rate": sum(o[0] for o in ob) / sum(o[1] for o in ob),
                "leagues": [{"high": h[0] / h[1], "other": o[0] / o[1]} for h, o in zip(hb, ob)]}
    (ROOT / cfg["out_json"]).parent.mkdir(parents=True, exist_ok=True)
    out = {"leagues": leagues, "families": fam["families"], "items": res, "secondary": sec,
           "analysis_status": "reanalysis; no new confirmation eligible",
           "permutations": {lg: len(perms(i, "T1")) for i, lg in enumerate(leagues)},
           "rows": {lg: it["rows"] for lg, it in zip(leagues, I)}, "matches": {lg: it["matches"] for lg, it in zip(leagues, I)}}
    with open(ROOT / cfg["out_json"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    write_md(out, ROOT / cfg["out_md"])
    print("written", cfg["out_json"], cfg["out_md"], flush=True)

def write_md(out, path):
    lines = ["# Registered-hypothesis analysis: current results", "",
             f"Leagues: {', '.join(out['leagues'])}. Matches: {out['matches']}.", "",
             "| item | role | pooled estimate | p | Holm | criteria | consistent leagues | decision |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in out["items"].items():
        holm_s = ("reject" if r.get("holm_reject") else "-") if r["role"] != "secondary" else "n/a"
        if r["role"] != "secondary":
            decision = "confirmed" if r["confirmed"] else "not confirmed"
        else:
            decision = "secondary"
        lines.append(f"| {n} | {r['role']} | {fmt_est(r)} | {r['p']:.4g} | {holm_s} | {'yes' if r['direction_ok'] else 'no'} | "
                     f"{r['consistent']}/{len(out['leagues'])} | {decision} |")
    lines += ["", "Per-league estimates are in the JSON file. Confirmation also requires the eligibility ceiling in configs/confirmation_eligibility.json."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
