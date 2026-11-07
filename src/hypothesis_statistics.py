"""Registered GLM, variance-component, type and possession statistics."""
import zlib
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import chi2
import entity_types as r8
from numeric import newton, loglik, pop_offset
from common import Z, permute_units, standardize


STATE = {}
STEPS_L = 5
BOX = (84.0, 19.0, 81.0)  # opponent's box: x >= 84 and 19 <= y <= 81
EFFECTIVE = ["short_pass", "long_pass", "cross", "through_ball", "head_pass", "carry", "take_on", "shot"]
ACTIVE = ["interception", "duel_won"]
CONTROL = ["opp_pass_lost", "opp_clearance", "after_opp_shot", "other"]
ROW_KEYS = ("y", "g", "zone", "unit", "attr", "level", "x", "k", "match", "style_cell")

def subset(spec, rows):
    return {k: (v[rows] if k in ROW_KEYS and v is not None else v) for k, v in spec.items()}


def null_fit(spec, labels):
    """g + entity style for the given labels. The style cells are the 3x3 zones, or spec["style_cell"] when a rule is
    finer than the zones. Returns the fitted probabilities and the entity of each row."""
    cfg = STATE["cfg"]
    n_ent = (int(spec["unit_label"].max()) + 1) * spec["n_attr"]
    ent = labels[spec["unit"]] * spec["n_attr"] + spec["attr"]
    cell_s, n_s = (spec["style_cell"], spec["n_style"]) if "style_cell" in spec else (spec["zone"], Z)
    style = ent * n_s + cell_s
    return expit(spec["g"] + newton(spec["g"], spec["y"], style, n_ent * n_s, cfg["alpha_beta"])[style]), ent


def score(spec, mu, ent):
    """Score statistic for entity x level cells (or random slopes) given the null fit."""
    cfg = STATE["cfg"]
    n_ent = (int(spec["unit_label"].max()) + 1) * spec["n_attr"]
    r, v = spec["y"] - mu, mu * (1 - mu)
    if spec["kind"] == "slope":
        S, V = np.bincount(ent, spec["x"] * r, n_ent), np.bincount(ent, spec["x"] ** 2 * v, n_ent)
        keep = V >= cfg["v_min"]
        return float((S[keep] ** 2 / V[keep]).sum()), float(V[keep].sum()), int(keep.sum()), None, None
    cell = ent * spec["L"] + spec["level"]
    S, V = np.bincount(cell, r, n_ent * spec["L"]), np.bincount(cell, v, n_ent * spec["L"])
    keep = V >= cfg["v_min"]
    T_cell = np.zeros(len(S))
    T_cell[keep] = S[keep] ** 2 / V[keep]
    V_cell = np.where(keep, V, 0.0)
    return (float(T_cell.sum()), float(V_cell.sum()), int(keep.sum()),
            T_cell.reshape(-1, spec["L"]).sum(0), V_cell.reshape(-1, spec["L"]).sum(0))


def vc_stat(spec, labels):
    return score(spec, *null_fit(spec, labels))


def t4_stat(spec, labels):
    """T4': team random slope on the within-team-centred active-regain indicator, on top of the P3' population part
    (position splines, active effect) plus team intercepts refitted for the given labels."""
    cfg = STATE["cfg"]
    n_ent = int(spec["unit_label"].max()) + 1
    ent = labels[spec["unit"]]
    mu = expit(spec["g"] + newton(spec["g"], spec["y"], ent, n_ent, cfg["alpha_beta"])[ent])
    x = spec["active"] - (np.bincount(ent, spec["active"], n_ent) / np.bincount(ent, None, n_ent))[ent]
    S = np.bincount(ent, x * (spec["y"] - mu), n_ent)
    V = np.bincount(ent, x * x * mu * (1 - mu), n_ent)
    keep = V >= cfg["v_min"]
    return float((S[keep] ** 2 / V[keep]).sum()), float(V[keep].sum()), int(keep.sum()), None, None


def vc_task(job):
    kind, name, b = job
    cfg = STATE["cfg"]
    spec = STATE["est_specs"][name] if kind.startswith("est") else STATE["specs"][name]
    labels = spec["unit_label"]
    if kind.endswith("perm"):
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(f"confirm|{cfg['league']}|{kind}|{name}|{b}".encode())])
        labels = permute_units(labels, spec["strata"], rng)
    stat = t4_stat if name == "T4prime" else vc_stat
    return (kind, name, b) + stat(spec, labels)


def ratio_task(b):
    """One match-bootstrap replicate for the TY1 tau ratio. Rows of a match drawn w times enter w times (the copies keep
    their player-match unit, so they share its label); type terms and entity style are refitted. Per spec: T, the mean
    T over cfg["ratio_permutations"] permutations of the replicate (its null expectation), and sum V."""
    cfg, sp = STATE["cfg"], STATE["ty1"]
    rng = np.random.default_rng([cfg["seed"], zlib.crc32(f"ratio|{cfg['league']}|{b}".encode())])
    n_m = STATE["ty1_n_matches"]
    w_row = np.bincount(rng.integers(0, n_m, n_m), minlength=n_m)[STATE["ty1_match_i"]]
    rows = np.repeat(np.arange(len(sp["y"])), w_row)
    rep = subset(sp, rows)
    cells = [(idx[rows], n) for idx, n in STATE["ty1_cells"]]
    slopes = [(x[rows], idx[rows], n) for x, idx, n in STATE["ty1_slopes"]]
    out = [b]
    for g in (rep["g"], rep["g"] + r8.fit_terms(rep["g"], rep["y"], np.ones(len(rows), bool), cells, slopes, cfg["type_ridge"])):
        spec = {**rep, "g": g}
        T, V = vc_stat(spec, spec["unit_label"])[:2]
        null = [vc_stat(spec, permute_units(spec["unit_label"], spec["strata"], rng))[0] for _ in range(cfg["ratio_permutations"])]
        out += [T, float(np.mean(null)), V]
    return tuple(out)


def glm_fit(y, X, offset, groups):
    """Logistic GLM fitted by IRLS from beta = 0 (the offset alone), cluster-robust covariance; stops if not converged
   ."""
    res = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit(
        start_params=np.zeros(X.shape[1]), cov_type="cluster", cov_kwds={"groups": groups})
    if not res.converged:
        raise RuntimeError("GLM did not converge")
    return res


def glm_item(y, X, names, offset, groups, test):
    res = glm_fit(y, X, offset, groups)
    beta = np.asarray(res.params)[test]
    cov = np.asarray(res.cov_params())[np.ix_(test, test)]
    wald = float(beta @ np.linalg.solve(cov, beta))
    return {"names": [names[j] for j in test], "beta": beta.tolist(), "cov": cov.tolist(), "wald": wald,
            "df": len(test), "p": float(chi2.sf(wald, len(test))), "n": int(len(y))}


def p1_counts(W, roles):
    cell = W["role"] * STEPS_L + W["steps"]
    n = np.bincount(cell, None, len(roles) * STEPS_L).reshape(-1, STEPS_L)
    s = np.bincount(cell, W["y"], len(roles) * STEPS_L).reshape(-1, STEPS_L)
    return {"roles": roles, "acts": s.tolist(), "opportunities": n.astype(int).tolist()}


def possessions(A):
    """Open-play possessions won actively (interception, duel) or otherwise (opponent's misplaced pass, opponent's
    clearance, after an opponent's shot, other), starting in the opponent's half. Start = the possession's first
    on-ball action: its time, x and y."""
    first = A.groupby(["match_id", "possession_id"], sort=False).head(1)
    keep = first["possession_start"].isin(ACTIVE + CONTROL).to_numpy() & (first["x"].to_numpy() >= 50)
    P = first.loc[keep, ["match_id", "possession_id", "team_id", "possession_start", "event_sec", "x", "y"]].reset_index(drop=True)
    P["active"] = P["possession_start"].isin(ACTIVE).astype(float)
    P["in_box"] = (P["x"] >= BOX[0]) & P["y"].between(BOX[1], BOX[2])
    P["dist"] = np.hypot(1.05 * (100 - P["x"]), 0.68 * (50 - P["y"]))
    P["lat"] = (P["y"] - 50).abs()
    return P


def quick_attack(A, P):
    """Among the first three effective actions (pass, carry, take-on, shot) of each possession that happen within 10 s
    of its first on-ball action: any shot or action ending in the opponent's box (quick), any shot (shot)."""
    E = A.loc[A["action"].isin(EFFECTIVE), ["match_id", "possession_id", "event_sec", "action", "end_x", "end_y"]]
    E = E.reset_index(names="row").merge(P[["match_id", "possession_id", "event_sec"]].rename(columns={"event_sec": "t0"}),
                                         on=["match_id", "possession_id"]).sort_values("row")
    E = E[E["event_sec"] - E["t0"] <= 10.0].groupby(["match_id", "possession_id"], sort=False).head(3)
    shot = E["action"].eq("shot").to_numpy()
    box = ((E["end_x"] >= BOX[0]) & E["end_y"].between(BOX[1], BOX[2])).to_numpy()
    flags = pd.DataFrame({"match_id": E["match_id"].to_numpy(), "possession_id": E["possession_id"].to_numpy(),
                          "quick": shot | box, "shot": shot}).groupby(["match_id", "possession_id"])[["quick", "shot"]].any()
    out = P[["match_id", "possession_id"]].join(flags, on=["match_id", "possession_id"])
    return out["quick"].eq(True).to_numpy(), out["shot"].eq(True).to_numpy()


def p3_glm(D, y):
    """Logistic regression: active + natural splines of distance to goal (df 4) and |y - 50| (df 3) + team fixed
    effects, cluster-robust by match. Returns the active coefficient and the linear predictor without team effects."""
    spl = np.asarray(patsy.dmatrix("cr(dist, df=4, constraints='center') + cr(lat, df=3, constraints='center') - 1", D))
    teams = pd.get_dummies(D["team_id"], drop_first=True, dtype=float).to_numpy()
    X = np.column_stack([np.ones(len(D)), D["active"].to_numpy(), spl, teams])
    res = glm_fit(y, X, np.zeros(len(D)), pd.factorize(D["match_id"])[0])
    n_pop = 2 + spl.shape[1]
    beta = np.asarray(res.params)
    var = float(np.asarray(res.cov_params())[1, 1])
    rec = {"names": ["active"], "beta": [float(beta[1])], "cov": [[var]], "wald": float(beta[1] ** 2 / var), "df": 1,
           "p": float(chi2.sf(beta[1] ** 2 / var, 1)), "n": int(len(D))}
    return rec, X[:, :n_pop] @ beta[:n_pop]


def p3_items(A, cfg):
    P = possessions(A)
    main = P[~P["in_box"]].reset_index(drop=True)
    box = P[P["in_box"]].reset_index(drop=True)
    out = {"outcome": cfg["p3prime_outcome"],
           "counts": {"active": int(main["active"].sum()), "control": int((main["active"] == 0).sum()),
                      "by_start": {k: int(v) for k, v in main["possession_start"].value_counts().items()},
                      "box_active": int(box["active"].sum()), "box_control": int((box["active"] == 0).sum())}}
    if cfg["p3prime_outcome"] == "observed":
        quick = quick_attack(A, main)[0]
        box_shot = quick_attack(A, box)[1]
    elif cfg["p3prime_outcome"] == "synthetic":
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(b"synthetic outcome")])
        quick, box_shot = rng.random(len(main)) < 0.1, rng.random(len(box)) < 0.1
    else:
        raise ValueError(f"p3prime_outcome must be observed or synthetic, got {cfg['p3prime_outcome']}")
    y = quick.astype(float)
    out["glm"], eta_pop = p3_glm(main, y)
    sens = ((main["active"] == 1) | (main["possession_start"] == "opp_pass_lost")).to_numpy()
    out["glm_control_opp_pass_lost"] = p3_glm(main[sens].reset_index(drop=True), y[sens])[0]
    # Mantel-Haenszel strata: x in [50, 66.7), [66.7, 84), [84, 100] x y thirds x team
    team_codes, _ = pd.factorize(main["team_id"])
    n_team = int(team_codes.max()) + 1
    stratum = (np.digitize(main["x"], [200 / 3, BOX[0]]) * 3 + np.digitize(main["y"], [100 / 3, 200 / 3])) * n_team + team_codes
    act = main["active"].to_numpy() == 1
    tables = np.column_stack([np.bincount(stratum, act & quick, 9 * n_team), np.bincount(stratum, act & ~quick, 9 * n_team),
                              np.bincount(stratum, ~act & quick, 9 * n_team), np.bincount(stratum, ~act & ~quick, 9 * n_team)])
    out["box_shot"] = {"active": [int(box_shot[box["active"] == 1].sum()), int((box["active"] == 1).sum())],
                       "control": [int(box_shot[box["active"] == 0].sum()), int((box["active"] == 0).sum())]}
    tm = main.groupby(["match_id", "team_id"], sort=False).ngroup().to_numpy()
    tm_label = pd.Series(team_codes).groupby(tm).first().sort_index().to_numpy()
    t4 = {"y": y, "g": eta_pop, "active": main["active"].to_numpy(), "unit": tm, "unit_label": tm_label,
          "strata": [np.arange(len(tm_label))]}
    return out, tables[tables.sum(1) > 0], t4


def style_fit(spec, base, est, alpha):
    n_ent = (int(spec["unit_label"].max()) + 1) * spec["n_attr"]
    sidx = (spec["unit_label"][spec["unit"]] * spec["n_attr"] + spec["attr"]) * Z + spec["zone"]
    return base + newton(base[est], spec["y"][est], sidx[est], n_ent * Z, alpha)[sidx]


def per_match(d, match):
    s = pd.Series(d).groupby(match).agg(["sum", "size"])
    return {"match": s.index.to_numpy(), "sum": s["sum"].to_numpy(), "n": s["size"].to_numpy()}
