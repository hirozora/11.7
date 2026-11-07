"""Regularized type effects and team types estimated from the first 19 matches."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from numeric import newton, newton_slope


def fit_terms(base, y, mask, cells, slopes, lam, rounds=3):
    """Backfitted ridge-logistic fixed effects: cells [(idx, n)], slopes [(x, idx, n)]; returns the total contribution."""
    parts = [np.zeros(len(y)) for _ in range(len(cells) + len(slopes))]
    for _ in range(rounds):
        for j, (idx, n) in enumerate(cells):
            other = base + sum(parts) - parts[j]
            theta = newton(other[mask], y[mask], idx[mask], n, lam)
            parts[j] = theta[idx]
        for j, (x, idx, n) in enumerate(slopes):
            jj = len(cells) + j
            other = base + sum(parts) - parts[jj]
            b = newton_slope(other[mask], y[mask], x[mask], idx[mask], n, lam)
            parts[jj] = b[idx] * x
    return sum(parts)


def team_types(A, k_values, seed):
    """K-means team style types from each team's first 19 matches."""
    E = A[A["k"] <= 19]
    poss = E.groupby(["team_id", "match_id", "possession_id"]).agg(n=("action", "size"), start=("possession_start", "first"),
                                                                  sx=("poss_start_x", "first"))
    active = poss["start"].isin(["interception", "duel_won"])
    per_match = E.groupby(["match_id", "team_id"]).size().rename("own").reset_index()
    tot = per_match.groupby("match_id")["own"].transform("sum")
    per_match["share"] = per_match["own"] / tot
    feats = pd.DataFrame({
        "long_pass": E.groupby("team_id")["t_long_pass"].mean(), "cross": E.groupby("team_id")["t_cross"].mean(),
        "back_pass": E.groupby("team_id")["t_back_pass"].mean(), "forward": E.groupby("team_id")["t_forward"].mean(),
        "final_third": (E["x"] >= 200 / 3).groupby(E["team_id"]).mean(),
        "possession_length": poss.groupby("team_id")["n"].mean(),
        "active_regain_share": active.groupby("team_id").mean(),
        "regain_height": poss[active].groupby("team_id")["sx"].mean(),
        "possession_share": per_match.groupby("team_id")["share"].mean()})
    Xs = (feats - feats.mean()) / feats.std()
    types = {}
    for K in k_values:
        km = KMeans(n_clusters=K, n_init=50, random_state=seed).fit(Xs.to_numpy())
        types[K] = dict(zip(feats.index, km.labels_))
    return feats, types


def eb_effect(spec, base, est, tau2, phi, cfg):
    """Shrunken entity effects (cells or slopes) fitted on the estimation rows on top of `base`, for all rows."""
    if tau2 <= 0:
        return np.zeros(len(spec["y"]))
    n_ent = (int(spec["unit_label"].max()) + 1) * spec["n_attr"]
    ent = spec["unit_label"][spec["unit"]] * spec["n_attr"] + spec["attr"]
    lam = phi / tau2
    if spec["kind"] == "slope":
        b = newton_slope(base[est], spec["y"][est], spec["x"][est], ent[est], n_ent, lam)
        return b[ent] * spec["x"]
    cidx = ent * spec["L"] + spec["level"]
    return newton(base[est], spec["y"][est], cidx[est], n_ent * spec["L"], lam)[cidx]
