"""Team-match and player-match permutation units."""
import numpy as np
import pandas as pd


def units(match, team_i, ent_id, prior_role, era, opp, nR):
    """Team-match units (slots) and player-match units with their strata (team x prior role x half x opponent class)."""
    tm_frame = pd.DataFrame({"m": match, "t": team_i})
    tm = tm_frame.groupby(["m", "t"]).ngroup().to_numpy()
    tm_label = tm_frame.groupby(["m", "t"])["t"].first().to_numpy()
    pm_frame = pd.DataFrame({"m": match, "p": ent_id, "t": team_i, "r": prior_role, "e": era, "o": opp})
    pm = pm_frame.groupby(["m", "p"]).ngroup().to_numpy()
    first = pm_frame.groupby(["m", "p"]).first()
    pidx = {p: i for i, p in enumerate(sorted(set(ent_id)))}
    stratum = ((first["t"].to_numpy() * nR + first["r"].to_numpy()) * 2 + first["e"].to_numpy()) * 2 + first["o"].to_numpy()
    return ({"unit": tm, "unit_label": tm_label, "strata": [np.arange(len(tm_label))]},
            {"unit": pm, "unit_label": first.index.get_level_values("p").map(pidx).to_numpy(),
             "strata": [np.flatnonzero(stratum == s) for s in np.unique(stratum)]})


def make(y, g, zone, k, match, ent, kind, attr=None, n_attr=1, x=None, level=None, L=None):
    return {"y": y, "g": g, "zone": zone, "k": k, "match": match, **ent, "attr": np.zeros(len(y), int) if attr is None else attr,
            "n_attr": n_attr, "kind": kind, "x": x, "level": level, "L": L}
