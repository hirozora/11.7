"""Load action tables and form cross-fitted population features."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from event_schema import FINE_ROLES
from features import CAT_CTX, NUM_CTX, context, derive
from numeric import team_match_index


ROOT = Path(__file__).resolve().parents[1]
Z = 9
STEPS = ["0", "1", "2", "3-5", "6+"]
GROUP_OF = {"GK": "GK", "deep/halfL": "CB", "deep/centre": "CB", "deep/halfR": "CB",
            "deep/wideL": "FB", "deep/wideR": "FB", "backmid/wideL": "FB", "backmid/wideR": "FB",
            "backmid/halfL": "DM", "backmid/centre": "DM", "backmid/halfR": "DM",
            "mid/halfL": "CM", "mid/centre": "CM", "mid/halfR": "CM",
            "mid/wideL": "WIDE", "mid/wideR": "WIDE", "front/wideL": "WIDE", "front/wideR": "WIDE",
            "front/halfL": "FW", "front/centre": "FW", "front/halfR": "FW"}
GROUPS = ["CM", "GK", "CB", "FB", "DM", "WIDE", "FW"]  # CM is the reference level
OPP5 = {"duel": "duel", "interception": "interception", "pass": "pass", "clearance": "clearance"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def standardize(x):
    return (x - x.mean()) / x.std()


def team_features(A):
    ctx = context(A)
    return np.column_stack([ctx[c] for c in CAT_CTX + NUM_CTX]), [True] * len(CAT_CTX) + [False] * len(NUM_CTX)


def crossfit_g(X, y, cat, match, cfg, folds=5):
    """Out-of-fold logit of the population model (no entity information), folds by match."""
    ms = np.unique(match)
    rng = np.random.default_rng(cfg["seed"])
    fold_of = dict(zip(rng.permutation(ms), np.arange(len(ms)) % folds))
    fold = pd.Series(match).map(fold_of).to_numpy()
    g = np.zeros(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=200,
                                           l2_regularization=1.0, categorical_features=cat, early_stopping=False,
                                           random_state=cfg["seed"]).fit(X[tr], y[tr])
        p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        g[te] = np.log(p / (1 - p))
    return g


def permute_units(labels, strata_idx, rng):
    out = labels.copy()
    for idx in strata_idx:
        out[idx] = labels[rng.permutation(idx)]
    return out


def build(cfg):
    """Action rows and fixed role vocabulary; candidate arrays are loaded explicitly by each analysis."""
    A = derive(pd.read_parquet(ROOT / cfg["actions_path"]), cfg)
    A["k"] = team_match_index(A)
    A["U"] = -A["score_diff"].clip(-2, 2) * A["minute"].clip(upper=95) / 90.0
    A["opp5"] = A["opp_last_event"].map(OPP5).fillna("other")
    A["steps_code"] = A["steps_since_change"].map({s: i for i, s in enumerate(STEPS)})
    roles = sorted(set(FINE_ROLES) - {"unassigned"})
    teams = sorted(A["team_id"].unique())
    A["team_i"] = A["team_id"].map({t:i for i,t in enumerate(teams)})
    return A, {"roles": roles, "teams": teams}
