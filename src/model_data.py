"""Loading of the scopes' model arrays (src/model_inputs.py) for training and evaluation, and the cross-fitting folds.

All scopes are concatenated in the order given; a target's global index is its position in that concatenation.
Inputs and labels stay in separate dictionaries: training code reads labels only to form the loss.
"""
import json

import numpy as np
import pandas as pd

from common import ROOT, load

INPUT_KEYS = ["win_cat", "win_num", "win_valid", "ctx_cat", "ctx_num", "urgency", "cand_cat", "cand_num", "cand_mask",
              "cand_player"]
META_COLUMNS = ["match_id", "team_id", "period", "clock", "minute", "k", "seq"]
SHARED_META = ["window", "candidates", "sub_event_names", "vocab", "groups", "roles", "cand_num"]


def scope_dirs(scope_configs):
    return [ROOT / load(ROOT / path)["out_dir"] for path in scope_configs]


def load_meta(scope_configs):
    """Per-target meta (with block 0 = training, 1 = validation, and the league index) and labels of the scopes."""
    labels, meta = [], []
    for league_index, d in enumerate(scope_dirs(scope_configs)):
        labels.append(np.load(d / "arrays_label.npz")["actor_slot"])
        tm = pd.read_parquet(d / "targets_meta.parquet")
        block = tm["block"].map({"model_train": 0, "model_validation": 1})
        if block.isna().any():
            raise ValueError(f"{d}: targets outside the model split")
        meta.append(tm[META_COLUMNS].assign(block=block.astype(np.int8), league=league_index))
    return {"actor_slot": np.concatenate(labels)}, pd.concat(meta, ignore_index=True)


def load_scopes(scope_configs):
    """Concatenated inputs, labels and per-target meta of the given model scopes."""
    inputs, shared = {k: [] for k in INPUT_KEYS}, None
    for d in scope_dirs(scope_configs):
        arrays_meta = load(d / "arrays_meta.json")
        this = {k: arrays_meta[k] for k in SHARED_META}
        if shared is None:
            shared = this
        elif this != shared:
            raise ValueError(f"{d}: array vocabulary or shapes differ from the first scope")
        A = np.load(d / "arrays_in.npz")
        for k in INPUT_KEYS:
            inputs[k].append(A[k])
    labels, meta = load_meta(scope_configs)
    return {k: np.concatenate(v) for k, v in inputs.items()}, labels, meta, shared


def folds(meta, n_folds, seed):
    """Fold of each target: training matches are permuted with the seed and dealt round-robin into n_folds folds;
    validation targets get -1."""
    train_matches = np.sort(meta.loc[meta["block"] == 0, "match_id"].unique())
    order = np.random.default_rng(seed).permutation(train_matches)
    fold_of = dict(zip(order, np.arange(len(order)) % n_folds))
    fold = meta["match_id"].map(fold_of).fillna(-1).astype(int).to_numpy()
    if (fold[meta["block"].to_numpy() == 0] < 0).any():
        raise ValueError("a training target has no fold")
    return np.where(meta["block"].to_numpy() == 1, -1, fold)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
