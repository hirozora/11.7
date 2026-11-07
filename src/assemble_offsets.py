"""Assemble the cross-fitted M1 offsets of one seed (docs/MODEL_PLAN.md, section 3).

Training targets take the scores and context vector of the fold run that did not see their match; validation targets
those of the full run. Every target must be predicted exactly once, by the run its fold calls for. Writes
<runs>/offsets.npz (scores (N, C) with -inf padding, context (N, 64), source: fold index, -1 for the full run) and
<runs>/offsets.json (per-run best epoch and validation NLL; out-of-fold and validation NLL of the assembled offsets).
"""
import argparse

import numpy as np
import torch

from common import ROOT, load
from model_data import folds, load_meta, write_json
from seq_model import actor_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    runs = ROOT / args.runs
    Y, meta = load_meta(cfg["scopes"])
    fold = folds(meta, cfg["folds"], cfg["fold_seed"])
    N = len(meta)
    scores = context = None
    source = np.full(N, -2)
    info = {}
    for f in list(range(cfg["folds"])) + ["full"]:
        P = np.load(runs / f"fold_{f}" / "predictions.npz")
        rows = P["rows"]
        expected = np.flatnonzero(fold == f) if f != "full" else np.flatnonzero(meta["block"].to_numpy() == 1)
        if not np.array_equal(rows, expected):
            raise ValueError(f"fold {f}: predicted rows differ from the rows its fold calls for")
        if scores is None:
            scores = np.full((N, P["scores"].shape[1]), -np.inf, np.float32)
            context = np.zeros((N, P["context"].shape[1]), np.float32)
        if (source[rows] != -2).any():
            raise ValueError(f"fold {f}: targets predicted twice")
        scores[rows], context[rows] = P["scores"], P["context"]
        source[rows] = f if f != "full" else -1
        run = load(runs / f"fold_{f}" / "run.json")
        info[str(f)] = {k: run[k] for k in ("best_epoch", "best_validation_nll", "train_targets", "wall_seconds")}
    if (source == -2).any():
        raise ValueError("targets without a prediction")
    actor = Y["actor_slot"]
    reach = actor >= 0
    nll = actor_loss(torch.as_tensor(scores[reach]), torch.as_tensor(actor[reach])).numpy()
    block = meta["block"].to_numpy()[reach]
    info["assembled"] = {"out_of_fold_training_nll": float(nll[block == 0].mean()), "validation_nll": float(nll[block == 1].mean()),
                         "targets": int(N), "reachable": int(reach.sum())}
    np.savez(runs / "offsets.npz", scores=scores, context=context, source=source)
    write_json(runs / "offsets.json", info)
    print(info["assembled"], flush=True)


if __name__ == "__main__":
    main()
