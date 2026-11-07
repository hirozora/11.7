"""Scores of the targets of some scopes, cut from a scores file over more scopes (the England-only control,
docs/MODEL_PLAN.md, section 3).

--scores holds scores (N, C) over --from-scopes (concatenated in that order, src/model_data.py); the output keeps the
targets of --to-scopes, matched by (match_id, seq), and must cover each of them exactly once.
"""
import argparse

import numpy as np
import pandas as pd

from common import ROOT
from model_data import load_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--from-scopes", nargs="+", required=True)
    parser.add_argument("--to-scopes", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    S = np.load(ROOT / args.scores)["scores"]
    _, src = load_meta(args.from_scopes)
    _, dst = load_meta(args.to_scopes)
    if len(src) != len(S):
        raise ValueError("scores and --from-scopes have different numbers of targets")
    key = pd.MultiIndex.from_frame(src[["match_id", "seq"]])
    if not key.is_unique:
        raise ValueError("(match_id, seq) does not identify the targets")
    rows = key.get_indexer(pd.MultiIndex.from_frame(dst[["match_id", "seq"]]))
    if (rows < 0).any() or len(np.unique(rows)) != len(rows):
        raise ValueError("--to-scopes targets are not covered exactly once")
    for col in ("team_id", "period", "clock", "block"):
        if not np.array_equal(src[col].to_numpy()[rows], dst[col].to_numpy()):
            raise ValueError(f"matched targets differ in {col}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, scores=S[rows])
    print("written", out, len(rows), "targets", flush=True)


if __name__ == "__main__":
    main()
