"""Checks of the fixed-shape model arrays (src/model_inputs.py) against independent slow reconstructions from the
model-view tables. Stops at the first failure; writes the results into the scope's checks_inputs.json.
  12 window tokens (side, frame-converted start, start validity, within-period dt, position) for sampled targets
  13 own-action pooling for sampled candidate rows: only the candidate's on-ball actions before the target
  14 candidate slots, masks and state columns equal candidates_in
  15 the input arrays contain no label array
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT, load
from event_schema import ACTIONS, ACTION_OF

INPUT_KEYS = {"win_cat", "win_num", "win_valid", "ctx_cat", "ctx_num", "urgency", "cand_cat", "cand_num", "cand_mask",
              "cand_player", "cand_slot_rows"}


def check(results, number, ok, detail):
    results[str(number)] = {"passed": bool(ok), "detail": detail}
    print(f"check {number}: {'passed' if ok else 'FAILED'} {detail}", flush=True)
    if not ok:
        raise SystemExit(f"check {number} failed: {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    d = ROOT / cfg["out_dir"]
    rng = np.random.default_rng(args.seed)
    A = dict(np.load(d / "arrays_in.npz"))
    E = pd.read_parquet(d / "events_in.parquet")
    T = pd.read_parquet(d / "targets_in.parquet")
    Tm = pd.read_parquet(d / "targets_meta.parquet")
    W = np.load(d / "windows.npy")
    C = dict(np.load(d / "candidates_in.npz"))
    K = W.shape[1]
    row_of = {(m, s): r for r, (m, s) in enumerate(zip(E["match_id"], E["seq"]))}
    ev = E.to_dict("list")
    results = {}

    bad = 0
    sample = rng.choice(len(T), size=min(1000, len(T)), replace=False)
    for i in sample:
        m, team, per_t, clk_t = T["match_id"].iat[i], T["team_id"].iat[i], Tm["period"].iat[i], Tm["clock"].iat[i]
        for k, s in enumerate(W[i]):
            if s < 0:
                bad += bool(A["win_valid"][i, k]) or A["win_cat"][i, k, 2] != 0
                continue
            r = row_of[(m, s)]
            own = ev["team_id"][r] == team
            ok_xy = ev["xy_valid"][r]
            x = (ev["x"][r] if own else 100 - ev["x"][r]) if ok_xy else 0.0
            y = (ev["y"][r] if own else 100 - ev["y"][r]) if ok_xy else 0.0
            same = ev["period"][r] == per_t
            dt = np.log1p(max(clk_t - ev["clock"][r], 0.0)) / 5 if same else 0.0
            ref = np.array([x / 100, y / 100, float(ok_xy), dt, float(same), k / K])
            bad += not np.allclose(A["win_num"][i, k], ref, atol=1e-5)
            bad += A["win_cat"][i, k, 2] != (1 if own else 2)
    check(results, 12, bad == 0, f"window tokens of {len(sample)} targets rebuilt from events_in; mismatches {bad}")

    by_match = T.groupby("match_id").indices
    performer = {(m, s): p for m, s, p in zip(E["match_id"], E["seq"], E["player_id"])}
    kind = {(m, s): (a, b) for m, s, a, b in zip(E["match_id"], E["seq"], E["event_name"], E["sub_event_name"])}
    seq_t, x_t, y_t, clk = T["seq"].to_numpy(), T["x"].to_numpy(), T["y"].to_numpy(), Tm["clock"].to_numpy()
    bad = 0
    rows = rng.choice(len(C["act"]), size=min(1000, len(C["act"])), replace=False)
    for r in rows:
        i, p = C["act"][r], C["cand"][r]
        m = T["match_id"].iat[i]
        prev = [j for j in by_match[m] if seq_t[j] < seq_t[i] and performer[(m, seq_t[j])] == p][-32:]
        if prev:
            oh = np.array([np.eye(len(ACTIONS))[ACTIONS.index(ACTION_OF[kind[(m, seq_t[j])]])] for j in prev])
            F = np.concatenate([oh, np.stack([x_t[prev], y_t[prev]], 1) / 100], 1)
            mdt = np.log1p(max(clk[i] - clk[prev].mean(), 0.0)) / 8
            ref = np.concatenate([F.mean(0), [mdt, len(prev) / 32], F[-1]])
        else:
            ref = np.zeros(26)
        bad += not np.allclose(A["cand_num"][i, C["slot"][r], 10:], ref, atol=1e-5)
    check(results, 13, bad == 0, f"own-action pooling of {len(rows)} candidate rows rebuilt by a slow scan; mismatches {bad}")

    act, slot = C["act"], C["slot"]
    ok = bool(A["cand_mask"][act, slot].all()) and int(A["cand_mask"].sum()) == len(act)
    ok &= np.array_equal(A["cand_player"][act, slot], C["cand"]) and np.array_equal(A["cand_slot_rows"][act, slot], np.arange(len(act)))
    ok &= np.allclose(A["cand_num"][act, slot, 0], C["mins_on"] / 90) and np.allclose(A["cand_num"][act, slot, 3], C["since"] / 200)
    ok &= np.array_equal(A["cand_cat"][act, slot, 0], C["prior_role"] + 1)
    check(results, 14, ok, f"{len(act)} candidate rows placed in their slots with their state")

    check(results, 15, set(A) == INPUT_KEYS, f"input arrays: {sorted(A)}")
    with open(d / "checks_inputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("all input-array checks passed", flush=True)


if __name__ == "__main__":
    main()
