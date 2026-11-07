"""Causal off-ball position estimates per (target, candidate) for the who-acts probe (docs/MODEL_PLAN.md, note 21).

Only the team's earlier on-ball actions (targets) of the same match are used: their start positions (in the team's
attacking frame) and performers. Earlier targets are context events and are never coinciding records of a later
target, so the estimates use nothing a target's own inputs exclude; of the target itself only its time and start
(given in condition A) enter. Per (target i, candidate p):
  idle      seconds since p's last earlier action in the match (-1: none yet)
  d0        metres from p's last earlier action position to the start of i (E0, last seen; -1: none)
  d2        metres from anchor(i) + offset(p) to the start of i (E2, team anchor plus the player's offset; -1: none),
            anchor(t) = mean start of the team's last cfg["anchor_actions"] earlier actions within cfg["anchor_seconds"]
            seconds before t; offset(p) = mean over p's earlier actions j of (start(j) - anchor(j))
  n_obs     number of p's earlier actions with an anchor (the offset's sample size)
Writes <scope out_dir>/offball.npz with arrays (N, C) aligned with arrays_in.npz.
"""
import argparse
from collections import deque

import numpy as np
import pandas as pd

from common import ROOT, load
from model_data import scope_dirs

SX, SY = 1.05, 0.68  # Wyscout percent to metres


def scope_features(d, anchor_actions, anchor_seconds):
    T = pd.read_parquet(d / "targets_in.parquet", columns=["match_id", "seq", "team_id", "x", "y"])
    M = pd.read_parquet(d / "targets_meta.parquet", columns=["clock", "period"])
    E = pd.read_parquet(d / "events_in.parquet", columns=["match_id", "seq", "player_id"])
    C = dict(np.load(d / "candidates_in.npz"))
    A = np.load(d / "arrays_in.npz")
    N, Cmax = A["cand_mask"].shape
    performer = E.set_index(["match_id", "seq"])["player_id"].reindex(pd.MultiIndex.from_frame(T[["match_id", "seq"]])).to_numpy()
    if np.isnan(performer.astype(float)).any():
        raise ValueError("a target without a performer in events_in")
    x, y = T["x"].to_numpy() * SX, T["y"].to_numpy() * SY
    clock = M["clock"].to_numpy()
    by_target = pd.DataFrame({"act": C["act"], "cand": C["cand"], "slot": C["slot"]}).groupby("act")
    cand_of = {a: (g["cand"].to_numpy(), g["slot"].to_numpy()) for a, g in by_target}
    out = {k: np.full((N, Cmax), -1.0, np.float32) for k in ("idle", "d0", "d2")}
    out["n_obs"] = np.zeros((N, Cmax), np.float32)
    order = np.lexsort((T["seq"].to_numpy(), T["team_id"].to_numpy(), T["match_id"].to_numpy()))
    rows = pd.Series(order).groupby([T["match_id"].to_numpy()[order], T["team_id"].to_numpy()[order]]).apply(list)
    for idx in rows:
        recent = deque()  # (clock, x, y) of the team's earlier actions
        last, offs = {}, {}  # player -> (clock, x, y); player -> [sum_dx, sum_dy, n]
        for i in idx:
            t = clock[i]
            win = [r for r in recent if r[0] >= t - anchor_seconds][-anchor_actions:]
            anchor = (np.mean([r[1] for r in win]), np.mean([r[2] for r in win])) if win else None
            for p, s in zip(*cand_of.get(i, ((), ()))):
                if p in last:
                    lt, lx, ly = last[p]
                    out["idle"][i, s] = t - lt
                    out["d0"][i, s] = np.hypot(x[i] - lx, y[i] - ly)
                o = offs.get(p)
                if o is not None and o[2] > 0 and anchor is not None:
                    out["d2"][i, s] = np.hypot(x[i] - (anchor[0] + o[0] / o[2]), y[i] - (anchor[1] + o[1] / o[2]))
                    out["n_obs"][i, s] = o[2]
            p = performer[i]  # the action itself becomes history for later targets
            if anchor is not None:
                o = offs.setdefault(p, [0.0, 0.0, 0])
                o[0] += x[i] - anchor[0]; o[1] += y[i] - anchor[1]; o[2] += 1
            last[p] = (t, x[i], y[i])
            recent.append((t, x[i], y[i]))
            while len(recent) > anchor_actions:
                recent.popleft()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    for d in scope_dirs(cfg["scopes"]):
        F = scope_features(d, cfg["anchor_actions"], cfg["anchor_seconds"])
        np.savez(d / "offball.npz", **F)
        mask = np.load(d / "arrays_in.npz")["cand_mask"]
        print(d.parent.name, {k: float((v[mask] >= 0).mean()) for k, v in F.items() if k != "n_obs"}, flush=True)


if __name__ == "__main__":
    main()
