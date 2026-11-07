"""Fixed-shape model arrays of one scope (docs/MODEL_PLAN.md, sections 2 and 3), from the tables of
src/build_model_data.py. Reads only the input tables (events_in, targets_in, candidates_in, windows) and, for the
labels file, targets_label and candidates_label. Writes to the scope's model directory:
  arrays_in.npz     per target (row order of targets_in):
    win_cat  (N, K, 4) int16   event name, sub-event name, side (1 own, 2 opponent; 0 padding), zone in the target
                               team's attacking frame (9 = unknown start)
    win_num  (N, K, 6) float32 x/100, y/100 in that frame (0 if unknown), start known, log1p(dt)/5 with dt = target
                               clock - record clock within the period, dt valid (same period), position in the window / K
    win_valid (N, K) bool
    ctx_cat  (N, 7) int16      zone, phase, prev_event, opp_last_event, possession_start, venue, opponent_class
    ctx_num  (N, 7) float32    x/100, y/100, steps/4, possession start x/100, possession share, own and opponent
                               shots in the momentum window (/5)
    urgency  (N,) float32      U (raw; standardised by the training script on training rows)
    cand_cat (N, C, 4) int16   prior role + 1 (0 = none), Wyscout role, position group (7 = unknown prior), goalkeeper
    cand_num (N, C, 36) float32 state (10): minutes on/90, starter, log1p(actions so far)/4, actions since last/200,
                               log1p(seconds since last)/8, minutes since entering/90, prior matches/38,
                               log1p(prior events)/8, distance/50 (0 if unknown), distance known;
                               own on-ball actions before the target in the match, last 32 (26): mean action-type
                               frequencies (10), mean x/100, mean y/100, mean log1p(dt)/8, count/32, then the last
                               action: type one-hot (10), x/100, y/100
    cand_mask (N, C) bool, cand_player (N, C) int64, cand_slot_rows (N, C) int64 (row in candidates_in, -1 padding)
  arrays_label.npz  actor_slot (N,) int64 (-1 unreachable)
  arrays_meta.json  vocabularies and shapes
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import GROUP_OF, GROUPS, ROOT, load
from event_schema import ACTIONS, ACTION_OF

CTX_CAT = ["zone", "phase", "prev_event", "opp_last_event", "possession_start", "venue", "opponent_class"]
POOL = 32
C_MAX = 11
VOCAB = {
    "event_name": ["Duel", "Foul", "Free Kick", "Goalkeeper leaving line", "Interruption", "Offside", "Others on the ball",
                   "Pass", "Save attempt", "Shot"],
    "sub_event_name": ["", "Acceleration", "Air duel", "Ball out of the field", "Clearance", "Corner", "Cross", "Foul",
                       "Free Kick", "Free kick cross", "Free kick shot", "Goal kick", "Goalkeeper leaving line",
                       "Ground attacking duel", "Ground defending duel", "Ground loose ball duel", "Hand foul", "Hand pass",
                       "Head pass", "High pass", "Late card foul", "Launch", "Out of game foul", "Penalty", "Protest",
                       "Reflexes", "Save attempt", "Shot", "Simple pass", "Simulation", "Smart pass", "Throw in",
                       "Time lost foul", "Touch", "Violent Foul", "Whistle"],
    "phase": ["first", "early", "settled"],
    "prev_event": ["period_start", "own", "opp_pass", "opp_shot", "opp_goalkeeper", "opp_set_piece", "opp_clearance",
                   "opp_other_on_ball"],
    "opp_last_event": ["none", "foul", "duel", "interception", "clearance", "shot", "pass", "set_piece", "goalkeeper",
                       "offside", "other_on_ball"],
    "possession_start": ["period_start", "restart_out", "restart_foul", "interception", "duel_won", "after_opp_shot",
                         "opp_clearance", "opp_pass_lost", "other"],
    "venue": ["home", "away"],
    "opponent_class": ["big_six", "other"],
}


def codes(values, vocab, name):
    index = {v: i for i, v in enumerate(vocab)}
    out = pd.Series(values).map(index)
    if out.isna().any():
        raise ValueError(f"{name}: values outside the vocabulary: {sorted(set(pd.Series(values)[out.isna()]))[:5]}")
    return out.to_numpy().astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    d = ROOT / cfg["out_dir"]
    E = pd.read_parquet(d / "events_in.parquet")
    T = pd.read_parquet(d / "targets_in.parquet")
    W = np.load(d / "windows.npy")
    C = dict(np.load(d / "candidates_in.npz"))
    meta_t = pd.read_parquet(d / "targets_meta.parquet")
    if not (meta_t["event_id"].to_numpy() == T["event_id"].to_numpy()).all():
        raise ValueError("targets_meta and targets_in are not row-aligned")
    N, K = W.shape
    subs = VOCAB["sub_event_name"]

    # ---- window tokens ----
    offset = dict(zip(E["match_id"], np.arange(len(E)) - E["seq"].to_numpy()))
    if not (np.diff(E["seq"].to_numpy()) == 1)[(E["match_id"].to_numpy()[1:] == E["match_id"].to_numpy()[:-1])].all():
        raise ValueError("events_in is not ordered by match and seq")
    off_t = T["match_id"].map(offset).to_numpy()
    seq_w = W.reshape(-1)
    valid = seq_w >= 0
    rows = np.where(valid, np.repeat(off_t, K) + seq_w, -1)
    ev_name = codes(E["event_name"], VOCAB["event_name"], "event_name") + 1
    ev_sub = codes(E["sub_event_name"], subs, "sub_event_name") + 1
    team_t = np.repeat(T["team_id"].to_numpy(), K)
    own = E["team_id"].to_numpy()[np.maximum(rows, 0)] == team_t
    x, y = E["x"].to_numpy()[np.maximum(rows, 0)], E["y"].to_numpy()[np.maximum(rows, 0)]
    xy_ok = E["xy_valid"].to_numpy()[np.maximum(rows, 0)] & valid
    x, y = np.where(own, x, 100 - x), np.where(own, y, 100 - y)
    x, y = np.where(xy_ok, x, 0.0), np.where(xy_ok, y, 0.0)
    zone = np.where(xy_ok, 3 * np.minimum((x * 3 / 100).astype(int), 2) + np.minimum((y * 3 / 100).astype(int), 2), 9)
    per_e, clk_e = E["period"].to_numpy()[np.maximum(rows, 0)], E["clock"].to_numpy()[np.maximum(rows, 0)]
    per_t, clk_t = np.repeat(meta_t["period"].to_numpy(), K), np.repeat(meta_t["clock"].to_numpy(), K)
    dt_ok = valid & (per_e == per_t)
    dt = np.where(dt_ok, np.log1p(np.maximum(clk_t - clk_e, 0.0)) / 5, 0.0)
    position = np.tile(np.arange(K) / K, N)
    win_cat = np.stack([np.where(valid, ev_name[np.maximum(rows, 0)], 0), np.where(valid, ev_sub[np.maximum(rows, 0)], 0),
                        np.where(valid, np.where(own, 1, 2), 0), np.where(valid, zone, 9)], -1).reshape(N, K, 4).astype(np.int16)
    win_num = np.stack([x / 100, y / 100, xy_ok, dt, dt_ok, np.where(valid, position, 0)], -1).reshape(N, K, 6).astype(np.float32)

    # ---- context summary ----
    ctx_cat = np.stack([T["zone"].to_numpy().astype(np.int16)] +
                       [codes(T[c], VOCAB[c], c) for c in CTX_CAT[1:]], -1)
    ctx_num = np.stack([T["x"] / 100, T["y"] / 100, T["steps_code"] / 4, T["poss_start_x"] / 100, T["poss_share_5"],
                        T["own_shots_5"] / 5, T["opp_shots_5"] / 5], -1).astype(np.float32)

    # ---- candidates ----
    act, slot = C["act"], C["slot"]
    if slot.max() >= C_MAX:
        raise ValueError(f"more than {C_MAX} candidates in a target")
    roles = load(d / "summary.json")["roles"]
    group = np.array([GROUPS.index(GROUP_OF[roles[r]]) if r >= 0 else len(GROUPS) for r in C["prior_role"]])
    dist_ok = np.isfinite(C["dist"])
    state = np.stack([C["mins_on"] / 90, C["starter"], np.log1p(C["cum"]) / 4, C["since"] / 200, np.log1p(C["since_sec"]) / 8,
                      np.minimum(C["minutes_entered"], 120) / 90, C["prior_matches"] / 38, np.log1p(C["prior_events"]) / 8,
                      np.where(dist_ok, C["dist"], 0.0) / 50, dist_ok], -1)

    # own on-ball actions before the target (target-view safe: own actions are never coinciding records)
    action_code = pd.Series(list(zip(E["event_name"], E["sub_event_name"]))).map(
        {k: ACTIONS.index(v) for k, v in ACTION_OF.items()})
    t_rows = off_t + T["seq"].to_numpy()
    a_type = action_code.to_numpy()[t_rows].astype(int)
    performer = E["player_id"].to_numpy()[t_rows]  # who performed each action, read from the input event table
    feats = np.concatenate([np.eye(len(ACTIONS))[a_type], (T[["x", "y"]].to_numpy() / 100)], 1)
    clock = meta_t["clock"].to_numpy()
    pooled = np.zeros((len(act), 26))
    grp = pd.DataFrame({"m": T["match_id"].to_numpy(), "p": performer}).groupby(["m", "p"]).indices
    cand_groups = pd.DataFrame({"m": T["match_id"].to_numpy()[act], "p": C["cand"]}).groupby(["m", "p"]).indices
    cum = C["cum"].astype(int)
    for (m, p), idx in grp.items():
        # idx: this player's actions in the match, in time order. A candidate row uses only the first `cum` of them,
        # the actions before its target, so the scored target's own performer never enters its inputs.
        rows_c = cand_groups.get((m, p))
        if rows_c is None:
            continue
        F = np.vstack([np.zeros((1, feats.shape[1])), np.cumsum(feats[idx], 0)])
        Tc = np.concatenate([[0.0], np.cumsum(clock[idx])])
        n_before = cum[rows_c]
        lo = np.maximum(n_before - POOL, 0)
        cnt = n_before - lo
        has = cnt > 0
        mean = np.where(has[:, None], (F[n_before] - F[lo]) / np.maximum(cnt, 1)[:, None], 0.0)
        mean_clock = np.where(has, (Tc[n_before] - Tc[lo]) / np.maximum(cnt, 1), 0.0)
        mdt = np.where(has, np.log1p(np.maximum(clock[act[rows_c]] - mean_clock, 0.0)) / 8, 0.0)
        last = np.where(has[:, None], feats[idx][np.maximum(n_before - 1, 0)], 0.0)
        pooled[rows_c] = np.concatenate([mean, mdt[:, None], (cnt / POOL)[:, None], last], 1)
    cand_num_rows = np.concatenate([state, pooled], 1).astype(np.float32)

    cand_cat = np.zeros((N, C_MAX, 4), np.int16)
    cand_num = np.zeros((N, C_MAX, cand_num_rows.shape[1]), np.float32)
    cand_mask = np.zeros((N, C_MAX), bool)
    cand_player = np.zeros((N, C_MAX), np.int64)
    cand_rows = np.full((N, C_MAX), -1, np.int64)
    cand_cat[act, slot] = np.stack([C["prior_role"] + 1, C["wyscout_role"], group, C["is_gk"]], -1)
    cand_num[act, slot] = cand_num_rows
    cand_mask[act, slot] = True
    cand_player[act, slot] = C["cand"]
    cand_rows[act, slot] = np.arange(len(act))

    labels = pd.read_parquet(d / "targets_label.parquet")
    np.savez(d / "arrays_in.npz", win_cat=win_cat, win_num=win_num, win_valid=valid.reshape(N, K), ctx_cat=ctx_cat,
             ctx_num=ctx_num, urgency=T["U"].to_numpy().astype(np.float32), cand_cat=cand_cat, cand_num=cand_num,
             cand_mask=cand_mask, cand_player=cand_player, cand_slot_rows=cand_rows)
    np.savez(d / "arrays_label.npz", actor_slot=labels["actor_slot"].to_numpy().astype(np.int64))
    meta = {"targets": int(N), "window": int(K), "candidates": C_MAX, "sub_event_names": VOCAB["sub_event_name"],
            "vocab": VOCAB, "groups": GROUPS + ["unknown"], "roles": roles, "cand_num": int(cand_num.shape[2])}
    with open(d / "arrays_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k in ("targets", "window", "candidates", "cand_num")}), flush=True)


if __name__ == "__main__":
    main()
