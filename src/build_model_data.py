"""Model view of one data scope for the who-acts model (docs/MODEL_PLAN.md, section 2).

Reads the tables of src/build_actions.py for the scope (actions, on-pitch intervals, player-match positions, coinciding
pairs), the raw events and matches, and the explicit model split. Writes to cfg["out_dir"]:
  matches.parquet         match_id, kickoff, date (UTC day), block (model_train / model_validation)
  events_in.parquet       every raw record in time order (seq), allowlisted fields only: period, sec, clock, team,
                          player, event and sub-event name, start x/y with placeholder starts masked (xy_valid)
  targets_in.parquet      one row per target action: keys and the admissible summary inputs (no game-state group; U
                          only for the urgency x position-group term)
  targets_meta.parquet    known-at variables used for evaluation subsets and bookkeeping, not model inputs
  targets_label.parquet   actor, action, end point, the four event targets, reachability and the actor's slot
  windows.npy             per target, seqs of the last cfg["window_events"] records before it, its coinciding records
                          removed, oldest first, -1 left padding
  candidates_in.npz       per (target, candidate): slot, in-match state and prematch history, computed in the target view
  candidates_label.npz    per (target, candidate): y (candidate is the actor)
  substitutions.parquet   match_id, team_id, player_in, player_out, minute (pairing from the raw match data)
  history.parquet         per (match, player on a roster): prematch summary under the split rule
  summary.json

Candidate rule (strict, symmetric, causal). Clock: first half min(sec, 2700), second half 2700 + sec. A starter is a
candidate from kickoff. A substitute becomes one once clock >= (minute + 1) * 60 or once he has a record in the target
view (sorted before the target and not one of its coinciding records). The player he replaces stays until the earlier of
the same two moments. A red card or second yellow ends candidacy at the card event (period, second). The actor is never
added; targets whose actor is outside the set are unreachable. A substitution record whose incoming player is 0 (not
identified in the match data) only removes the outgoing player at clock >= (minute + 1) * 60; the unidentified player is
never a candidate, and his actions are unreachable (counted in the summary).

History rule. Pool = training matches. A training target uses the pool matches of strictly earlier UTC days; a
validation target uses the whole pool (frozen at the training cut).
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from build_candidates import role_index
from common import ROOT, build, load
from event_schema import PERIOD_ORDER

PLACEHOLDER_SUBEVENTS = {34, 40, 51, 90, 91}  # starts that are not real positions (goal kick, goalkeeper events, whistle)
SUMMARY_INPUTS = ["zone", "x", "y", "phase", "prev_event", "steps_code", "opp_last_event", "possession_start", "venue",
                  "opponent_class", "poss_start_x", "poss_share_5", "own_shots_5", "opp_shots_5", "U"]
META = ["match_id", "event_id", "seq", "block", "team_id", "opponent_id", "period", "sec", "clock", "minute",
        "possession_id", "first_in_view", "n_coinciding_opponent", "n_coinciding_own", "k"]
LABELS = ["player_id", "action", "end_x", "end_y", "t_long_pass", "t_back_pass", "t_forward", "t_cross"]
CLOCK_CAP = 3600.0


def clock_of(period, sec):
    return np.where(period == 0, np.minimum(sec, 2700.0), 2700.0 + sec)


def split_table(train_manifest, validation_manifest):
    rows = [(r["match_id"], r["kickoff"], "model_train") for r in train_manifest]
    rows += [(r["match_id"], r["kickoff"], "model_validation") for r in validation_manifest]
    M = pd.DataFrame(rows, columns=["match_id", "kickoff", "block"]).sort_values(["kickoff", "match_id"], kind="stable")
    if M["match_id"].duplicated().any():
        raise ValueError("a match is in both the training and the validation list")
    M["date"] = (M["kickoff"] // 86400).astype(np.int64)
    if M.loc[M["block"] == "model_train", "kickoff"].max() >= M.loc[M["block"] == "model_validation", "kickoff"].min():
        raise ValueError("training matches must kick off before every validation match")
    return M.reset_index(drop=True)


def event_stream(cfg, data_cfg, match_ids):
    """Raw records of the matches in the order of src/build_actions.py, allowlisted fields only."""
    fix = {p["event_id"]: (p["x"], p["y"]) for p in data_cfg["position_overrides"]}
    by_match = defaultdict(list)
    for e in load(ROOT / data_cfg["raw_dir"] / "events" / f"events_{data_cfg['league']}.json"):
        if e["matchId"] in match_ids:
            if e["id"] in fix:
                e["positions"][0] = {"x": fix[e["id"]][0], "y": fix[e["id"]][1]}
            by_match[e["matchId"]].append(e)
    if set(by_match) != set(match_ids):
        raise ValueError("matches without events")
    rows = []
    for m in sorted(by_match):
        for s, e in enumerate(sorted(by_match[m], key=lambda e: (PERIOD_ORDER[e["matchPeriod"]], e["eventSec"]))):
            sub = -1 if e["subEventId"] == "" else int(e["subEventId"])
            valid = bool(e["positions"]) and sub not in PLACEHOLDER_SUBEVENTS
            rows.append({"match_id": m, "seq": s, "event_id": e["id"], "period": PERIOD_ORDER[e["matchPeriod"]],
                         "sec": float(e["eventSec"]), "team_id": e["teamId"], "player_id": e["playerId"],
                         "event_name": e["eventName"], "sub_event_name": e["subEventName"],
                         "x": float(e["positions"][0]["x"]) if valid else 0.0,
                         "y": float(e["positions"][0]["y"]) if valid else 0.0, "xy_valid": valid})
    E = pd.DataFrame(rows)
    E["clock"] = clock_of(E["period"].to_numpy(), E["sec"].to_numpy())
    return E


def substitutions(data_cfg, match_ids):
    rows = []
    for m in load(ROOT / data_cfg["raw_dir"] / "matches" / f"matches_{data_cfg['league']}.json"):
        if m["wyId"] not in match_ids:
            continue
        for team, d in m["teamsData"].items():
            subs = d["formation"]["substitutions"]
            for s in ([] if subs == "null" else subs):
                rows.append({"match_id": m["wyId"], "team_id": int(team), "player_in": s["playerIn"],
                             "player_out": s["playerOut"], "minute": float(s["minute"])})
    S = pd.DataFrame(rows, columns=["match_id", "team_id", "player_in", "player_out", "minute"])
    known_in = S[S["player_in"] != 0]
    if known_in.duplicated(["match_id", "player_in"]).any() or S.duplicated(["match_id", "player_out"]).any():
        raise ValueError("a player enters or leaves more than once in a match")
    return S


def windows(T, E, excl, k):
    """Seqs of the last k records before each target, excluding its coinciding records; oldest first, -1 padding."""
    out = np.full((len(T), k), -1, dtype=np.int32)
    seq_t = T["seq"].to_numpy()
    for i, s in enumerate(seq_t):
        drop = excl.get(i, ())
        lo = max(0, s - k - len(drop))
        kept = [j for j in range(lo, s) if j not in drop][-k:]
        out[i, k - len(kept):] = kept
    return out


def candidates(T, E, onpitch, S, excl, excluded_by, cap):
    """Candidate rows under the strict symmetric rule, with in-match state; see the module docstring.
    excl: target row -> seqs of its coinciding records; excluded_by: (match, record seq) -> target rows excluding it."""
    out = {key: [] for key in ("act", "cand", "slot", "mins_on", "starter", "cum", "since", "since_sec", "minutes_entered")}
    counts = {"targets": len(T), "unreachable": 0, "max_candidates": 0}
    spans = {key: g for key, g in onpitch.groupby(["match_id", "team_id"])}
    subs_in = {(r.match_id, r.player_in): r for r in S.itertuples(index=False) if r.player_in != 0}
    subs_out = {(r.match_id, r.player_out): r for r in S.itertuples(index=False)}
    rec = {key: (g["seq"].to_numpy(), g["clock"].to_numpy()) for key, g in E[E["player_id"] > 0].groupby(["match_id", "player_id"])}
    actor_all = T["player_id"].to_numpy()
    for (m, t), idx in T.groupby(["match_id", "team_id"]).indices.items():
        sp = spans[(m, t)]
        players = np.sort(sp["player_id"].to_numpy())
        sp = sp.set_index("player_id").loc[players]
        seq_t, clock_t = T["seq"].to_numpy()[idx], T["clock"].to_numpy()[idx]
        per_t, sec_t, minute_t = T["period"].to_numpy()[idx], T["sec"].to_numpy()[idx], T["minute"].to_numpy()[idx]
        n, P = len(idx), len(players)

        position = {row: a for a, row in enumerate(idx)}

        def first_view_clock(p):
            """Per target: clock of the player's first record in the target view (inf if none). The first record decides
            unless it is one of the target's coinciding records; those few targets are resolved record by record."""
            seqs, clocks = rec.get((m, p), (np.array([], int), np.array([])))
            if len(seqs) == 0:
                return np.full(n, np.inf)
            res = np.where(seqs[0] < seq_t, clocks[0], np.inf)
            for row in excluded_by.get((m, int(seqs[0])), ()):
                a = position.get(row)
                if a is None:
                    continue
                res[a] = np.inf
                for q, c in zip(seqs, clocks):
                    if q >= seq_t[a]:
                        break
                    if q not in excl[row]:
                        res[a] = c
                        break
            return res

        enter = np.zeros((n, P))  # clock at which the player became a candidate (inf: not yet)
        leave = np.full((n, P), np.inf)
        for j, p in enumerate(players):
            if (m, p) in subs_in:
                r = subs_in[(m, p)]
                enter[:, j] = np.minimum(np.where(clock_t >= (r.minute + 1) * 60, (r.minute + 1) * 60, np.inf), first_view_clock(p))
            if (m, p) in subs_out:
                r = subs_out[(m, p)]
                leave[:, j] = np.minimum(np.where(clock_t >= (r.minute + 1) * 60, (r.minute + 1) * 60, np.inf),
                                         first_view_clock(r.player_in))
        cper, csec = sp["card_period"].to_numpy(), sp["card_sec"].to_numpy()
        before_card = ((cper[None, :] < 0) | (per_t[:, None] < cper[None, :])
                       | ((per_t[:, None] == cper[None, :]) & (sec_t[:, None] < csec[None, :])))
        on = np.isfinite(enter) & ~np.isfinite(leave) & before_card
        actor = actor_all[idx]
        is_actor = actor[:, None] == players[None, :]
        reach = (on & is_actor).any(1)
        counts["unreachable"] += int((~reach).sum())
        counts["max_candidates"] = max(counts["max_candidates"], int(on.sum(1).max()))
        cum = np.cumsum(is_actor, axis=0) - is_actor
        pos = np.where(is_actor, np.arange(n)[:, None], -1)
        last = np.vstack([np.full((1, P), -1), np.maximum.accumulate(pos, axis=0)[:-1]])
        since = np.where(last >= 0, np.arange(n)[:, None] - last, cap)
        last_clock = np.where(last >= 0, clock_t[np.maximum(last, 0)], -np.inf)
        since_sec = np.minimum(clock_t[:, None] - last_clock, CLOCK_CAP)
        start = sp["start"].to_numpy()
        i, j = np.nonzero(on)
        slot = np.cumsum(on, axis=1) - 1
        out["act"].append(idx[i])
        out["cand"].append(players[j])
        out["slot"].append(slot[i, j])
        out["mins_on"].append(np.maximum(minute_t[i] - start[j], 0.0))
        out["starter"].append((start[j] == 0).astype(float))
        out["cum"].append(cum[i, j].astype(float))
        out["since"].append(np.minimum(since[i, j], cap).astype(float))
        out["since_sec"].append(since_sec[i, j])
        out["minutes_entered"].append((clock_t[i] - enter[i, j]) / 60.0)
    return {key: np.concatenate(v) for key, v in out.items()}, counts


def history(pairs, player_match, M, roles, gk_players):
    """Prematch summary per (match, player) under the split rule of the module docstring."""
    pool = M[M["block"] == "model_train"][["match_id", "date"]]
    pm = player_match.merge(pool, on="match_id")
    valid = pm["dx"].notna().to_numpy()
    n = pm["n_events"].to_numpy().astype(float)
    pm = pm.assign(w=np.where(valid, n, 0.0), w_dx=np.where(valid, pm["dx"] * n, 0.0), w_y=np.where(valid, pm["mean_y"] * n, 0.0),
                   n_x=pm["mean_x"] * n, n_y=pm["mean_y"] * n)
    by_player = {p: g.sort_values("date") for p, g in pm.groupby("player_id")}
    info = M.set_index("match_id")
    empty = pm.iloc[:0]
    rows = []
    for m, p in pairs.itertuples(index=False):
        h = by_player.get(p, empty)
        before = h if info.at[m, "block"] == "model_validation" else h[h["date"] < info.at[m, "date"]]
        w, ne = float(before["w"].sum()), float(before["n_events"].sum())
        rows.append({"match_id": m, "player_id": p, "prior_matches": int(len(before)), "prior_events": ne, "prior_weight": w,
                     "prior_dx": float(before["w_dx"].sum() / w) if w > 0 else np.nan,
                     "prior_y": float(before["w_y"].sum() / w) if w > 0 else np.nan,
                     "prior_mean_x": float(before["n_x"].sum() / ne) if ne > 0 else np.nan,
                     "prior_mean_y": float(before["n_y"].sum() / ne) if ne > 0 else np.nan})
    H = pd.DataFrame(rows)
    gk = H["player_id"].isin(gk_players).to_numpy()
    role = np.full(len(H), -1)
    role[gk] = roles.index("GK")
    ok = ~gk & (H["prior_weight"].to_numpy() > 0)
    role[ok] = role_index(H["prior_dx"].to_numpy()[ok], H["prior_y"].to_numpy()[ok], roles)
    return H.assign(prior_role=role, is_gk=gk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    data_cfg = load(ROOT / cfg["data_config"])
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    M = split_table(load(ROOT / cfg["train_manifest"]), load(ROOT / cfg["validation_manifest"]))
    match_ids = set(M["match_id"])
    A, meta = build(data_cfg)
    roles = meta["roles"]
    if set(A["match_id"]) != match_ids:
        raise ValueError("the action table and the model split cover different matches")
    E = event_stream(cfg, data_cfg, match_ids)
    seq_of = dict(zip(E["event_id"], E["seq"]))
    if not A["event_id"].isin(seq_of.keys()).all():
        raise ValueError("target actions missing from the event stream")

    # targets: inputs, meta, labels (rows in the order of the action table)
    T = A.copy()
    T["seq"] = T["event_id"].map(seq_of).astype(np.int64)
    T["block"] = T["match_id"].map(dict(zip(M["match_id"], M["block"])))
    T["period"] = T["period"].map(PERIOD_ORDER)
    T["sec"] = T["event_sec"].astype(float)
    T["clock"] = clock_of(T["period"].to_numpy(), T["sec"].to_numpy())
    check = E.set_index(["match_id", "seq"]).loc[pd.MultiIndex.from_arrays([T["match_id"], T["seq"]])]
    if not (np.isclose(check["sec"].to_numpy(), T["sec"].to_numpy()) & (check["team_id"].to_numpy() == T["team_id"].to_numpy())).all():
        raise ValueError("targets do not line up with the event stream")

    # coinciding records per target row
    C = pd.read_parquet(ROOT / data_cfg["coinciding_path"])
    row_of = dict(zip(T["event_id"], range(len(T))))
    excl, excluded_by = defaultdict(set), defaultdict(list)
    for m, tgt, r in zip(C["match_id"], C["target_event_id"], C["record_event_id"]):
        excl[row_of[tgt]].add(seq_of[r])
        excluded_by[(m, seq_of[r])].append(row_of[tgt])
    W = windows(T, E, excl, cfg["window_events"])

    S = substitutions(data_cfg, match_ids)
    onpitch = pd.read_parquet(ROOT / data_cfg["onpitch_path"])
    Cn, counts = candidates(T, E, onpitch, S, excl, excluded_by, data_cfg["since_cap"])
    actor = T["player_id"].to_numpy()[Cn["act"]]
    y = (actor == Cn["cand"]).astype(np.int8)
    actor_slot = np.full(len(T), -1)
    actor_slot[Cn["act"][y == 1]] = Cn["slot"][y == 1]

    players = {p["wyId"]: p for p in load(ROOT / data_cfg["raw_dir"] / "entities" / "players.json")}
    gk_players = {p for p, v in players.items() if v["role"]["code2"] == "GK"}
    pairs = pd.DataFrame({"match_id": T["match_id"].to_numpy()[Cn["act"]], "player_id": Cn["cand"]}).drop_duplicates()
    H = history(pairs, pd.read_parquet(ROOT / data_cfg["player_match_path"]), M, roles, gk_players)
    key = pd.MultiIndex.from_arrays([T["match_id"].to_numpy()[Cn["act"]], Cn["cand"]])
    Hk = H.set_index(["match_id", "player_id"]).loc[key]
    for c in ("prior_role", "prior_matches", "prior_events", "prior_mean_x", "prior_mean_y"):
        Cn[c] = Hk[c].to_numpy()
    Cn["is_gk"] = Hk["is_gk"].to_numpy().astype(np.int8)
    Cn["wyscout_role"] = np.array([["GK", "DF", "MD", "FW"].index(players[p]["role"]["code2"]) for p in Cn["cand"]], dtype=np.int8)
    tx, ty = T["x"].to_numpy()[Cn["act"]], T["y"].to_numpy()[Cn["act"]]
    Cn["dist"] = np.hypot(1.05 * (Cn["prior_mean_x"] - tx), 0.68 * (Cn["prior_mean_y"] - ty))

    M.to_parquet(out / "matches.parquet", index=False)
    E.to_parquet(out / "events_in.parquet", index=False)
    T[["match_id", "event_id", "seq", "team_id", "opponent_id"] + SUMMARY_INPUTS].to_parquet(out / "targets_in.parquet", index=False)
    T[META].to_parquet(out / "targets_meta.parquet", index=False)
    T[LABELS].assign(reachable=actor_slot >= 0, actor_slot=actor_slot).to_parquet(out / "targets_label.parquet", index=False)
    np.save(out / "windows.npy", W)
    np.savez(out / "candidates_in.npz", **Cn)
    np.savez(out / "candidates_label.npz", y=y)
    S.to_parquet(out / "substitutions.parquet", index=False)
    H.to_parquet(out / "history.parquet", index=False)
    blk = T["block"].to_numpy()
    summary = {"league": data_cfg["league"], "matches": M["block"].value_counts().to_dict(), "events": int(len(E)),
               "targets": {b: int((blk == b).sum()) for b in ("model_train", "model_validation")},
               "unreachable": {b: int(((actor_slot < 0) & (blk == b)).sum()) for b in ("model_train", "model_validation")},
               "candidate_rows": int(len(y)), "max_candidates": counts["max_candidates"],
               "excluded_pairs": int(sum(len(v) for v in excl.values())), "substitutions": int(len(S)),
               "substitutions_without_incoming_player": int((S["player_in"] == 0).sum()),
               "targets_by_unlisted_actor": int((~pd.Series(list(zip(T["match_id"], T["player_id"]))).isin(
                   set(zip(onpitch["match_id"], onpitch["player_id"])))).sum()),
               "history_rows": int(len(H)), "without_prior_role": float(np.mean(H["prior_role"] < 0)),
               "placeholder_starts": int((~E["xy_valid"]).sum()), "roles": roles}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({key: v for key, v in summary.items() if key != "roles"}), flush=True)


if __name__ == "__main__":
    main()
