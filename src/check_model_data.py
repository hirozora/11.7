"""Acceptance checks of the model view (docs/MODEL_PLAN.md, section 8) on a built scope. Stops at the first failure.

Checks on the built tables:
  1  allowlist: the input tables hold exactly the allowlisted columns; no game-state input
  2  windows: no coinciding record of a target is in its window; windows equal a slow scan (sample)
  3  candidates do not depend on the actor (actor swapped, sample of team-matches)
  4  substitution timing: no candidacy starts or ends before the time the substitution record allows; half-time
     substitutions and first-half stoppage time checked separately; red cards end candidacy at the card event
  5  candidate state equals a slow scan of the team's earlier actions (sample)
  6  history: corrupting validation matches, or training matches from a given day on, leaves earlier history unchanged
  7  test isolation: no test-list match in any table, candidate rows included
  8  placeholder starts are masked
  9  target set: reachability equals a recount from the candidate rows
End-to-end perturbation on a small scope rebuilt from raw data (3 matches of the league: 2 training, 1 validation):
  10 changing every end point, and every tag outside the derived-variable list, leaves all inputs unchanged
  11 future invariance: changing the records after a target and the target's own event type (to another on-ball
     action, so that it stays a target), tags, end point and actor leaves its inputs unchanged; changing a record
     before it changes them (positive control)
Writes cfg["out_dir"]/checks.json.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_model_data as bm
from common import ROOT, load
from event_schema import FINE_ROLES

ALLOWED_EVENTS = ["match_id", "seq", "event_id", "period", "sec", "team_id", "player_id", "event_name", "sub_event_name",
                  "x", "y", "xy_valid", "clock"]
ALLOWED_TARGETS = ["match_id", "event_id", "seq", "team_id", "opponent_id"] + bm.SUMMARY_INPUTS
ALLOWED_CANDIDATES = ["act", "cand", "slot", "mins_on", "starter", "cum", "since", "since_sec", "minutes_entered", "prior_role",
                      "prior_matches", "prior_events", "prior_mean_x", "prior_mean_y", "is_gk", "wyscout_role", "dist"]
GAME_STATE = ["score_state", "score_diff", "minute", "time_bin", "minutes_since_goal", "state_duration"]
DERIVED_TAGS = {101, 102, 701, 703, 1401, 1701, 1703}  # goals, duel lost/won, interception, red card, second yellow


def check(results, number, ok, detail):
    results[str(number)] = {"passed": bool(ok), "detail": detail}
    print(f"check {number}: {'passed' if ok else 'FAILED'} {detail}", flush=True)
    if not ok:
        raise SystemExit(f"check {number} failed: {detail}")


def load_scope(out):
    return {"M": pd.read_parquet(out / "matches.parquet"), "E": pd.read_parquet(out / "events_in.parquet"),
            "T": pd.read_parquet(out / "targets_in.parquet"), "Tm": pd.read_parquet(out / "targets_meta.parquet"),
            "L": pd.read_parquet(out / "targets_label.parquet"), "W": np.load(out / "windows.npy"),
            "C": dict(np.load(out / "candidates_in.npz")), "y": np.load(out / "candidates_label.npz")["y"],
            "S": pd.read_parquet(out / "substitutions.parquet"), "H": pd.read_parquet(out / "history.parquet")}


def coinciding_sets(data_cfg, D):
    C = pd.read_parquet(ROOT / data_cfg["coinciding_path"])
    seq_of = dict(zip(D["E"]["event_id"], D["E"]["seq"]))
    row_of = dict(zip(D["T"]["event_id"], range(len(D["T"]))))
    excl = {}
    for tgt, r in zip(C["target_event_id"], C["record_event_id"]):
        excl.setdefault(row_of[tgt], set()).add(seq_of[r])
    return excl


def static_checks(cfg, data_cfg, D, results, rng):
    E, T, Tm, L, W, C, y, S = (D[k] for k in ("E", "T", "Tm", "L", "W", "C", "y", "S"))
    check(results, 1, list(E.columns) == ALLOWED_EVENTS and list(T.columns) == ALLOWED_TARGETS
          and sorted(C) == sorted(ALLOWED_CANDIDATES) and not set(GAME_STATE) & set(T.columns),
          "events, targets and candidates hold only allowlisted inputs; no game-state input")

    excl = coinciding_sets(data_cfg, D)
    inside = sum(len(set(W[i]) & s) for i, s in excl.items())
    sample = rng.choice(len(T), size=min(2000, len(T)), replace=False)
    seq = T["seq"].to_numpy()
    slow_ok = True
    for i in sample:
        kept = [s for s in range(seq[i]) if s not in excl.get(i, set())][-cfg["window_events"]:]
        ref = np.full(cfg["window_events"], -1)
        ref[cfg["window_events"] - len(kept):] = kept
        slow_ok &= bool(np.array_equal(ref, W[i]))
    check(results, 2, inside == 0 and slow_ok, f"coinciding records inside windows: {inside}; slow scan on {len(sample)} targets")

    # 3: actor swap on sampled team-matches
    onpitch = pd.read_parquet(ROOT / data_cfg["onpitch_path"])
    if not (T["event_id"].to_numpy() == Tm["event_id"].to_numpy()).all():
        raise ValueError("targets_in and targets_meta are not row-aligned")
    Tfull = T.assign(**{c: Tm[c].to_numpy() for c in ("period", "sec", "clock", "minute")}, player_id=L["player_id"].to_numpy())
    groups = list(Tfull.groupby(["match_id", "team_id"]).indices.items())
    same = True
    for g in rng.choice(len(groups), size=min(20, len(groups)), replace=False):
        (m, t), idx = groups[g]
        sub = Tfull.iloc[idx].reset_index(drop=True)
        ex = {a: excl[i] for a, i in enumerate(idx) if i in excl}
        exb = {}
        for a, s in ex.items():
            for r in s:
                exb.setdefault((m, r), []).append(a)
        Em = E[E["match_id"] == m]
        a0, _ = bm.candidates(sub, Em, onpitch, S, ex, exb, data_cfg["since_cap"])
        swapped = sub.copy()
        swapped["player_id"] = rng.permutation(swapped["player_id"].to_numpy())
        a1, _ = bm.candidates(swapped, Em, onpitch, S, ex, exb, data_cfg["since_cap"])
        same &= all(np.array_equal(a0[k], a1[k]) for k in ("act", "cand", "slot", "mins_on", "starter", "minutes_entered"))
    check(results, 3, same, "candidate sets and entry times are unchanged when the actors of 20 team-matches are permuted")

    # 4: substitution timing on the built candidate rows (independent of the builder's code path)
    act, cand = C["act"], C["cand"]
    on = set(zip(act.tolist(), cand.tolist()))
    team_t, match_t, seq_t, clock_t = (Tm[c].to_numpy() for c in ("team_id", "match_id", "seq", "clock"))
    per_t, sec_t = Tm["period"].to_numpy(), Tm["sec"].to_numpy()
    rec = {k: g["seq"].to_numpy() for k, g in E[E["player_id"] > 0].groupby(["match_id", "player_id"])}
    rows_of = Tm.groupby(["match_id", "team_id"]).indices
    start_of = dict(zip(zip(onpitch["match_id"], onpitch["player_id"]), onpitch["start"]))
    card_of = {(m, p): (cp, cs) for m, p, cp, cs in zip(onpitch["match_id"], onpitch["player_id"], onpitch["card_period"],
                                                         onpitch["card_sec"]) if cp >= 0}

    def seen(m, p, idx):
        """Per target row: does the player have a record before it that is not one of its coinciding records?"""
        r = rec.get((m, p), np.array([], int))
        before = np.searchsorted(r, seq_t[idx])
        dropped = np.array([len(excl[i] & set(r[:b].tolist())) if i in excl else 0 for i, b in zip(idx, before)])
        return before - dropped > 0

    bad_in, bad_out, half_time, checked = 0, 0, 0, 0
    for r in S.itertuples(index=False):
        idx = rows_of[(r.match_id, r.team_id)]
        allowed = (clock_t[idx] >= (r.minute + 1) * 60) | (seen(r.match_id, r.player_in, idx) if r.player_in else False)
        in_on = np.array([(i, r.player_in) in on for i in idx])
        out_on = np.array([(i, r.player_out) in on for i in idx])
        s_out = start_of[(r.match_id, r.player_out)]
        entered_out = (s_out == 0) | (clock_t[idx] >= (s_out + 1) * 60) | seen(r.match_id, r.player_out, idx)
        c = card_of.get((r.match_id, r.player_out))
        before_card = np.ones(len(idx), bool) if c is None else (per_t[idx] < c[0]) | ((per_t[idx] == c[0]) & (sec_t[idx] < c[1]))
        bad_in += int((in_on & ~allowed).sum())
        bad_out += int((~out_on & ~allowed & entered_out & before_card).sum())
        checked += len(idx)
        half_time += int(45 <= r.minute <= 46)
    stoppage = int(((per_t == 0) & (sec_t >= 2700)).sum())
    check(results, 4, bad_in == 0 and bad_out == 0,
          f"{len(S)} substitutions ({half_time} at half time, minute 45-46), {checked} target checks; early entries {bad_in}, "
          f"early exits {bad_out}; first-half stoppage-time targets {stoppage}")

    # 5: candidate state against a slow scan
    actor = L["player_id"].to_numpy()
    ok5 = True
    for i in rng.choice(len(T), size=min(500, len(T)), replace=False):
        rows = np.flatnonzero(act == i)
        same_team = np.flatnonzero((match_t == match_t[i]) & (team_t == team_t[i]) & (seq_t < seq_t[i]))
        for r in rows:
            mine = same_team[actor[same_team] == cand[r]]
            cum = len(mine)
            since = (len(same_team) - 1 - np.flatnonzero(actor[same_team] == cand[r])[-1]) + 1 if cum else data_cfg["since_cap"]
            ok5 &= C["cum"][r] == cum and C["since"][r] == min(since, data_cfg["since_cap"])
    check(results, 5, ok5, "in-match participation and recency match a slow scan on 500 targets")

    # 7: test isolation
    test = {r["match_id"] for r in load(ROOT / cfg["test_manifest"])}
    tables = {k: set(D[k]["match_id"]) for k in ("M", "E", "T", "Tm", "S", "H")}
    hit = {k: len(v & test) for k, v in tables.items()}
    hit["candidates"] = len(set(match_t[act]) & test)
    check(results, 7, sum(hit.values()) == 0, f"test-list matches per table: {hit}")

    placeholder = E["sub_event_name"].isin(["Goal kick", "Goalkeeper leaving line", "Whistle", "Reflexes", "Save attempt"])
    check(results, 8, (~E.loc[placeholder, "xy_valid"]).all() and (E.loc[~E["xy_valid"], ["x", "y"]] == 0).all().all(),
          f"{int(placeholder.sum())} placeholder starts masked")

    reach = np.zeros(len(T), bool)
    reach[act[y == 1]] = True
    check(results, 9, np.array_equal(reach, L["reachable"].to_numpy()) and np.bincount(act[y == 1], minlength=len(T)).max() <= 1,
          f"reachable {int(reach.sum())}, unreachable {int((~reach).sum())}")


def history_check(data_cfg, D, results, rng):
    M, H = D["M"], D["H"]
    pm = pd.read_parquet(ROOT / data_cfg["player_match_path"])
    players = {p["wyId"]: p for p in load(ROOT / data_cfg["raw_dir"] / "entities" / "players.json")}
    gk = {p for p, v in players.items() if v["role"]["code2"] == "GK"}
    roles = sorted(set(FINE_ROLES) - {"unassigned"})
    pairs = H[["match_id", "player_id"]]
    base = bm.history(pairs, pm, M, roles, gk)
    val = set(M.loc[M["block"] == "model_validation", "match_id"])
    bad = pm.copy()
    hit = bad["match_id"].isin(val)
    bad.loc[hit, ["dx", "mean_x", "mean_y"]] = rng.normal(0, 30, (int(hit.sum()), 3))
    bad.loc[hit, "n_events"] = rng.integers(1, 200, int(hit.sum()))
    ok = base.equals(bm.history(pairs, bad, M, roles, gk))
    train = M[M["block"] == "model_train"].sort_values("date")
    for d in rng.choice(train["date"].unique(), size=5, replace=False):
        late = set(train.loc[train["date"] >= d, "match_id"])
        bad = pm.copy()
        hit = bad["match_id"].isin(late)
        bad.loc[hit, ["dx", "mean_x", "mean_y"]] = rng.normal(0, 30, (int(hit.sum()), 3))
        early_pairs = pairs[pairs["match_id"].isin(set(train.loc[train["date"] <= d, "match_id"]))]
        ok &= bm.history(early_pairs, pm, M, roles, gk).equals(bm.history(early_pairs, bad, M, roles, gk))
    check(results, 6, ok, "history unchanged after corrupting validation matches, and training matches from 5 sampled days on")


def mini_scope(cfg, data_cfg, root, raw_events, n_train, n_val, tag):
    """Rebuild a small scope from (possibly perturbed) raw events: build_actions, then build_model_data."""
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw"
    for sub in ("events", "matches", "entities"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    league = data_cfg["league"]
    with open(raw / "events" / f"events_{league}.json", "w") as f:
        json.dump(raw_events, f)
    src_raw = ROOT / data_cfg["raw_dir"]
    for name in ("players.json", "teams.json"):
        target = raw / "entities" / name
        if not target.exists():
            target.symlink_to((src_raw / "entities" / name).resolve())
    matches = [m for m in load(src_raw / "matches" / f"matches_{league}.json")
               if m["wyId"] in {e["matchId"] for e in raw_events}]
    with open(raw / "matches" / f"matches_{league}.json", "w") as f:
        json.dump(matches, f)
    rows = sorted(load(ROOT / cfg["train_manifest"]) + load(ROOT / cfg["validation_manifest"]), key=lambda r: r["kickoff"])
    keep = {e["matchId"] for e in raw_events}
    rows = [r for r in rows if r["match_id"] in keep]
    for name, part, split in (("train", rows[:n_train], "model_train"), ("validation", rows[n_train:], "model_validation"),
                              ("nontest", rows, "nontest")):
        with open(root / f"{name}.json", "w") as f:
            json.dump([{**r, "split": split} for r in part], f)
    d = {**data_cfg, "raw_dir": str(raw), "train_manifest": str(root / "nontest.json"), "manifest_split": "nontest",
         **{k: str(root / tag / Path(data_cfg[k]).name) for k in ("actions_path", "onpitch_path", "player_match_path",
                                                                  "coinciding_path", "build_summary_path", "who_path",
                                                                  "causal_counts_path")},
         "position_overrides": [p for p in data_cfg["position_overrides"] if p["event_id"] in {e["id"] for e in raw_events}],
         "card_overrides": [c for c in data_cfg["card_overrides"] if c["match_id"] in keep],
         "goal_overrides": [g for g in data_cfg["goal_overrides"] if g["match_id"] in keep]}
    with open(root / f"{tag}_data.json", "w") as f:
        json.dump(d, f)
    mcfg = {**cfg, "data_config": str(root / f"{tag}_data.json"), "train_manifest": str(root / "train.json"),
            "validation_manifest": str(root / "validation.json"), "out_dir": str(root / tag / "model")}
    with open(root / f"{tag}_model.json", "w") as f:
        json.dump(mcfg, f)
    for script, conf in (("build_actions.py", f"{tag}_data.json"), ("build_model_data.py", f"{tag}_model.json")):
        run = subprocess.run([sys.executable, str(ROOT / "src" / script), "--config", str(root / conf)],
                             capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"{script} failed on the {tag} scope:\n{run.stderr[-3000:]}")
    return load_scope(root / tag / "model")


def inputs_equal(a, b, rows=None):
    """True when the model inputs of two builds agree (all targets, or the given target rows)."""
    if rows is None:
        return (a["E"].equals(b["E"]) and a["T"].equals(b["T"]) and np.array_equal(a["W"], b["W"])
                and all(np.array_equal(a["C"][k], b["C"][k], equal_nan=True) for k in a["C"]))
    ok = a["T"].iloc[rows].reset_index(drop=True).equals(b["T"].iloc[rows].reset_index(drop=True))
    ok &= np.array_equal(a["W"][rows], b["W"][rows])
    for i in rows:
        ra, rb = np.flatnonzero(a["C"]["act"] == i), np.flatnonzero(b["C"]["act"] == i)
        ok &= all(np.array_equal(a["C"][k][ra], b["C"][k][rb], equal_nan=True) for k in a["C"])
        ok &= a["E"].set_index("seq").loc[a["W"][i][a["W"][i] >= 0]].equals(b["E"].set_index("seq").loc[b["W"][i][b["W"][i] >= 0]])
    return ok


def perturbation_checks(cfg, data_cfg, D, results, rng, tmp):
    league = data_cfg["league"]
    M = D["M"]
    pick = list(M[M["block"] == "model_train"]["match_id"].iloc[[0, 40]]) + [M[M["block"] == "model_validation"]["match_id"].iloc[0]]
    events = [e for e in load(ROOT / data_cfg["raw_dir"] / "events" / f"events_{league}.json") if e["matchId"] in set(pick)]
    base = mini_scope(cfg, data_cfg, tmp, events, 2, 1, "base")

    noisy = copy.deepcopy(events)
    for e in noisy:
        if len(e["positions"]) > 1:
            e["positions"][1] = {"x": int(rng.integers(0, 101)), "y": int(rng.integers(0, 101))}
        keep = [t for t in e["tags"] if t["id"] in DERIVED_TAGS]
        extra = rng.choice([301, 302, 1801, 1802, 1901, 2001, 501, 502, 503, 504], size=int(rng.integers(0, 3)), replace=False)
        e["tags"] = keep + [{"id": int(t)} for t in extra]
    other = mini_scope(cfg, data_cfg, tmp, noisy, 2, 1, "noisy")
    check(results, 10, inputs_equal(base, other), "end points and non-derived tags of every record changed: inputs identical")

    # 11: future invariance for sampled targets of the validation match
    m_val = pick[2]
    E0 = base["E"]
    order = {(m, s): eid for m, s, eid in zip(E0["match_id"], E0["seq"], E0["event_id"])}
    rows = np.flatnonzero(base["T"]["match_id"].to_numpy() == m_val)
    ok_future, ok_control = True, True
    for i in rng.choice(rows, size=6, replace=False):
        s = int(base["T"]["seq"].iat[i])
        later = {order[(m_val, q)] for q in range(s + 1, int(E0.loc[E0["match_id"] == m_val, "seq"].max()) + 1)}
        target_id = int(base["T"]["event_id"].iat[i])
        changed = copy.deepcopy(events)
        for e in changed:
            # goal and card records are kept: the build checks rebuilt scores and red cards against the match data
            if e["id"] in later and not {t["id"] for t in e["tags"]} & {101, 102, 1701, 1703}:
                e["positions"][0] = {"x": int(rng.integers(0, 101)), "y": int(rng.integers(0, 101))}
                e["eventName"], e["subEventName"], e["subEventId"], e["eventId"] = "Pass", "Simple pass", 85, 8
                e["tags"] = [{"id": 1801}]
            if e["id"] == target_id:
                if e["eventName"] == "Pass":
                    e["eventName"], e["subEventName"], e["subEventId"], e["eventId"] = "Others on the ball", "Touch", 72, 7
                else:
                    e["eventName"], e["subEventName"], e["subEventId"], e["eventId"] = "Pass", "Simple pass", 85, 8
                e["tags"] = [t for t in e["tags"] if t["id"] in {101, 102, 1701, 1703}] + [{"id": 1802}, {"id": 302}]
                if len(e["positions"]) > 1:
                    e["positions"][1] = {"x": 50, "y": 50}
                e["playerId"] = int(rng.choice([p for p in base["C"]["cand"][base["C"]["act"] == i]]))
        fut = mini_scope(cfg, data_cfg, tmp, changed, 2, 1, "future")
        j = int(np.flatnonzero((fut["T"]["event_id"] == target_id).to_numpy())[0])
        ok_future &= j == i and inputs_equal(base, fut, [i])
        earlier = copy.deepcopy(events)
        prev_ids = [order[(m_val, q)] for q in range(max(0, s - 5), s)]
        for e in earlier:
            if e["id"] in prev_ids:
                e["positions"][0] = {"x": int(rng.integers(0, 101)), "y": int(rng.integers(0, 101))}
        ctrl = mini_scope(cfg, data_cfg, tmp, earlier, 2, 1, "control")
        jj = np.flatnonzero((ctrl["T"]["event_id"] == target_id).to_numpy())
        ok_control &= len(jj) == 1 and not inputs_equal(base, ctrl, [int(jj[0])])
    check(results, 11, ok_future and ok_control,
          "6 validation targets: later records and the target's own type, tags, end and actor changed -> inputs identical; "
          "earlier records changed -> inputs differ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    data_cfg = load(ROOT / cfg["data_config"])
    out = ROOT / cfg["out_dir"]
    rng = np.random.default_rng(args.seed)
    D = load_scope(out)
    results = {}
    static_checks(cfg, data_cfg, D, results, rng)
    history_check(data_cfg, D, results, rng)
    perturbation_checks(cfg, data_cfg, D, results, rng, ROOT / "results" / "model_checks" / data_cfg["league"])
    with open(out / "checks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("all checks passed; written", out / "checks.json", flush=True)


if __name__ == "__main__":
    main()
