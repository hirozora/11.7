"""Build target actions, possession acquisition, overlap pairs and exact on-pitch intervals.
See docs/DATA.md for the input contract.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from event_schema import (ACTION_OF, FINE_ROLES, PERIOD_ORDER, ROLES, goals_of_match, is_context, load, match_minute,
                           opp_last_level, opponent_level, player_match_roles, steps_level, tags_of)

ROOT = Path(__file__).resolve().parents[1]
CARD_TAGS = {1701, 1703}  # red card, second yellow


def acquisition_level(first, between, prev_ctx, team, use_interception):
    """Classify acquisition from preceding records and the explicitly permitted first-event tag."""
    if prev_ctx is None:
        return "period_start"
    if first["eventName"] == "Free Kick":
        return "restart_out" if first["subEventName"] in ("Throw in", "Corner", "Goal kick") else "restart_foul"
    if use_interception and 1401 in tags_of(first):
        return "interception"
    for e in between:
        if e["eventName"] == "Duel":
            t = tags_of(e)
            if (e["teamId"] == team and 703 in t) or (e["teamId"] != team and 701 in t):
                return "duel_won"
    level = opponent_level(prev_ctx)
    if level == "opp_shot":
        return "after_opp_shot"
    if level == "opp_clearance":
        return "opp_clearance"
    if level == "opp_pass":
        return "opp_pass_lost"
    return "other"


def validate_card_overrides(events_by_match, overrides):
    """Resolve explicitly listed dismissals; require their match and player identities."""
    fixed = set()
    for item in overrides:
        found = [e for e in events_by_match[item["match_id"]] if e["id"] == item["event_id"]]
        if len(found) != 1 or found[0]["playerId"] != item["player_id"]:
            raise ValueError(f"card override does not identify one selected event: {item['event_id']}")
        if item["event_id"] in fixed:
            raise ValueError(f"duplicate card override: {item['event_id']}")
        fixed.add(item["event_id"])
    return fixed


def onpitch_intervals(match, roles, events, card_overrides):
    """Start and end of every player who appeared; red cards end at the card event (exact time kept separately)."""
    card = {}
    for e in events:
        if (tags_of(e) & CARD_TAGS or e["id"] in card_overrides) and e["playerId"] not in card:
            card[e["playerId"]] = (PERIOD_ORDER[e["matchPeriod"]], float(e["eventSec"]), match_minute(e["matchPeriod"], e["eventSec"]))
    rows, listed = [], set()
    for team_key, data in match["teamsData"].items():
        formation = data["formation"]
        subs = formation["substitutions"]
        if subs == "null":
            subs = []
        sub_in = {s["playerIn"]: float(s["minute"]) for s in subs}
        sub_out = {s["playerOut"]: float(s["minute"]) for s in subs}
        for group, starter in ((formation["lineup"], True), (formation["bench"], False)):
            for p in group:
                pid = p["playerId"]
                if not starter and pid not in sub_in:
                    continue
                if p["redCards"] != "0" and pid not in card:
                    raise ValueError(f"match {match['wyId']}: red card of player {pid} without a card event")
                listed.add(pid)
                sub_end = sub_out.get(pid, math.inf)
                c = card.get(pid)
                rows.append({"match_id": match["wyId"], "team_id": int(team_key), "player_id": pid, "role": roles[pid],
                             "start": 0.0 if starter else sub_in[pid], "sub_end": sub_end,
                             "end": min(sub_end, c[2]) if c else sub_end,
                             "card_period": c[0] if c else -1, "card_sec": c[1] if c else float("nan")})
    return rows, len(set(card) - listed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    raw = ROOT / cfg["raw_dir"]
    tol = cfg["onpitch_tolerance_minutes"]
    T, M = cfg["coincide_seconds"], cfg["coincide_tolerance"]
    own_in_ps, use_ic = cfg["exclude_own_in_possession_start"], cfg["first_action_interception"]
    overrides = {g["event_id"]: g["scorer_team"] for g in cfg["goal_overrides"]}

    manifest = load(ROOT / cfg["train_manifest"])
    if any(r["split"] != cfg["manifest_split"] for r in manifest):
        raise ValueError(f"manifest contains rows outside split {cfg['manifest_split']}")
    kickoff = {r["match_id"]: r["kickoff"] for r in manifest}
    order = sorted(kickoff, key=lambda m: kickoff[m])
    half = {m: i % 2 for i, m in enumerate(order)}
    matches = {x["wyId"]: x for x in load(raw / "matches" / f"matches_{cfg['league']}.json")}
    missing = set(kickoff) - set(matches)
    if missing:
        raise ValueError(f"manifest matches not in raw matches: {sorted(missing)[:5]}")
    team_name = {t["wyId"]: t["name"] for t in load(raw / "entities" / "teams.json")}
    roles = {p["wyId"]: p["role"]["code2"] for p in load(raw / "entities" / "players.json")}
    big_six = {tid for tid, name in team_name.items() if name in set(cfg["big_six"])}
    if len(big_six) != 6:
        raise ValueError(f"expected 6 big-six teams, found {len(big_six)}")
    position_fix = {p["event_id"]: (p["x"], p["y"]) for p in cfg["position_overrides"]}
    position_fixed = set()
    events_by_match = defaultdict(list)
    for e in load(raw / "events" / f"events_{cfg['league']}.json"):
        if e["matchId"] in kickoff:
            if e["id"] in position_fix:
                e["positions"][0] = {"x": position_fix[e["id"]][0], "y": position_fix[e["id"]][1]}
                position_fixed.add(e["id"])
            events_by_match[e["matchId"]].append(e)
    if position_fixed != set(position_fix):
        raise ValueError(f"position overrides not found in the selected matches: {sorted(set(position_fix) - position_fixed)}")
    card_fix = validate_card_overrides(events_by_match, cfg["card_overrides"])

    for key in ("actions_path", "onpitch_path", "player_match_path", "coinciding_path", "build_summary_path"):
        (ROOT / cfg[key]).parent.mkdir(parents=True, exist_ok=True)
    action_rows, onpitch_rows, player_match_rows, pair_rows = [], [], [], []
    actor_outside, cards_unlisted = 0, 0
    unattributed = defaultdict(int)
    for mid in order:
        match = matches[mid]
        side = {int(k): v["side"] for k, v in match["teamsData"].items()}
        home = next(t for t, s in side.items() if s == "home")
        away = next(t for t, s in side.items() if s == "away")
        events = events_by_match[mid]
        periods = {e["matchPeriod"] for e in events}
        if not periods <= set(PERIOD_ORDER):
            raise ValueError(f"match {mid} has periods {periods}")
        events = sorted(events, key=lambda e: (PERIOD_ORDER[e["matchPeriod"]], e["eventSec"]))  # stable: file order on ties
        goals = goals_of_match(events, home, away, overrides)
        final = {home: sum(t == home for _, t in goals), away: sum(t == away for _, t in goals)}
        expected = {int(k): v["score"] for k, v in match["teamsData"].items()}
        if final != expected:
            raise ValueError(f"match {mid}: rebuilt score {final} != recorded {expected}")

        n = len(events)
        per = np.array([PERIOD_ORDER[e["matchPeriod"]] for e in events])
        sec = np.array([float(e["eventSec"]) for e in events])
        tm = np.array([e["teamId"] for e in events])
        xy = np.array([[e["positions"][0]["x"], e["positions"][0]["y"]] for e in events], dtype=float)
        ctx = np.array([is_context(e) for e in events])
        interruption = np.array([e["eventName"] == "Interruption" for e in events])
        shot = np.array([e["eventName"] == "Shot" or e["subEventName"] in ("Free kick shot", "Penalty") for e in events])
        minutes_all = np.array([match_minute(e["matchPeriod"], e["eventSec"]) for e in events])
        ctx_pos = np.flatnonzero(ctx)
        shot_pos = np.flatnonzero(shot)
        goal_list = sorted(goals)

        def window(idx_arr, i, minute, team):
            """Own and opponent events of idx_arr before i within the momentum window ."""
            p = np.searchsorted(idx_arr, i)
            q = np.searchsorted(minutes_all[idx_arr[:p]], minute - cfg["momentum_minutes"])
            own = int((tm[idx_arr[q:p]] == team).sum())
            return own, int(p - q) - own

        # coinciding records sorted before each event, same period, within the window
        co_opp, co_own = [], []
        for i in range(n):
            lo = i
            while lo > 0 and per[lo - 1] == per[i] and sec[i] - sec[lo - 1] <= T:
                lo -= 1
            js = np.arange(lo, i)
            opp = tm[js] != tm[i]
            mirror = np.maximum(np.abs(xy[i, 0] - (100 - xy[js, 0])), np.abs(xy[i, 1] - (100 - xy[js, 1])))
            same = np.maximum(np.abs(xy[i, 0] - xy[js, 0]), np.abs(xy[i, 1] - xy[js, 1]))
            co_opp.append(set(js[opp & (mirror <= M)].tolist()))
            co_own.append(set(js[~opp & ~ctx[js] & (same <= M)].tolist()))

        def prev_ctx(p, excl):
            """Last context event before p in p's period, skipping excl."""
            k = int(np.searchsorted(ctx_pos, p)) - 1
            while k >= 0 and per[ctx_pos[k]] == per[p]:
                if ctx_pos[k] not in excl:
                    return int(ctx_pos[k])
                k -= 1
            return None

        def run_of(c, excl):
            """Consecutive context events of c's team ending at c, skipping excl: their number and the first one."""
            k = int(np.searchsorted(ctx_pos, c))
            count, first = 0, c
            while k >= 0 and per[ctx_pos[k]] == per[c]:
                j = int(ctx_pos[k])
                if j not in excl:
                    if tm[j] != tm[c]:
                        break
                    count, first = count + 1, j
                k -= 1
            return count, first

        def between(c, p, excl):
            return [events[j] for j in range(c + 1, p) if not ctx[j] and j not in excl]

        def mode_at(s, extra):
            """Acquisition mode of a possession whose first context event is s (possession level)."""
            excl = co_opp[s] | extra | (co_own[s] if own_in_ps else set())
            c = prev_ctx(s, excl)
            if c is None:
                return "period_start"
            if tm[c] == tm[s]:
                return "other"
            return acquisition_level(events[s], between(c, s, excl), events[c], tm[s], use_ic)

        stats, back_line = player_match_roles(events, roles, cfg["fine_role_min_events"])
        intervals, unlisted = onpitch_intervals(match, roles, events, card_fix)
        cards_unlisted += unlisted
        for r in intervals:
            s = stats.get(r["player_id"])
            r["fine_role"] = s["fine_role"] if s else ("GK" if r["role"] == "GK" else "unassigned")
        onpitch_rows.extend(intervals)
        for pid, s in stats.items():
            player_match_rows.append({"match_id": mid, "player_id": pid, "role": roles[pid], **s,
                                      "back_line": back_line.get(s["team_id"], "bl2")})
        by_team = defaultdict(list)
        for r in intervals:
            by_team[r["team_id"]].append(r)
        fine_of = {r["player_id"]: r["fine_role"] for r in intervals}

        period, last_team, possession_id, poss_first = None, None, -1, None
        acq_cache = {}
        for i, e in enumerate(events):
            if e["matchPeriod"] != period:
                period, last_team = e["matchPeriod"], None
            team = e["teamId"]
            if ctx[i] and team != last_team:
                possession_id += 1
                poss_first = i
            action = ACTION_OF.get((e["eventName"], e["subEventName"]))
            if action is not None and e["playerId"] == 0:
                unattributed[action] += 1  # Wyscout leaves some duel records without a player
            elif action is not None:
                opponent = away if team == home else home
                E = co_opp[i]
                for j in sorted(co_opp[i] | co_own[i]):
                    pair_rows.append({"match_id": mid, "target_event_id": e["id"], "record_event_id": events[j]["id"],
                                      "side": "opponent" if tm[j] != team else "own", "dt": float(sec[i] - sec[j])})
                c = prev_ctx(i, E)
                x, y = e["positions"][0]["x"], e["positions"][0]["y"]
                if c is None:
                    prev_event, own_run, ps, psx, first_in_view = "period_start", 0, "period_start", float(x), True
                elif tm[c] == team:
                    own_run, s = run_of(c, E)
                    prev_event, ps, psx, first_in_view = "own", mode_at(s, E), float(xy[s, 0]), False
                else:
                    excl_ps = E | (co_own[i] if own_in_ps else set())
                    prev_event, own_run, first_in_view = opponent_level(events[c]), 0, True
                    ps = acquisition_level(e, between(c, i, excl_ps), events[c], team, use_interception=False)
                    psx = float(x)
                if poss_first not in acq_cache:
                    acq_cache[poss_first] = mode_at(poss_first, set())
                phase = "first" if own_run == 0 else ("early" if own_run <= 3 else "settled")
                if not (0 <= x <= 100 and 0 <= y <= 100):
                    raise ValueError(f"event {e['id']} start outside the pitch: {x}, {y}")
                zone = 3 * min(int(x * 3 / 100), 2) + min(int(y * 3 / 100), 2)
                minute = match_minute(e["matchPeriod"], e["eventSec"])
                if e["matchPeriod"] == "1H":
                    time_bin = "first_half"
                else:
                    time_bin = "second_half_to_75" if e["eventSec"] < 1800 else "after_75"
                g_before = [(g, t) for g, t in goal_list if g < i and g not in E]
                own = sum(1 for _, t in g_before if t == team)
                opp = len(g_before) - own
                score_state = "leading" if own > opp else ("drawing" if own == opp else "trailing")
                last, state, so, sp = 0.0, 0, 0, 0
                for g, t in g_before:
                    so, sp = so + (t == team), sp + (t != team)
                    if int(np.sign(so - sp)) != state:
                        state, last = int(np.sign(so - sp)), minutes_all[g]
                own_ctx, opp_ctx = window(ctx_pos, i, minute, team)
                own_shots, opp_shots = window(shot_pos, i, minute, team)
                in_w = [j for j in E if minutes_all[j] >= minute - cfg["momentum_minutes"]]  # opponent records only
                opp_ctx -= sum(1 for j in in_w if ctx[j])
                opp_shots -= sum(1 for j in in_w if shot[j])
                oj = i - 1
                while oj >= 0 and per[oj] == per[i] and not (tm[oj] == opponent and not interruption[oj] and oj not in E):
                    oj -= 1
                opp_last = events[oj] if oj >= 0 and per[oj] == per[i] else None
                on = [r for r in by_team[team]
                      if r["start"] - tol <= minute < r["sub_end"] + tol
                      and (r["card_period"] < 0 or (per[i], sec[i]) < (r["card_period"], r["card_sec"]))]
                if e["playerId"] not in {r["player_id"] for r in on}:
                    actor_outside += 1
                counts = {role: sum(1 for r in on if r["role"] == role) for role in ROLES}
                fine_counts = {f: 0 for f in FINE_ROLES}
                for r in on:
                    fine_counts[r["fine_role"]] += 1
                action_rows.append({
                    "match_id": mid, "kickoff": kickoff[mid], "half": half[mid], "gameweek": match["gameweek"],
                    "event_id": e["id"], "period": e["matchPeriod"], "event_sec": float(e["eventSec"]), "minute": minute,
                    "team_id": team, "opponent_id": opponent, "player_id": e["playerId"],
                    "actor_role": roles[e["playerId"]], "zone": zone, "action": action,
                    "phase": phase, "prev_event": prev_event, "time_bin": time_bin,
                    "score_state": score_state, "venue": side[team],
                    "opponent_class": "big_six" if opponent in big_six else "other",
                    "opp_last_event": opp_last_level(opp_last),
                    "possession_start": ps,
                    "acquisition_mode": acq_cache[poss_first],
                    "steps_since_change": steps_level(own_run),
                    "poss_start_x": psx,
                    "score_diff": own - opp,
                    "minutes_since_goal": minute - (minutes_all[g_before[-1][0]] if g_before else 0.0),
                    "state_duration": minute - last,
                    "poss_share_5": own_ctx / (own_ctx + opp_ctx) if own_ctx + opp_ctx else 0.5,
                    "own_shots_5": own_shots, "opp_shots_5": opp_shots,
                    "possession_id": possession_id,
                    "x": float(x), "y": float(y),
                    "end_x": float(e["positions"][1]["x"]) if len(e["positions"]) > 1 else float("nan"),
                    "end_y": float(e["positions"][1]["y"]) if len(e["positions"]) > 1 else float("nan"),
                    "actor_fine": fine_of.get(e["playerId"], "unassigned"),
                    "back_line": back_line.get(team, "bl2"),
                    "n_coinciding_opponent": len(co_opp[i]), "n_coinciding_own": len(co_own[i]),
                    "first_in_view": first_in_view,
                    **{f"n_on_{role}": counts[role] for role in ROLES},
                    **{f"n_onf_{f}": fine_counts[f] for f in FINE_ROLES},
                })
            if ctx[i]:
                last_team = team

    actions = pd.DataFrame(action_rows)
    actions.to_parquet(ROOT / cfg["actions_path"], index=False)
    pd.DataFrame(onpitch_rows).to_parquet(ROOT / cfg["onpitch_path"], index=False)
    player_match = pd.DataFrame(player_match_rows)
    player_match.to_parquet(ROOT / cfg["player_match_path"], index=False)
    pd.DataFrame(pair_rows, columns=["match_id", "target_event_id", "record_event_id", "side", "dt"]).to_parquet(ROOT / cfg["coinciding_path"], index=False)
    summary = {
        "matches": len(order), "first_kickoff": min(kickoff.values()), "last_kickoff": max(kickoff.values()),
        "actions": len(actions), "action_counts": actions["action"].value_counts().to_dict(),
        "actor_outside_onpitch_interval": actor_outside, "card_events_of_unlisted_players": cards_unlisted,
        "dropped_without_player": dict(unattributed), "scores_checked": len(order),
        "actions_with_coinciding_opponent_record": int((actions["n_coinciding_opponent"] > 0).sum()),
        "actions_with_coinciding_own_record": int((actions["n_coinciding_own"] > 0).sum()),
        "levels": {v: actions[v].value_counts().to_dict() for v in cfg["variables_all"] + ["acquisition_mode"]},
        "actor_fine_role": actions["actor_fine"].value_counts().to_dict(),
        "team_matches_by_back_line": player_match.drop_duplicates(["match_id", "team_id"])["back_line"].value_counts().to_dict(),
    }
    with open(ROOT / cfg["build_summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
