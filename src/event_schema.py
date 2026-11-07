"""Wyscout event definitions, goal reconstruction and per-match position summaries."""
import json
import math
from collections import defaultdict
import numpy as np


ON_BALL = {"Pass", "Others on the ball", "Shot", "Free Kick", "Save attempt", "Goalkeeper leaving line"}
ACTIONS = ["short_pass", "long_pass", "cross", "through_ball", "head_pass", "carry", "take_on", "shot", "clearance", "touch"]
ACTION_OF = {
    ("Pass", "Simple pass"): "short_pass",
    ("Pass", "High pass"): "long_pass",
    ("Pass", "Launch"): "long_pass",
    ("Pass", "Cross"): "cross",
    ("Pass", "Smart pass"): "through_ball",
    ("Pass", "Head pass"): "head_pass",
    ("Others on the ball", "Acceleration"): "carry",
    ("Duel", "Ground attacking duel"): "take_on",
    ("Shot", "Shot"): "shot",
    ("Others on the ball", "Clearance"): "clearance",
    ("Others on the ball", "Touch"): "touch",
}
ROLES = ["GK", "DF", "MD", "FW"]
PERIOD_ORDER = {"1H": 0, "2H": 1}
DEPTH = [(-10.0, "deep"), (-2.0, "backmid"), (8.0, "mid"), (math.inf, "front")]  # upper bounds of dx
LATERAL = [(22.0, "wideL"), (42.0, "halfL"), (58.0, "centre"), (78.0, "halfR"), (math.inf, "wideR")]
FINE_ROLES = ["GK"] + [f"{d}/{l}" for _, d in DEPTH for _, l in LATERAL] + ["unassigned"]
POSITION_EXCLUDED = {"Interruption", "Save attempt", "Goalkeeper leaving line"}

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tags_of(e):
    return {t["id"] for t in e["tags"]}


def is_context(e):
    """On-ball events that define who has the ball; duels count only through the attacker's record."""
    return e["eventName"] in ON_BALL or e["subEventName"] == "Ground attacking duel"


def opponent_level(e):
    name, sub = e["eventName"], e["subEventName"]
    if name == "Pass":
        return "opp_pass"
    if name == "Shot" or sub in ("Free kick shot", "Penalty"):
        return "opp_shot"
    if name in ("Save attempt", "Goalkeeper leaving line"):
        return "opp_goalkeeper"
    if name == "Free Kick":
        return "opp_set_piece"
    if sub == "Clearance":
        return "opp_clearance"
    return "opp_other_on_ball"


def opp_last_level(e):
    """Type of the opponent's most recent event (interruptions excluded), first matching rule wins."""
    if e is None:
        return "none"
    name, sub = e["eventName"], e["subEventName"]
    if name == "Foul":
        return "foul"
    if name == "Duel":
        return "duel"
    if 1401 in tags_of(e):
        return "interception"
    if sub == "Clearance":
        return "clearance"
    if name == "Shot" or sub in ("Free kick shot", "Penalty"):
        return "shot"
    if name == "Pass":
        return "pass"
    if name == "Free Kick":
        return "set_piece"
    if name in ("Save attempt", "Goalkeeper leaving line"):
        return "goalkeeper"
    if name == "Offside":
        return "offside"
    return "other_on_ball"


def steps_level(k):
    if k <= 2:
        return str(k)
    return "3-5" if k <= 5 else "6+"


def match_minute(period, sec):
    if period == "1H":
        return min(sec / 60.0, 45.0)
    return 45.0 + sec / 60.0


def band(value, bands):
    return next(name for upper, name in bands if value < upper)


def goals_of_match(events, home, away, overrides):
    """(event index, scoring team) of every goal: tag 101 on shots and free kicks, tag 102 (own goal),
    plus the configured overrides. Tag 101 on save attempts is not used: it also appears without a goal."""
    goals = []
    for i, e in enumerate(events):
        tags = tags_of(e)
        if 102 in tags:
            goals.append((i, away if e["teamId"] == home else home))
        elif 101 in tags and e["eventName"] in ("Shot", "Free Kick"):
            goals.append((i, e["teamId"]))
        elif e["id"] in overrides:
            goals.append((i, overrides[e["id"]]))
    return sorted(goals)


def player_match_roles(events, roles, min_events):
    """Mean event position per player in the match, depth relative to his team's outfield mean, fine role,
    and each team's back-line size (outfield players with dx < -10 and 22 <= y < 78)."""
    pos = defaultdict(list)
    team_of = {}
    for e in events:
        if e["playerId"] and e["eventName"] not in POSITION_EXCLUDED and e["subEventName"] != "Goal kick":
            pos[e["playerId"]].append((e["positions"][0]["x"], e["positions"][0]["y"]))
            team_of[e["playerId"]] = e["teamId"]
    stats = {}
    for pid, xy in pos.items():
        a = np.asarray(xy, dtype=float)
        stats[pid] = {"team_id": team_of[pid], "n_events": len(a), "mean_x": a[:, 0].mean(), "mean_y": a[:, 1].mean()}
    outfield = {pid: s for pid, s in stats.items() if roles[pid] != "GK" and s["n_events"] >= min_events}
    team_x = defaultdict(list)
    for s in outfield.values():
        team_x[s["team_id"]].append(s["mean_x"])
    back_line = defaultdict(int)
    for pid, s in stats.items():
        if roles[pid] == "GK":
            s["fine_role"], s["dx"] = "GK", float("nan")
        elif pid in outfield:
            s["dx"] = s["mean_x"] - float(np.mean(team_x[s["team_id"]]))
            s["fine_role"] = f"{band(s['dx'], DEPTH)}/{band(s['mean_y'], LATERAL)}"
            if s["dx"] < -10.0 and 22.0 <= s["mean_y"] < 78.0:
                back_line[s["team_id"]] += 1
        else:
            s["fine_role"], s["dx"] = "unassigned", float("nan")
    return stats, {t: ("bl3" if k >= 3 else "bl2") for t, k in back_line.items()}
