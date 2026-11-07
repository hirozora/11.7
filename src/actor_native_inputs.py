"""Causal Wyscout-v2 inputs for local Seq2Event, NMSTPP and Unified-LEM.

Consume observed records only; remove coinciding records BEFORE deriving features.
No player ID, target action/end position or cross-match summary is read here.
"""
import math
import numpy as np

SEQ_ACTION = {"@": 0, "_": 1, "d": 2, "g": 3, "p": 4, "s": 5, "x": 6}
NM_ACTION = {"p": 0, "_": 1, "d": 2, "x": 3, "s": 4}
CENTERS = np.array(list(zip(
    [8.5, 25.25, 41.75, 58.25, 74.75, 91.5] * 2 + [33.5, 66.5] * 3 + [8.5, 91.5],
    [89.45] * 6 + [10.55] * 6 + [71.05] * 2 + [50.] * 2 + [28.95] * 2 + [50.] * 2)))
LEM_TYPES = ['pass', 'long_pass', 'cross', 'touch', 'aerial_duel', 'clearance', 'interception',
             'loose_ball_duel', 'defensive_duel', 'offensive_duel', 'dribble', 'carry',
             'game_interruption', 'own_goal', 'throw_in', 'free_kick', 'goal_kick', 'infraction',
             'corner', 'acceleration', 'offside', 'right_foot_shot', 'left_foot_shot', 'head_shot',
             'goalkeeper_exit', 'save', 'shot_against', 'fairplay', 'yellow_card', 'red_card',
             'first_half_end', 'game_end']


def initial_action(e, tags):
    name, sub = e["eventName"], e["subEventName"]
    if name in ("Foul", "Offside"):
        return "foul"
    if name == "Pass":
        return "x" if sub == "Cross" else "p"
    if name == "Shot":
        return "s"
    if name == "Free Kick":
        return {"Corner": "x", "Free kick cross": "x", "Free kick shot": "s", "Penalty": "s",
                "Goal kick": "p", "Throw in": "p", "Free Kick": "p"}[sub]
    if sub == "Ground attacking duel" and tags & {501, 502, 503, 504}:
        return "d"
    if sub == "Clearance":
        return "p"
    if sub == "Acceleration" or (sub == "Touch" and not tags):
        return "d"
    return None  # Native event filter, not an unknown-action substitution.


def lem_type(e, tags):
    name, sub = e["eventName"], e["subEventName"]
    if tags & {1701, 1703}:
        value = "red_card"
    elif 1702 in tags:
        value = "yellow_card"
    elif 102 in tags:
        value = "own_goal"
    elif name == "Shot" or sub in ("Free kick shot", "Penalty"):
        value = "left_foot_shot" if 401 in tags else "head_shot" if 403 in tags else "right_foot_shot"
    elif name == "Duel":
        value = {"Air duel": "aerial_duel", "Ground attacking duel": "offensive_duel",
                 "Ground defending duel": "defensive_duel", "Ground loose ball duel": "loose_ball_duel"}[sub]
        if sub == "Ground attacking duel" and tags & {501, 502, 503, 504}:
            value = "dribble"
    elif name == "Pass":
        value = "cross" if sub == "Cross" else "long_pass" if sub in ("High pass", "Launch") else "pass"
    elif 1401 in tags:
        value = "interception"
    elif name == "Free Kick":
        value = {"Throw in": "throw_in", "Goal kick": "goal_kick", "Corner": "corner",
                 "Free Kick": "free_kick", "Free kick cross": "cross"}[sub]
    elif name == "Others on the ball":
        value = {"Acceleration": "acceleration", "Clearance": "clearance", "Touch": "touch"}[sub]
    else:
        value = {"Foul": "infraction", "Offside": "offside", "Interruption": "game_interruption",
                 "Goalkeeper leaving line": "goalkeeper_exit", "Save attempt": "save"}[name]
    return LEM_TYPES.index(value)


class NativeState:
    def __init__(self, home, away, goal_overrides):
        self.home, self.away, self.overrides = home, away, goal_overrides
        self.seq, self.nm, self.lem = [], [], []
        self.owner = None
        self.counts = np.zeros(6)  # home/away goals, reds, yellows
        self.last_clock = 0.
        self.context = np.zeros(9)

    def snapshot(self):
        return (len(self.seq), len(self.nm), len(self.lem), self.owner,
                self.counts.copy(), self.last_clock, self.context.copy())

    def branch(self, snapshot):
        """Copy the unaffected prefix, then replay from the earliest excluded record."""
        a, b, c, owner, counts, clock, context = snapshot
        out = NativeState(self.home, self.away, self.overrides)
        out.seq, out.nm, out.lem = self.seq[:a], self.nm[:b], self.lem[:c]
        out.owner, out.counts, out.last_clock, out.context = owner, counts.copy(), clock, context.copy()
        return out

    def sequence_token(self, code, team, clock, x, y, score):
        # Deltas are computed in the match stream BEFORE the Seq2Event team filter.
        previous = self.seq[-1] if self.seq else None
        dx = x - previous[3][1] if previous else 0.
        dy = y - previous[3][2] if previous else 0.
        dt = max(0., clock - previous[3][0]) if previous else 0.
        sg = math.hypot((x - 1) * 1.05, (y - .5) * .68)
        theta = abs(math.atan2((y - .5) * .68, (x - 1) * 1.05))
        cont = [clock, x, y, sg, theta, score, dt, dx, dy, math.hypot(dx * 1.05, dy * .68)]
        self.seq.append((team, code, None, cont))
        if code == "g":
            return
        zone = int(np.argmin(((CENTERS - [x * 100, y * 100]) ** 2).sum(1)))
        cx, cy = CENTERS[zone]
        if self.nm:
            old = self.nm[-1]
            px, py = CENTERS[old[1]]
            gap = np.clip(clock - old[2], 0, 60)
        else:
            px, py, gap = cx, cy, 0.
        # Reversed delta names/scales as in NMSTPP's released feature construction.
        zx, zy = cy - py, cx - px
        zc = [gap, math.hypot(zx * 1.05, zy * .68), zy, zx,
              math.hypot((cx - 100) * 1.05, (cy - 50) * .68),
              abs(math.atan2((cy - 50) * .68, (cx - 100) * 1.05))]
        self.nm.append((code, zone, clock, zc))

    def consume(self, e):
        tags = {t["id"] for t in e["tags"]}
        period = {"1H": 0, "2H": 1}[e["matchPeriod"]]
        team = e["teamId"]
        if team not in (self.home, self.away):
            raise ValueError("event team outside the two match teams")
        x, y = e["positions"][0]["x"], e["positions"][0]["y"]
        if not (0 <= x <= 100 and 0 <= y <= 100):
            raise ValueError("invalid Wyscout start coordinate")
        sec = math.floor(e["eventSec"])
        clock, seq_clock = period * 2700 + sec, period * 3600 + sec
        side = int(team == self.away)
        scorer = None
        if 102 in tags:
            scorer = 1 - side
        elif 101 in tags and e["eventName"] in ("Shot", "Free Kick"):
            scorer = side
        elif e["id"] in self.overrides:
            scorer = int(self.overrides[e["id"]] == self.away)
        act = initial_action(e, tags)
        if act is not None and act != "foul":
            if self.owner is not None and team != self.owner and self.seq:
                prev = self.seq[-1]
                self.sequence_token("_", prev[0], prev[3][0], prev[3][1], prev[3][2], prev[3][5])
            self.owner = team
            self.sequence_token(act, team, seq_clock, x / 100, y / 100, self.counts[side] - self.counts[1 - side])
        if scorer is not None:
            self.counts[scorer] += 1
            # Seq2Event has separate goal tokens; NMSTPP removes these tokens.
            self.sequence_token("g", team, seq_clock, x / 100, y / 100,
                                self.counts[side] - self.counts[1 - side])
        self.counts[2 + side] += bool(tags & {1701, 1703})
        self.counts[4 + side] += bool(tags & {1702, 1703})
        shot = e["eventName"] == "Shot" or e["subEventName"] in ("Free kick shot", "Penalty")
        accurate = int(101 in tags) if shot else int(1801 in tags)
        delta = np.clip(clock - self.last_clock, 0, 100) if self.lem else 0
        self.lem.append([1 - side, lem_type(e, tags), x, y, delta, accurate])
        self.last_clock = clock
        self.context = np.r_[period, clock // 60, clock % 60, self.counts]

    def view(self, team, windows):
        seq = [r for r in self.seq if r[0] == team][-windows["seq2event"]:]
        nm = self.nm[-windows["nmstpp"]:]
        result = {}
        for name, values, width, codes in (("seq2event", seq, 10, SEQ_ACTION), ("nmstpp", nm, 6, NM_ACTION)):
            n, k = len(values), windows[name]
            cat, cont, valid = np.zeros(k, np.int16), np.zeros((k, width), np.float32), np.zeros(k, bool)
            zone = np.zeros(k, np.int16)
            if n:
                cat[-n:] = [codes[r[1] if name == "seq2event" else r[0]] for r in values]
                cont[-n:] = [r[3] for r in values]
                valid[-n:] = True
                if name == "nmstpp":
                    zone[-n:] = [r[1] for r in values]
            result[name] = dict(actions=cat, continuous=cont, valid=valid)
            if name == "nmstpp":
                result[name]["zones"] = zone
        past = np.full((windows["lem"], 6), -1., np.float32)
        observed = self.lem[-windows["lem"]:][::-1]
        if observed:
            past[:len(observed)] = observed
        # All current-event fields unknown; query time/start enter the actor head separately.
        result["lem"] = {"tabular": np.r_[np.full(6, -1), self.context, past.ravel()].astype(np.float32)}
        return result


def match_views(events, targets, exclusions, home, away, overrides, windows):
    """Yield (target row, inputs), independent of target/future record contents."""
    state = NativeState(home, away, overrides)
    snapshots, cursor = [], 0
    for row, seq, team in sorted(targets, key=lambda r: r[1]):
        while cursor < seq:
            snapshots.append(state.snapshot())
            state.consume(events[cursor])
            cursor += 1
        removed = {s for s in exclusions.get(row, ()) if s < seq}
        view = state
        if removed:
            lo = min(removed)
            view = state.branch(snapshots[lo])
            for j in range(lo, seq):
                if j not in removed:
                    view.consume(events[j])
        yield row, view.view(team, windows)
