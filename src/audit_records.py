"""Count spatially and temporally overlapping records in the selected non-test matches."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from event_schema import ACTION_OF, ON_BALL, PERIOD_ORDER, load, tags_of

ROOT = Path(__file__).resolve().parents[1]


def label(e):
    if e["eventName"] in ("Pass", "Others on the ball", "Duel", "Free Kick", "Interruption", "Foul"):
        lab = f"{e['eventName']}: {e['subEventName']}"
    else:
        lab = e["eventName"]
    return lab + (" (interception)" if 1401 in tags_of(e) else "")


def is_context(e):
    return e["eventName"] in ON_BALL or e["subEventName"] == "Ground attacking duel"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    windows = [tuple(w) for w in cfg["windows"]]
    dt_max = max(w[0] for w in windows)
    # counts[(side, before, window)][(target type, record label)]; hit[(side, window)][target type] = targets with a record before
    counts = defaultdict(Counter)
    hit = defaultdict(Counter)
    n_targets = Counter()
    same_player = defaultdict(Counter)
    per_league = {}
    for league, manifest in cfg["manifests"].items():
        mids = {r["match_id"] for r in load(ROOT / manifest)}
        by_match = defaultdict(list)
        for e in load((ROOT / cfg["raw_dir"]) / "events" / f"events_{league}.json"):
            if e["matchId"] in mids:
                by_match[e["matchId"]].append(e)
        if set(by_match) != mids:
            raise ValueError(f"{league}: matches without events")
        lg_hit, lg_n = Counter(), 0
        for m, events in by_match.items():
            events = sorted(events, key=lambda e: (PERIOD_ORDER[e["matchPeriod"]], e["eventSec"]))
            period = np.array([PERIOD_ORDER[e["matchPeriod"]] for e in events])
            sec = np.array([e["eventSec"] for e in events])
            xy = np.array([[e["positions"][0]["x"], e["positions"][0]["y"]] for e in events], dtype=float)
            for i, e in enumerate(events):
                kind = ACTION_OF.get((e["eventName"], e["subEventName"]))
                if kind is None or e["playerId"] == 0:
                    continue
                n_targets[kind] += 1
                lg_n += 1
                lo = np.searchsorted(period * 1e6 + sec, period[i] * 1e6 + sec[i] - dt_max, side="left")
                hi = np.searchsorted(period * 1e6 + sec, period[i] * 1e6 + sec[i] + dt_max, side="right")
                found = defaultdict(bool)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    o = events[j]
                    d_t = abs(sec[j] - sec[i])
                    if o["teamId"] != e["teamId"]:
                        side = "opponent"
                        err = max(abs(xy[i, 0] - (100 - xy[j, 0])), abs(xy[i, 1] - (100 - xy[j, 1])))
                    elif not is_context(o):
                        side = "own"
                        err = max(abs(xy[i, 0] - xy[j, 0]), abs(xy[i, 1] - xy[j, 1]))
                    else:
                        continue
                    for w in windows:
                        if d_t <= w[0] and err <= w[1]:
                            counts[(side, j < i, w)][(kind, label(o))] += 1
                            if j < i:
                                found[(side, w)] = True
                                if side == "own" and w == windows[-1]:
                                    same_player[kind]["same" if o["playerId"] == e["playerId"] else "other"] += 1
                for (side, w), v in found.items():
                    if v:
                        hit[(side, w)][kind] += 1
                        if side == "opponent" and w == windows[-1]:
                            lg_hit[kind] += 1
        per_league[league] = {"targets": lg_n, "with_opponent_record_before": sum(lg_hit.values()),
                              "take_ons_with_opponent_record_before": lg_hit["take_on"]}
        print(league, json.dumps(per_league[league]), flush=True)

    out = {"windows": windows, "targets": dict(n_targets), "per_league": per_league,
           "targets_with_record_before": {f"{s}|{w[0]}s|{w[1]}": dict(c) for (s, w), c in hit.items()},
           "records": {f"{s}|{'before' if b else 'after'}|{w[0]}s|{w[1]}": {f"{k}|{r}": n for (k, r), n in c.items()}
                       for (s, b, w), c in counts.items()},
           "own_same_player": {k: dict(v) for k, v in same_player.items()}}
    with open(ROOT / cfg["out_json"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    kinds = sorted(n_targets, key=lambda k: -n_targets[k])
    wl = [f"{w[0]} 秒 / {w[1]}" for w in windows]
    L = ["# 与目标动作重合的记录：审计", "",
         f"范围：{', '.join(cfg['manifests'])} 的非测试比赛。目标动作 = 运动战持球动作（`src/build_actions.py`）。",
         "对方记录：任何类型，起点与目标起点镜像；本方记录：非持球记录（对抗、犯规、越位），起点与目标起点相同。"
         "\"之前\" = 排序在目标之前，会进入目标的输入。", "",
         "## 1. 目标动作中，之前有重合记录的比例", "",
         "| 目标类型 | 目标数 | " + " | ".join(f"对方 {x}" for x in wl) + " | " + " | ".join(f"本方 {x}" for x in wl) + " |",
         "|---|---|" + "---|" * (2 * len(windows))]
    for k in kinds + ["全部"]:
        n = sum(n_targets.values()) if k == "全部" else n_targets[k]
        cells = []
        for s in ("opponent", "own"):
            for w in windows:
                h = sum(hit[(s, w)].values()) if k == "全部" else hit[(s, w)][k]
                cells.append(f"{h / n:.1%}")
        L.append(f"| {k} | {n:,} | " + " | ".join(cells) + " |")
    for s, title in (("opponent", "对方记录"), ("own", "本方非持球记录")):
        w = windows[-1]
        c = counts[(s, True, w)]
        rec = Counter()
        for (k, r), n in c.items():
            rec[r] += n
        top = [r for r, _ in rec.most_common(12)]
        L += ["", f"## {2 if s == 'opponent' else 3}. 排在目标之前的{title}（{w[0]} 秒 / {w[1]}）：目标类型 × 记录类型", "",
              "| 目标类型 | " + " | ".join(top) + " | 其他 |", "|---|" + "---|" * (len(top) + 1)]
        for k in kinds:
            row = [c[(k, r)] for r in top]
            other = sum(n for (kk, r), n in c.items() if kk == k and r not in top)
            L.append(f"| {k} | " + " | ".join(f"{v:,}" for v in row) + f" | {other:,} |")
        tight = counts[(s, True, windows[0])]
        L += ["", "各容差下的记录总数（之前 / 之后）：" + "；".join(
            f"{x}：{sum(counts[(s, True, ww)].values()):,} / {sum(counts[(s, False, ww)].values()):,}" for x, ww in zip(wl, windows)),
              f"（最严容差 {wl[0]} 下排在之前的：{sum(tight.values()):,}）"]
    L += ["", "本方非持球记录（最宽容差）中，与执行者是同一名球员的比例：" + "；".join(
        f"{k} {v['same'] / (v['same'] + v['other']):.0%}（{v['same'] + v['other']:,}）" for k, v in
        ((k, same_player[k]) for k in kinds) if v["same"] + v["other"] > 0)]
    (ROOT / cfg["out_md"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
