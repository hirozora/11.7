"""Development evaluation on the validation period (docs/MODEL_PLAN.md, section 5).

For each model (a scores file with scores (N, C), -inf padding, for every target of the scopes; cfg["models"] maps a
name to a path, or to {"variant": v} for the scores of entity variant v chosen in cfg["selection"], the file written
by src/select_entity.py):
  action level   NLL of the actor (reachable targets); top-1 and top-3 and the ECE of the top-1 confidence (15
                 equal-width bins, all targets) with ties broken at random, in expectation: an actor with g candidates
                 scored above him and tied with m (himself included) is in the top k with probability
                 clip(k - g, 0, m) / m; unreachable targets count as misses; pooled and per league
  subsets        known at the target's time: first 15 minutes; within 10 minutes after an own-team substitute became a
                 candidate; steps 0-2; at least one own candidate with fewer than 5 prior matches; six time segments
                 (0-15, 15-30, 30-45+ of the first half, 45-60, 60-75, 75-90+ of the second)
  candidate      binary log loss of every candidate's probability on reachable targets, summed over two known-at
  level          candidate groups: fewer than 5 prior matches; substitutes that became candidates less than 10 minutes ago
  descriptive    NLL by a group defined by the true actor (own actions since the actor's last action in the match);
                 for description only, never compared
Comparisons ([first, second] or [first, second, label] in the config, first minus second): per-match sums of log-likelihood differences (loss differences
for the candidate level, sign turned so that positive = first better), bootstrap of matches within league with the same
draws for every comparison (cfg["bootstrap"] draws), reported per action (or per candidate row) with 95% intervals,
pooled and per league. Writes cfg["out_json"] and cfg["out_md"].
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import ROOT, load
from model_data import load_meta, scope_dirs, write_json
from ranking_metrics import expected_topk

SEGMENTS = ["0–15", "15–30", "30–45+", "45–60", "60–75", "75–90+"]
ACTOR_GROUPS = ["no earlier action", "1", "2–3", "4–10", "11+"]
SUBSET_COLUMNS = [("all", "全部"), ("first_15_minutes", "开场 15 分钟"), ("after_substitution_10_minutes", "换人后 10 分钟"),
                  ("steps_0_2", "步数 0–2"), ("cold_candidate_on_pitch", "场上有历史少于 5 场的候选人"),
                  ("cold_candidates", "冷启动候选人（二元）"), ("recent_substitutes", "刚上场替补（二元）")]


def log_softmax(scores):
    return torch.log_softmax(torch.as_tensor(scores), -1).numpy()


def candidate_attributes(scope_configs):
    """prior matches, starter flag, minutes since becoming a candidate, actions so far and own actions since the last
    action per (target, slot), from the input arrays (src/model_inputs.py)."""
    cols = {k: [] for k in ("prior_matches", "starter", "minutes_entered", "cum", "since", "mask")}
    for d in scope_dirs(scope_configs):
        A = np.load(d / "arrays_in.npz")
        num = A["cand_num"]
        cols["prior_matches"].append(np.rint(num[..., 6] * 38)); cols["starter"].append(num[..., 1] > 0.5)
        cols["minutes_entered"].append(num[..., 5] * 90); cols["cum"].append(np.rint(np.expm1(num[..., 2] * 4)))
        cols["since"].append(np.rint(num[..., 3] * 200)); cols["mask"].append(A["cand_mask"])
    return {k: np.concatenate(v) for k, v in cols.items()}


def steps_of(scope_configs):
    return np.concatenate([np.rint(np.load(d / "arrays_in.npz")["ctx_num"][:, 2] * 4) for d in scope_dirs(scope_configs)])


def time_segment(meta):
    period, minute = meta["period"].to_numpy(), meta["minute"].to_numpy()
    if not np.isin(period, [0, 1]).all():
        raise ValueError("periods other than the two halves")
    return period * 3 + np.clip(((minute - 45 * period) // 15).astype(int), 0, 2)


def actor_group(attr, actor, reach):
    """Group of the true actor: no earlier action in the match, or own actions since his last action 1, 2-3, 4-10, 11+."""
    a = np.maximum(actor, 0)
    rows = np.arange(len(actor))
    cum, since = attr["cum"][rows, a], attr["since"][rows, a]
    g = np.select([cum == 0, since <= 1, since <= 3, since <= 10], [0, 1, 2, 3], 4)
    return np.where(reach, g, -1)


def per_match(values, match, matches):
    index = {m: i for i, m in enumerate(matches)}
    pos = np.array([index[m] for m in match])
    return np.bincount(pos, values, len(matches))


def bootstrap(diff_sum, count, picks):
    """sum(diff) / sum(count) with 95% intervals from the within-league match draws, pooled and per league."""
    total_s, total_n = 0.0, 0.0
    per_league = {}
    for lg, (idx, pick) in picks.items():
        s, n = diff_sum[idx][pick].sum(1), count[idx][pick].sum(1)
        total_s, total_n = total_s + s, total_n + n
        per_league[int(lg)] = {"estimate": float(diff_sum[idx].sum() / count[idx].sum()),
                               "ci": np.quantile(s / n, [0.025, 0.975]).tolist()}
    return {"estimate": float(diff_sum.sum() / count.sum()), "ci": np.quantile(total_s / total_n, [0.025, 0.975]).tolist(),
            "leagues": per_league}


def name_of(row):
    return (f"{row['label']} " if row["label"] else "") + f"{row['first']} − {row['second']}"


def fmt(r):
    return f"{r['estimate']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    rng = np.random.default_rng(cfg["seed"])
    Y, meta = load_meta(cfg["scopes"])
    actor = Y["actor_slot"]
    val = meta["block"].to_numpy() == 1
    reach = actor >= 0
    league = meta["league"].to_numpy()
    attr = candidate_attributes(cfg["scopes"])
    mask = attr["mask"]
    steps = steps_of(cfg["scopes"])
    segment = time_segment(meta)
    group = actor_group(attr, actor, reach)
    sub_recent = mask & ~attr["starter"] & (attr["minutes_entered"] < 10)
    cold = mask & (attr["prior_matches"] < 5)
    subsets = {"all": np.ones(len(meta), bool), "first_15_minutes": (meta["minute"].to_numpy() < 15),
               "after_substitution_10_minutes": sub_recent.any(1), "steps_0_2": steps <= 2,
               "cold_candidate_on_pitch": cold.any(1)}
    subsets.update({f"time_{s}": segment == i for i, s in enumerate(SEGMENTS)})
    matches = np.unique(meta["match_id"].to_numpy()[val])
    league_of_match = meta.drop_duplicates("match_id").set_index("match_id").loc[matches, "league"].to_numpy()
    picks = {}
    for lg in np.unique(league_of_match):
        idx = np.flatnonzero(league_of_match == lg)
        picks[lg] = (idx, rng.integers(0, len(idx), (cfg["bootstrap"], len(idx))))
    match_t = meta["match_id"].to_numpy()
    leagues = sorted(np.unique(league[val]).tolist())

    selection = load(ROOT / cfg["selection"]) if "selection" in cfg else {}
    paths = {}
    for name, spec in cfg["models"].items():
        if isinstance(spec, dict):
            if spec["variant"] not in selection:
                raise ValueError(f"{name}: variant {spec['variant']} is not in the selection file")
            paths[name] = selection[spec["variant"]]["scores"]
        else:
            paths[name] = spec
    results, sums = {}, {}
    for name, path in paths.items():
        S = np.load(ROOT / path)["scores"]
        if S.shape != mask.shape or not np.array_equal(np.isfinite(S), mask):
            raise ValueError(f"{name}: scores do not match the candidate masks")
        lp = log_softmax(S)
        p = np.exp(lp)
        ll = np.where(reach, lp[np.arange(len(lp)), np.maximum(actor, 0)], 0.0)
        hit = expected_topk(S, actor)
        conf = p.max(1)
        r = {}
        for sname, sel in subsets.items():
            s = sel & val
            sr = s & reach
            r[sname] = {"targets": int(s.sum()), "reachable": int(sr.sum()), "nll": float(-ll[sr].mean()),
                        "top1": float(hit[1][s].mean()), "top3": float(hit[3][s].mean())}
        bins = np.minimum((conf[val] * 15).astype(int), 14)
        r["all"]["ece"] = float(sum(abs(hit[1][val][bins == b].mean() - conf[val][bins == b].mean()) * (bins == b).mean()
                                    for b in range(15) if (bins == b).any()))
        r["leagues"] = {int(lg): {"nll": float(-ll[val & reach & (league == lg)].mean()),
                                  "top1": float(hit[1][val & (league == lg)].mean())} for lg in leagues}
        r["actor_groups"] = {g: {"targets": int((val & (group == i)).sum()), "nll": float(-ll[val & (group == i)].mean())}
                             for i, g in enumerate(ACTOR_GROUPS)}
        is_actor = np.arange(S.shape[1])[None] == actor[:, None]
        lp64 = lp.astype(np.float64)
        with np.errstate(divide="ignore"):  # padding has log-probability -inf; it is never selected below
            bce = -np.where(is_actor, lp64, np.log1p(-np.exp(lp64)))
        cand = {}
        for gname, gmask in (("cold_candidates", cold), ("recent_substitutes", sub_recent)):
            sel = gmask & (val & reach)[:, None]
            cand[gname] = {"rows": int(sel.sum()), "binary_log_loss": float(bce[sel].mean())}
            sums[(name, gname)] = (per_match(np.where(sel, bce, 0).sum(1)[val], match_t[val], matches),
                                   per_match(sel.sum(1)[val].astype(float), match_t[val], matches))
        r["candidate_level"] = cand
        results[name] = r
        for sname, sel in subsets.items():
            s = sel & val & reach
            sums[(name, sname)] = (per_match(np.where(s, ll, 0)[val], match_t[val], matches),
                                   per_match(s[val].astype(float), match_t[val], matches))
        print(name, {k: round(v, 4) for k, v in r["all"].items() if isinstance(v, float)}, flush=True)

    comparisons = []
    for entry in cfg["comparisons"]:
        a, b = entry[:2]
        row = {"first": a, "second": b, "label": entry[2] if len(entry) > 2 else ""}
        for sname in list(subsets) + ["cold_candidates", "recent_substitutes"]:
            (sa, na), (sb, nb) = sums[(a, sname)], sums[(b, sname)]
            if not np.array_equal(na, nb):
                raise ValueError(f"{a} and {b} are not evaluated on the same rows of {sname}")
            sign = -1.0 if sname in ("cold_candidates", "recent_substitutes") else 1.0  # loss vs log-likelihood
            row[sname] = bootstrap(sign * (sa - sb), na, picks)
        comparisons.append(row)
    names = [Path(s).stem.split("_")[0] for s in cfg["scopes"]]  # configs/model/<League>_nontest.json
    write_json(ROOT / cfg["out_json"], {"models": results, "comparisons": comparisons, "leagues": names,
                                        "validation_matches": int(len(matches)), "bootstrap": cfg["bootstrap"]})

    lines = [f"# {cfg['title']}", "", f"验证期 {len(matches)} 场；联赛：{'、'.join(names)}。单位：每动作 nats（候选人层面为每候选行）。"
             "比较为前者减后者的对数似然增益（正 = 前者更好），95% 区间来自联赛内按比赛自助"
             f"（{cfg['bootstrap']} 次，所有比较共用同一组抽样）。", "",
             "## 模型", "", "| 模型 | NLL | top-1 | top-3 | ECE | 冷启动候选人（二元） | 刚上场替补（二元） |", "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a, c = r["all"], r["candidate_level"]
        lines.append(f"| {name} | {a['nll']:.4f} | {a['top1']:.4f} | {a['top3']:.4f} | {a['ece']:.4f} | "
                     f"{c['cold_candidates']['binary_log_loss']:.4f} | {c['recent_substitutes']['binary_log_loss']:.4f} |")
    first = next(iter(results.values()))
    lines += ["", "目标数：" + "；".join(f"{k} {first[k]['targets']:,}（可达 {first[k]['reachable']:,}）" for k, _ in SUBSET_COLUMNS[:5])
              + f"；冷启动候选行 {first['candidate_level']['cold_candidates']['rows']:,}，刚上场替补候选行 "
              f"{first['candidate_level']['recent_substitutes']['rows']:,}。"]
    if len(leagues) > 1:
        lines += ["", "## 各联赛 NLL", "", "| 模型 | " + " | ".join(names[lg] for lg in leagues) + " |",
                  "|---|" + "---|" * len(leagues)]
        for name, r in results.items():
            lines.append(f"| {name} | " + " | ".join(f"{r['leagues'][lg]['nll']:.4f}" for lg in leagues) + " |")
    lines += ["", "## 比较", "", "| 比较 | " + " | ".join(label for _, label in SUBSET_COLUMNS) + " |",
              "|---|" + "---|" * len(SUBSET_COLUMNS)]
    for row in comparisons:
        lines.append(f"| {name_of(row)} | " + " | ".join(fmt(row[s]) for s, _ in SUBSET_COLUMNS) + " |")
    if len(leagues) > 1:
        lines += ["", "## 比较：各联赛（全部动作）", "", "| 比较 | " + " | ".join(names[lg] for lg in leagues) + " |",
                  "|---|" + "---|" * len(leagues)]
        for row in comparisons:
            lines.append(f"| {name_of(row)} | " + " | ".join(fmt(row["all"]["leagues"][lg]) for lg in leagues) + " |")
    lines += ["", "## 比较：比赛时间分段", "", "| 比较 | " + " | ".join(SEGMENTS) + " |", "|---|" + "---|" * len(SEGMENTS)]
    for row in comparisons:
        lines.append(f"| {name_of(row)} | " + " | ".join(fmt(row[f'time_{s}']) for s in SEGMENTS) + " |")
    lines += ["", "## 按真实执行者分组的 NLL（只作描述）", "",
              "分组：执行者本场此前没有动作，或距他上次动作的本方动作数。", "",
              "| 模型 | " + " | ".join(ACTOR_GROUPS) + " |", "|---|" + "---|" * len(ACTOR_GROUPS)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['actor_groups'][g]['nll']:.4f}" for g in ACTOR_GROUPS) + " |")
    lines.append("| 目标数 | " + " | ".join(f"{first['actor_groups'][g]['targets']:,}" for g in ACTOR_GROUPS) + " |")
    (ROOT / cfg["out_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
