"""Evaluation of the action-type models of src/train_type.py on the validation period (docs/MODEL_PLAN.md, note 22).

cfg["models"] maps a name to {"predictions": <predictions.npz>, "array": "type_log_probs" or
"type_log_probs_true_actor"} (the latter is the diagnostic ceiling of the conditional head). All models must cover the
same validation rows (reachable targets). Per model: type NLL, top-1 and top-3 accuracy, overall and on the first 15
minutes and steps 0-2, NLL per true action class (description), and for models with a who-head the who NLL and top-1.
Comparisons [first, second, label]: per-match log-likelihood differences, bootstrap of matches within league.
Writes cfg["out_json"] and cfg["out_md"].
"""
import argparse

import numpy as np
import torch

from common import ROOT, load
from event_schema import ACTIONS
from model_data import load_meta, scope_dirs, write_json
from seq_model import actor_loss
from train_type import type_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(ROOT / args.config)
    Y, meta = load_meta(cfg["scopes"])
    actor = Y["actor_slot"]
    label = type_labels(cfg["scopes"], actor)
    steps = np.concatenate([np.rint(np.load(d / "arrays_in.npz")["ctx_num"][:, 2] * 4) for d in scope_dirs(cfg["scopes"])])
    rows = None
    res, ll_rows = {}, {}
    for name, spec in cfg["models"].items():
        P = np.load(ROOT / spec["predictions"])
        if rows is None:
            rows = P["rows"]
        elif not np.array_equal(rows, P["rows"]):
            raise ValueError(f"{name}: different validation rows")
        lp = P[spec["array"]].astype(np.float64)
        y = label[rows]
        ll = lp[np.arange(len(rows)), y]
        rank = (lp > ll[:, None]).sum(1)
        subsets = {"all": np.ones(len(rows), bool), "first_15_minutes": meta["minute"].to_numpy()[rows] < 15,
                   "steps_0_2": steps[rows] <= 2}
        r = {s: {"targets": int(m.sum()), "nll": float(-ll[m].mean()), "top1": float((rank[m] == 0).mean()),
                 "top3": float((rank[m] < 3).mean())} for s, m in subsets.items()}
        r["by_action"] = {a: float(-ll[y == i].mean()) for i, a in enumerate(ACTIONS) if (y == i).any()}
        if spec["array"] == "type_log_probs" and "who" in P.files:
            w = torch.as_tensor(P["who"])
            a = torch.as_tensor(actor[rows])
            r["who_nll"] = float(actor_loss(w, a).mean())
            r["who_top1"] = float((w.argmax(1) == a).double().mean())
        res[name] = r
        ll_rows[name] = ll
    match = meta["match_id"].to_numpy()[rows]
    league = meta["league"].to_numpy()[rows]
    matches = np.unique(match)
    pos = np.searchsorted(matches, match)
    lg_of = np.array([league[pos == i][0] for i in range(len(matches))])
    rng = np.random.default_rng(cfg["seed"])
    picks = [(np.flatnonzero(lg_of == lg), None) for lg in np.unique(lg_of)]
    picks = [(idx, rng.integers(0, len(idx), (cfg["bootstrap"], len(idx)))) for idx, _ in picks]
    comps = []
    for a, b, lab in cfg["comparisons"]:
        d = np.bincount(pos, ll_rows[a] - ll_rows[b], len(matches)); n = np.bincount(pos, None, len(matches))
        s_tot, n_tot = 0.0, 0.0
        for idx, pick in picks:
            s_tot = s_tot + d[idx][pick].sum(1); n_tot = n_tot + n[idx][pick].sum(1)
        comps.append({"first": a, "second": b, "label": lab, "estimate": float(d.sum() / n.sum()),
                      "ci": np.quantile(s_tot / n_tot, [0.025, 0.975]).tolist(),
                      "top1_difference": res[a]["all"]["top1"] - res[b]["all"]["top1"]})
    write_json(ROOT / cfg["out_json"], {"models": res, "comparisons": comps, "validation_targets": int(len(rows))})
    lines = [f"# {cfg['title']}", "", f"验证期可达目标 {len(rows):,} 个、{len(matches)} 场。NLL 单位 nats/动作；比较为前者减后者的对数似然增益"
             "（正 = 前者更好），95% 区间来自联赛内按比赛自助。", "", "## 模型", "",
             "| 模型 | 类型 NLL | top-1 | top-3 | 开场 15 分钟 NLL | 步数 0–2 NLL | 执行者 NLL | 执行者 top-1 |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        w = f"{r['who_nll']:.4f} | {r['who_top1']:.4f}" if "who_nll" in r else "— | —"
        lines.append(f"| {name} | {r['all']['nll']:.4f} | {r['all']['top1']:.4f} | {r['all']['top3']:.4f} | "
                     f"{r['first_15_minutes']['nll']:.4f} | {r['steps_0_2']['nll']:.4f} | {w} |")
    lines += ["", "## 比较", "", "| 比较 | NLL 增益 | top-1 差 |", "|---|---|---|"]
    for c in comps:
        lines.append(f"| {c['label']}：{c['first']} − {c['second']} | {c['estimate']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] | "
                     f"{100 * c['top1_difference']:+.2f} 个百分点 |")
    present = [a for a in ACTIONS if all(a in r["by_action"] for r in res.values())]
    counts = {a: int((label[rows] == ACTIONS.index(a)).sum()) for a in present}
    lines += ["", "## 按真实动作类别的 NLL（只作描述）", "", "| 模型 | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for name, r in res.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['by_action'][a]:.3f}" for a in present) + " |")
    lines.append("| 目标数 | " + " | ".join(f"{counts[a]:,}" for a in present) + " |")
    (ROOT / cfg["out_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
