"""Pool exploratory gains and gate inclusion on current existence and learnability results."""
import argparse
import json

import numpy as np

from pooling import forward_pool
from common import ROOT, load

DECISIONS = {
    "T2a team x steps (long_pass)": ("T2a 球队 × 步数：长传", "显式", "候选，默认关闭"),
    "T2c team x steps (back_pass)": ("T2c 球队 × 步数：回传", "显式", "候选，默认关闭"),
    "who|game_state_form": ("执行者头的比赛状态（确认形式：紧迫度 × 位置组）", "执行者头只加入这一形式", "执行者头不使用比赛状态"),
}


def comparisons(npz):
    return sorted({key.rsplit("|", 1)[0] for key in npz.files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    rng = np.random.default_rng(cfg["seed"])
    leagues = cfg["leagues"]
    out = {"leagues": leagues, "comparisons": {}, "decisions": {}}
    for part in ("population", "ladder"):
        files = [np.load(ROOT / cfg["results_dirs"][lg] / f"{part}.npz") for lg in leagues]
        names = comparisons(files[0])
        if any(comparisons(f) != names for f in files):
            raise ValueError(f"{part}: leagues differ in their comparisons")
        for n in names:
            r = forward_pool([{"sum": f[f"{n}|sum"], "n": f[f"{n}|n"]} for f in files], cfg["bootstrap"], rng)
            out["comparisons"][n] = {"part": part, "gain": r["gain"], "ci": r["ci"], "learnable": bool(r["ci"][0] > 0),
                                     "rows": int(sum(f[f"{n}|n"].sum() for f in files)),
                                     "leagues": {lg: {"gain": x["gain"], "ci": x["ci"]} for lg, x in zip(leagues, r["leagues"])}}
    tau = {lg: load(ROOT / cfg["results_dirs"][lg] / "ladder.json")["interactions"] for lg in leagues}
    for n, c in out["comparisons"].items():
        if n in tau[leagues[0]]:
            c["tau_est"] = {lg: tau[lg][n]["tau_est"] for lg in leagues}
    confirmed = load(ROOT / cfg["existence_results"])["items"]
    hypothesis = {"T2a team x steps (long_pass)": "T2a", "T2c team x steps (back_pass)": "T2c", "who|game_state_form": "R1-Pa"}
    for n, (label, yes, no) in DECISIONS.items():
        c = out["comparisons"][n]
        exists = bool(confirmed[hypothesis[n]]["confirmed"])
        decision = yes if exists and c["learnable"] else ("候选，默认关闭" if exists or c["learnable"] else "不纳入")
        if n == "who|game_state_form" and exists and not c["learnable"]:
            decision = "执行者头不使用比赛状态"
        out["decisions"][n] = {"label": label, "confirmed": exists, "learnable": c["learnable"], "decision": decision}
    (ROOT / cfg["out_json"]).parent.mkdir(parents=True, exist_ok=True)
    with open(ROOT / cfg["out_json"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    def fmt(g, ci):
        return f"{g:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"

    L = ["# 五个联赛合并的可学习性（探索性）", "",
         "计划：`docs/LEARNABILITY.md`。法、德、意、西的非测试比赛已用于确认，本结果只作探索性证据。", "",
         "单位：10⁻³ nats/行。每个联赛内每队前 19 场估计、第 20 场及以后评估；合并区间来自联赛内按比赛的自助法（分层）。"
         "可学习 = 合并区间在 0 以上。", "",
         "## 预先写定的判定", "", "| 项 | 合并收益 | 可学习 | 决定 |", "|---|---|---|---|"]
    for n, d in out["decisions"].items():
        c = out["comparisons"][n]
        L.append(f"| {d['label']} | {fmt(c['gain'], c['ci'])} | {'是' if d['learnable'] else '否'} | {d['decision']} |")
    head = "| 比较 | 合并收益 | 可学习 | " + " | ".join(leagues) + " | 评估行 |"
    for part, title in (("population", "人群部分（去掉该组特征的损失）"), ("ladder", "主效应、交互与类型（叠加在前一层之上的收益）")):
        L += ["", f"## {title}", "", head, "|" + "---|" * (len(leagues) + 4)]
        for n, c in out["comparisons"].items():
            if c["part"] != part:
                continue
            per = " | ".join(f"{c['leagues'][lg]['gain']:+.3f}" for lg in leagues)
            label = n.replace("|", "\\|")
            L.append(f"| {label} | {fmt(c['gain'], c['ci'])} | {'是' if c['learnable'] else '否'} | {per} | {c['rows']:,} |")
    (ROOT / cfg["out_md"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
