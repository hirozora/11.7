# 场景化足球历史：执行者与动作类型预测

截至 2026-11-07 的代码与实验快照，最新实验产物来自 2026-10-26。英超是主要研究对象，法甲、德甲、意甲、西甲提供补充验证。本仓库保留当前实现和结果，不把开发期成绩当作最终测试或严格跨联赛迁移证据。

## 关键实验现象

| 问题 | 开发结果 | 当前判断 |
|---|---|---|
| 跨比赛历史能否帮助执行者预测？ | 英超 M1⁻→M1：NLL **1.1543→1.1321**；Top-1 **57.47%→58.16%** | 人工历史摘要有收益；开场 15 分钟 NLL 降幅 0.0976 |
| 学出来的历史是否有用？ | 英超 M1⁻→B0-M1⁻：**1.1543→1.1461**；M1→B0-M1：**1.1321→1.1275** | 历史项有附加收益 |
| 层级表示和图是否更好？ | 英超 B0-M1 **1.1275**、M2 **1.1276**；M4 对 M2+3 的增益区间跨零 | 暂无可靠额外图收益 |
| 无球位置近似是否有信息？ | 英超 M1⁻ 加 E0+E2：**1.1543→1.1502** | 小幅收益，不能称恢复真实无球轨迹 |
| 执行者监督能否帮助动作类型？ | 已知起点 T0→T1w：**1.0257→1.0154**；未知起点 **1.1374→1.1263** | NLL 有收益；T2 显式混合未超过 T1w |
| 历史能否帮助动作类型？ | 当前类型实验关闭候选人的跨比赛历史特征 | 尚未验证完整链条 |
| 冷启动与跨联赛迁移是否完成？ | 有自然低历史切片、多联赛统计及合训评估 | 专门留出与严格迁移尚未完成 |

**口径：** 执行者主实验给定目标球队、时间、起点；“未知起点”实验仍已知目标时间。执行者 NLL 仅计可达目标，Top-k 包含不可达目标；类型实验仅评估可达子集。T3 给定真实执行者，只用于诊断。冷启动候选二元损失不是冷启动执行者准确率。

## 阅读入口

- [全部实验及当前结果](reports/EXPERIMENT_SUMMARY.md)：全部实验族、完整模型表、负结果和限制。
- [英超结构比较](results/model/eval/england_structures/england_structures.md)与[五联赛模型比较](results/model/eval/dev_completed.md)。
- [已知起点类型预测](results/model/type_england_only/eval.md)、[未知起点类型预测](results/model/type_nostart_england_only/eval.md)。
- [英超无球探针](results/model/offball_probe/eval_m1_no_history_england_only.md)、[五联赛无球探针](results/model/offball_probe/eval_m1.md)。
- [统计存在性](reports/existence.md)、[五联赛可学习性](reports/learnability.md)、[英超可学习性](reports/England/report.md)。
- [外部基线协议](docs/ACTOR_BASELINES.md)、[当前发布与运行说明](docs/REPRODUCIBILITY.md)。

`src/` 是当前代码，`configs/` 是显式配置及比赛清单，`tests/` 是检查，`results/` 只发布小型评估/运行元数据。旧探索及计划放在 `archive/`，不作为当前状态入口。原始数据、权重、逐样本模型预测和大型生成表未上传。CSV 盘点索引记录的是原工作区的完整清单，部分条目不在本发布包内。

## 环境与运行

统计依赖见 `requirements.txt`，模型额外依赖见 `requirements-model.txt`。实际验证环境为 Python 3.13.5、PyTorch 2.6.0.dev20241112+cu121、torch-geometric 2.8.0；此前统计发布另在 Python 3.13.5 验证。模型依赖文件提供版本范围，不声称已逐版本验证。

```bash
python -m pip install -r requirements-model.txt
# 14 项 actor 测试中有 3 项需要本地上游源码，按复现说明先准备 external/baselines。
python -m unittest discover -s tests -p 'test_*.py'
python tests/check_quick_attack.py
python scripts/verify_published_results.py --report-dir reports/England
```

训练需要自行准备 Wyscout 2017/18 原始数据到 `data/raw/{entities,matches,events}/`。所有命令从仓库根目录运行，数据目录结构及正式训练入口见 [复现说明](docs/REPRODUCIBILITY.md)。仓库没有自动下载数据或自动启动长训练。
