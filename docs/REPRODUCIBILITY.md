# 发布内容和复现边界

本仓库从 `scene_regularity_release` 当前代码整理；历史探索作为归档保留。模型结构和训练逻辑未改；发布版将三项上游一致性测试的外部源码路径改为仓库内显式的 `external/baselines/`。工作区原项目及 `10.20` 仓库未被覆盖。

## 数据与依赖

原始数据不随仓库发布。自行准备：

```text
data/raw/entities/players.json
data/raw/entities/teams.json
data/raw/matches/matches_England.json
data/raw/events/events_England.json
# France、Germany、Italy、Spain 同样命名
```

统计依赖已固定于 requirements.txt；模型依赖范围与实际验证版本见首页。GPU 训练、Linux fork 多进程的资源配置由调用者显式选择。

三项上游结构一致性测试需要以下固定源码。网络训练不依赖这些源码；测试只提取类/函数，不执行上游数据读取和训练：

```bash
mkdir -p external/baselines
git clone https://github.com/statsonthecloud/Soccer-SEQ2Event.git external/baselines/Soccer-Seq2Event
git -C external/baselines/Soccer-Seq2Event checkout 22a551646c6fe8cab428b444e3b7b851c8441b9a
git clone https://github.com/calvinyeungck/Football-Match-Event-Forecast.git external/baselines/NMSTPP
git -C external/baselines/NMSTPP checkout 053ecd879536c4eaf9aa0a735f90b4c69d70b224
git clone https://github.com/nvsclub/LargeEventsModel.git external/baselines/Unified-LEM
git -C external/baselines/Unified-LEM checkout 7c09ef8d10afd3883fefb616f0e09981e47fa614
```

## 训练入口

```bash
# 统计分析：完整拟合与置换，耗时较长
bash scripts/run_existence.sh
bash scripts/run_learnability.sh
# 模型数组：先检查该脚本所需资源和原始数据
bash scripts/build_data.sh England_nontest France_nontest Germany_nontest Italy_nontest Spain_nontest
bash scripts/build_model_data.sh England_nontest France_nontest Germany_nontest Italy_nontest Spain_nontest
# 执行者各阶段，先查看 scripts/run_model_dev.sh 的参数
bash scripts/run_model_dev.sh population
# 类型示例；需要上述英超模型数组
python src/train_type.py --config configs/model/type_t0_england_only.json --seed 20260715 --device cuda:0 --out results/model/type_england_only/t0
python src/evaluate_type.py --config configs/model/eval_type_england_only.json
```

类型评估配置要求对应全部模型的 predictions.npz 已生成；只训练 T0 不足以执行整表评估。执行者流程使用交叉拟合偏移量，按 population→m1→entities→b0→entity 的依赖顺序运行；完整 GPU 参数说明在脚本头部。外部基线完整入口见 ACTOR_BASELINES.md。

## 结果复核

发布的评估 JSON/Markdown 可直接阅读。英超统计分析附有 per_match.csv，可以直接运行 `python scripts/verify_published_results.py --report-dir reports/England` 复核 52 项收益和区间。模型逐样本预测、权重、模型输入数组没有发布，因此仅凭这些汇总 JSON 无法重新计算模型指标，必须先重建数据并训练。

所有 current 模型评估属于开发期。五联赛切分未统一日历截止；最终测试、专门冷启动、外部基线正式比较和英超到未见联赛的迁移尚未完成。M4-no-ID 的 c=3 运行失败，选择表仅含其另外两个完成候选。部分实体模型因目标函数不再变化停止，未达到梯度阈值。

归档 Markdown 内涉及旧工作区的命令/路径是历史记录；不承诺这些旧探索脚本在此包中可以执行。复现当前实现应使用本页和 src/、configs/，不要使用归档计划作为当前状态。
