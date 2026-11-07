# 原生输入的球员预测基线

本实现将本地 Seq2Event、NMSTPP 和 Unified-LEM 适配为条件执行者预测。**保留各方法自己的比赛内输入**，不将我们的跨比赛历史、位置先验、场景摘要或候选球员动作池提供给它们。按当前任务约定，三者都额外获得目标动作的时间和起点。这是任务适配，不是原论文成绩的复现。当前软件测试和真实数据冒烟范围见 [验证记录](../reports/actor_baselines_validation.md)。

## 预测问题与输入边界

给定目标动作所属球队、半场、半场内秒数、起点，以及该目标之前可观察的记录，从同一套因果在场候选中预测执行者。球队用于候选集合及 Seq2Event 的本队序列筛选；输出头接收 `[period, period_seconds/3600, start_x/100, start_y/100]`。这些目标条件不送入原生序列编码器。

目标事件本身的动作类型、执行者、标签、结果、终点均不进入输入。与目标重合的记录在构建控球标记、时间差、比分和窗口之前排除。历史记录的动作结果可以进入 LEM，因为它们已经被观察。

| 方法 | 原生比赛内输入 | 默认编码器 |
|---|---|---|
| Seq2Event | 本队最近 40 个原生动作 token；`act` 与 `T,x,y,sg,thetag,scrad,deltaT,deltax,deltay,s` | 动作嵌入 7 + 连续映射 10；1 层、1 头 Transformer，FFN 8，ReLU 读出 64 |
| NMSTPP | 双方最近 40 个原生 token；`act,zone,deltaT,zone_s,zone_deltay,zone_deltax,zone_sg,zone_thetag` | 动作嵌入 5 + 区域嵌入 20 + 连续映射 6；1 层、1 头 Transformer，FFN 1024，线性读出 31 |
| Unified-LEM | 最近 3 条记录的 `h,e,x,y,t,a`，最近观察时刻的半场/分/秒与双方进球、红黄牌累计 | 顶层脚本中的 196×3 MLP，ReLU、dropout 0.3 |

LEM 原生当前事件的六个字段全部置为 `-1`；已允许的目标时间和起点通过单独的四维条件进入球员头。事件窗口按原方法顺序，LEM 为最近事件在前，另两者为最早事件在前。

球员输出头为 `Linear([native_representation, query], training_players)`，取候选球员对应的 logit 后做候选内 softmax。球员 ID 只标识输出类别，不作为事件输入特征。没有另加候选角色、站位或历史特征。

## 本地来源与必要适配

来源位于`external/baselines/`（固定版本准备方式见 [复现说明](REPRODUCIBILITY.md)），代码不需要导入或执行原仓库的训练脚本。

| 方法 | 本地原始文件 | Git revision |
|---|---|---|
| Seq2Event | `Soccer-Seq2Event/Seq2Event_Notebook01_DataImport.ipynb`、`Seq2Event_Notebook02_Modelling.ipynb` | `22a551646c6fe8cab428b444e3b7b851c8441b9a` |
| NMSTPP | `NMSTPP/dataset_create.py`、`train_valid_test_split.py`、`NMSTPP.py` | `053ecd879536c4eaf9aa0a735f90b4c69d70b224` |
| Unified-LEM | `Unified-LEM/0000 Preprocess Data to LEM.py`、`0001 Train Tabular LEMs.py` | `7c09ef8d10afd3883fefb616f0e09981e47fa614` |

- Seq2Event 使用本地 Transformer 配置，动作词表为 `@,_,d,g,p,s,x`。NMSTPP 使用 `p,_,d,x,s`；20 区域由原始质心的最近距离确定，与最大模糊隶属度的硬标签等价。
- 事件过滤保留传球、定位球、射门、带球标签的进攻对抗、加速、解围、无标签触球。控球转换标记仅在已观察到另一队取得球权后产生，复制此前保留的动作。原脚本先在中间表处理犯规再删去犯规，本适配不以被删犯规替代标记的坐标。这是明确的预处理差异。
- Seq2Event 使用 Wyscout 的 x/y 定义，避免原 Notebook 中交换坐标名称的行为；连续特征公式及 base-100 位置编码保留。NMSTPP 保留发布代码的区域差分命名及尺度公式，但位置编码改为序列轴，避免同一目标随批次位置改变输出。
- 所有经验分位数和 MinMax 参数只在训练目标的有效历史窗口 token 上拟合（窗口内重复出现的 token 保留权重）。验证、迁移阶段冻结这些参数。Seq2Event 的时间差截断阈值为训练窗口最大原生时钟的 1%；NMSTPP 时间差截断为 60 秒。
- 短窗口左填充并显式遮蔽，空窗口输出零表示；不因窗口不足丢掉开场目标。LEM 使用原生 `-1` 填充。历史不会跨比赛边界。
- 不按未来是否出现乌龙球删除整场比赛。Seq2Event 乌龙球使用 `g`，NMSTPP 不输入单独的进球 token；比分按 Wyscout 事件及当前数据配置中的显式进球例外构造。
- LEM 原始脚本面向 Wyscout v3。本数据为 v2，使用 `lem_type()` 的显式 32 类映射；不从未来补半场/全场结束 token，也不推断 v3 独有的 carry 附加记录。未知事件类别直接失败。比分和牌数只累计已观察事件，并应用数据配置中明确列出的进球/罚下例外，不使用目标事件之后的状态。
- 原动作/区域/时间输出头及其多任务损失替换为球员分类头和候选内交叉熵。因此 NMSTPP 当前适配检验的是其原生编码器用于执行者任务的表现，不能声称已经复现其原多任务训练收益。

编码器是从本地源码结构移植；单元测试隔离加载原始类，核对 Seq2Event、LEM 及修正位置编码轴后的 NMSTPP 编码输出。

## 比较与冷启动

共同控制为：同一比赛划分、目标行、因果候选集合、目标时间/起点条件、训练采样预算、学习率搜索次数、随机种子和评价规则。原生输入和编码器不同，参数量不强行相等。默认仅英格兰训练；跨范围联合训练必须满足所有训练比赛早于所有验证比赛，否则直接失败。

球员类别表仅来自训练目标中的候选球员。未见球员全部使用**训练类别 logit 的均值**，不创建未训练的随机输出类别。多个未见候选的分数相同；原生输入不能提供个体区分能力。跨联赛出现全新阵容时，这个限制尤其明显，应与可学习的历史不足球员分开解读。

评估保存完整候选分数与稳定键 `(match_id, seq, event_id)`：

- 可达目标 NLL；所有目标 Top-1、Top-3、ECE，不可达目标按准确率失败计入，并报告数量。
- 并列分数使用随机打破并列的精确期望命中率，与主实验共用 `ranking_metrics.py`。若有 g 个候选得分更高、m 个候选与执行者同分（含本人），Top-k 命中为 `clip(k-g,0,m)/m`；不需要随机采样，也不受球员 ID 排序影响。
- 历史为 0、1–4、少于 5 场，训练中未见、近期替补等候选分组的二元对数损失与校准。
- 可达真实执行者的历史不足/未见分组 NLL 与 Top-k；与“场上存在冷候选”的目标组分别报告。
- 按比赛配对、按联赛分层 bootstrap。配对前严格检查目标键、候选顺序、标签及分组元数据。空组写 `null`。

历史场数来自已有数据视图的训练池规则，仅用于分组，不送入模型。当前实现的是自然历史不足分组，不是人为移除某球员训练数据的留出实验。训练未见组不能直接支持“跨联赛球员辨识泛化”的结论。

默认 3 个学习率，搜索种子 `20261021`，每方法相同 120 epoch、每场每 epoch 128 个目标、batch 256、相同 Rotate 顺序。每 4 epoch 用验证 NLL 选 checkpoint，随后用选定学习率运行另外两个种子。NLL 与准确率同时报告，不从验证结果中事后改选主指标。这些开发集比较仍需独立测试集验证。

## 运行

从仓库根目录执行。使用已有的 PyTorch 环境；本机验证环境为 原工作区的 Python 3.13.5 / PyTorch 2.6.0.dev20241112+cu121 环境。以下 `python` 应指向该环境；模型依赖另列于 `requirements-model.txt`。

```bash
# 必须先完成 configs/model/England_nontest.json 对应的数据视图及检查。
# 已存在的输出目录会直接报错，不会覆盖。
python src/prepare_actor_baselines.py --config configs/actor_baselines/england.json --dataset 0

# 软件冒烟检查：2 场训练、128 个验证目标、2 epoch；不能用于研究比较。
python src/train_actor_baseline.py --config configs/actor_baselines/england.json \
  --family seq2event --seed 20261021 --lr 0.001 --mode smoke --device cpu \
  --out results/actor_baselines_smoke/seq2event
# 将 family 和输出目录依次换为 nmstpp、lem。

# 仅打印 9 条搜索命令；不会启动正式训练。
python src/plan_actor_baselines.py --config configs/actor_baselines/england.json \
  --stage search --python python --device cuda:0 --runs-root results/actor_baselines

# 9 次搜索完成后，核对数据和预算，选 LR，打印 6 条补充种子命令。
python src/plan_actor_baselines.py --config configs/actor_baselines/england.json \
  --stage confirm --python python --device cuda:0 --runs-root results/actor_baselines \
  --selection results/actor_baselines/selection.json

# 同一 seed 的两个正式运行目录做配对比较，正增益表示 candidate 更好。
python src/compare_actor_baselines.py --reference PATH_TO_REFERENCE_RUN \
  --candidate PATH_TO_CANDIDATE_RUN --draws 5000 --seed 20261021 \
  --out results/actor_baselines/comparison.json

python -m unittest discover -s tests -p test_actor_baselines.py -v
```

`predict_actor_baseline.py --checkpoint ... --config ... --device ... --out ...` 可在显式提供的其他联赛非测试配置上冻结推理。先为目标配置构建原生输入；编码器配置须一致，且被评分比赛必须晚于源训练截止时刻。没有自动联赛发现、验证集重拟合或隐式恢复训练。

## 文件职责与下一事件扩展

- `actor_native_inputs.py`：仅消费已观察记录，构造三套输入。
- `prepare_actor_baselines.py`：原始数据与既有目标/候选对齐；原生特征、评估元数据分开保存。
- `actor_benchmark_data.py`：训练集预处理、模型输入白名单和结果归档。
- `actor_baselines.py`：三个原生编码器及条件球员头。
- `train_actor_baseline.py`、`predict_actor_baseline.py`：训练和冻结推理。
- `actor_benchmark_metrics.py`、`compare_actor_baselines.py`：统一评估与配对比较。
- `plan_actor_baselines.py`：明确列出搜索和补充种子任务。

`encode_history()` 与球员输出头分离，便于后续接动作、时间、位置输出头；当前训练器只接受 `conditional_actor_A`。下一事件预测不能直接复用已知目标时间/起点的任务视图：必须另行定义可观察边界、标签与损失。当前未实现下一事件训练或人为球员留出协议。
