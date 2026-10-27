# 第一阶段 command 跟踪评测（2026-10-21）

本页是加入正常对打物理筛选前的历史基线。当前默认采样及最新712次结果见
[RALLY_VALID_RESULT.md](RALLY_VALID_RESULT.md)，不再把任意训练扰动组合都当作正常回球任务。

默认入口 `python -m deploy.benchmarks.racket_command.run` 现使用
`default.yaml` 中的 `evaluation_mode: policy_tracking`。没有物理来球构造，
不调用球状态估计或A3球轨迹/回球planner，不以碰球或回球评分。

该actor需要参考motion，所以本轮保留现有motion选择器，不改变选择规则或给入额外参考动作。
结果是“现有选择器 + actor”的command跟踪表现，不能单独归因于神经网络权重。
第二阶段“独立随机发球 → 球估计 → planner → command → 执行”本轮没有实施。
历史按command构造来球的诊断模式不属于第二阶段。

## 测试设置

- X=-0.25m；Y=[-1.51225,0.00200]m，Z=[-0.16225,0.71278]m，桌面Z=0。
- 从原8×6划分裁掉最低Z行，8×5中39格有效，每格随机选择一个固定点。
- 每点5组法向/速度方向 × 2档速度（1.5、4.5m/s）× 2档TTS（0.28、0.50s）。
- 共780条command，seed=42。序列提前冻结，不因拒绝或执行结果重采样。
- 每条恢复同一准备状态，通过原actor和PD执行，不加外力、动作修正或时间补偿。
- 在指定TTS时刻对实际球拍状态插值评分；不使用稍早/稍晚最接近目标的状态代替。
- 联合达标：接受且位置≤5cm、法向≤15°、速度向量误差≤0.5m/s，三项同时满足。
  完整成功还要求回位且没有执行失败。法向按圆盘正反面等价计算。

## 完整结果

输出：`deploy/recordings/racket_command/policy_tracking_full_20261021/`。

|指标|数量|比例|
|---|---:|---:|
|完成评测|780/780|100%|
|选择器接受|780/780|100%|
|指定时刻三项联合达标|76/780|9.74%|
|完整成功（含回位）|76/780|9.74%|
|位置单项达标|268/780|34.36%|
|法向单项达标|742/780|95.13%|
|速度向量单项达标|135/780|17.31%|
|完成回位|780/780|100%|

没有执行失败或失去站立。位置误差中位数6.40cm，90分位18.01cm；
法向误差中位数4.18°，90分位11.17°；速度向量误差中位数0.99m/s，90分位2.35m/s。

|请求球拍速度|TTS|联合达标|
|---|---|---:|
|1.5m/s|0.28s|17/195（8.72%）|
|1.5m/s|0.50s|56/195（28.72%）|
|4.5m/s|0.28s|0/195|
|4.5m/s|0.50s|3/195（1.54%）|

所选motion在导出联合训练支持域内344条、域外436条。该标记仅作诊断，不反馈选择器。
域内36/344达标、域外40/436达标；这是不同command的分组，不能作为更改选择器的因果对照。
当前误差在位置和速度上较明显，短TTS/高速样本表现更差；本轮没有球planner参与，
这些失败不能解释为planner算错TTS。也不能仅凭此确定actor权重、初始状态或sim差异各自的影响。

每格仅一个空间点、每点五组方向、两档速度和两档TTS；9.74%仅描述本批780条目标，
不代表任意连续来球或整个训练区域的成功率。

## 输出与复现

- `report.html`：离线交互热力图，默认查看指定时刻三项联合达标率，支持点位、方向、速度、TTS筛选。
- `capability.png`：接受率、联合达标率、位置/法向/速度误差、回位率。
- `points/point_*.png`：每点按方向分列，四行展示联合达标和三个误差的速度×TTS矩阵。
- `tracking_analysis.json`：TTS、速度及所选motion支持域分组统计。
- `suite.json`、`results.jsonl`、`traces/`：冻结目标、逐条结果与实际球拍轨迹。
- `adapter_metadata.json`：`planner_used=false`、`ball_dynamics=disabled`、`display_ball=false`。
  `ball_summary.enabled=false`，没有球不算碰球失败；不生成物理球热力图。

```bash
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --render \
  --output "deploy/recordings/racket_command/policy_$(date +%Y%m%d_%H%M%S)"
```

原样回放本批command：

```bash
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --suite deploy/recordings/racket_command/policy_tracking_full_20261021/suite.json \
  --render \
  --output "deploy/recordings/racket_command/policy_replay_$(date +%Y%m%d_%H%M%S)"
```

每个点连续执行20次才换点。可用`--point Y Z`检查单点，或`--limit N`做前N条的部分评测。
默认无需`--command-only`；历史suite可用该参数明确切换为无球评测。

验证：相关测试35 passed；新测试禁止球构造并检查无碰球失败统计，原有直接command测试
禁止调用planner；native MuJoCo viewer完成一条无球测试；交互HTML经Chrome验证，
第一阶段不显示物理来球/碰拍/回球指标。未修改或连接远端机器。
