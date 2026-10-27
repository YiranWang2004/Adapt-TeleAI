# HOPE G1：train/play 与 MuJoCo 部署核对（2026-10-19）

**最新状态：** 已在本机完成 A3 完整目标链路与部署行为对齐，最终三种子 **30/30 有效回球**。
详见 [A3_ALIGNMENT_RESULT.md](A3_ALIGNMENT_RESULT.md)。本文以下数值是修改前/中间基线记录。

**2026-10-20 补充更正：** 本文的 0/30 是固定参考拍速/法向 baseline，未接入 A3
完整 planner 目标链路，不能单凭它归因于 checkpoint。训练日志已确认 iteration=18000
的 curriculum scale 为 1.0，先前使用 play-start=0.15 的部署门槛已修正。
实际 Isaac 440 样本的观测/actor 对拍及完整差异见 [A3 对标报告](A3_DEPLOY_COMPARISON.md)。

已确认并修复部署中的参考坐标系和目标构造错误。本次验证的是实际球拍碰撞，
没有调整球拍半径、球半径、碰撞判定或 checkpoint。
**后续严格验收结论：碰拍不等于有效回球。按参考击球平面生成且实际过网、
己方一次落台的来球，三个种子共 30 球均碰拍，但有效回球为 0/30。**

## 已修复的问题

1. **绝对向量的坐标系。** `racket_target_vel_w` 和 `racket_target_normal_w`
   在训练中属于 NPZ/Isaac 的 world。部署将整个动作场景旋转到现场 world，
   本次启动时该旋转约 84.5°；原实现把旋转后的向量直接送入 actor。
   现保持物理世界中的目标用于规划/显示，在构造 actor 观测时用参考变换的逆旋转
   映回训练 world。相对位置、机身系速度、重力与关节观测保持各自的训练语义。
   回归测试验证：同一状态整体旋转/平移后，175 维 actor 输入全部保持一致。
2. **动作库的统一参考系。** play 对所有动作添加同一个 env origin。
   原部署在每次选动作时把每个动作首帧分别对齐当前机器人，参考位置随站姿漂移。
   现进入策略时固定一次公共变换，整个动作库共享它，后续选动作不重新居中。
3. **机身目标与击球目标的耦合。** train/play 使用
   `base_target_xy = racket_target_xy - reference_reach_offset_xy + jitter`。
   原部署改了接触位置却保留原机身目标；现两者应用同一个 XY 位移，jitter 取 0。
4. **目标范围。** 当前 play 从 `ref_perturb_curriculum_start=0.15` 开始，
   每轴位置扰动范围为 `0.25 × 0.15 = 0.0375m`。原选择器只检查 0.25m 欧氏距离。
   现保留距离检查，同时按每轴范围筛选；可在 rally 配置中显式设置
   `target_perturb_scale`。训练过程可能达到更大的 curriculum scale，不能把
   play 启动值解释为整个训练过程的最大范围。
5. **播放结束与回位的一帧偏差。** 原部署输出第 93 帧时就切换到回位，
   回位计时提前了一个控制周期（20ms），实际只输出 74 个回位命令。
   现第 93 帧保持完一个周期后，下一 tick 才开始回位，并按命令计数保持 75 帧。
   该偏差发生在第 43 帧击球之后，是次要问题，不能解释此前击球阶段的偏差。

## motion 播放长度与逐帧检查

全部 22 个动作均为 94 帧、50Hz。首末采样点的时间跨度是
`(94-1)/50=1.86s`，而从第 0 帧输出到最后一帧保持完成需要 `94/50=1.88s`。
部署按整数帧计数播放，不用首末采样点跨度作为提前结束条件。
从第 k 帧开始，应依次输出 k…93，共 `94-k` 个周期；
击球倒计时为 `(43-frame)×0.02s`，击球后保持负值，不提前截断动作。

新增测试覆盖起始帧 0、1、29，验证无跳帧/漏帧、第 93 帧仍为 EXECUTE、
下一 tick 才进入 RECOVER，以及回位命令索引连续 0…74。
实际接球运行的逐帧结果保存在审计 JSON 的 `playback_timing` 中。
新时序的 10 次接球均验证：执行帧连续，末帧保持 20ms，回位连续输出 75 帧、
持续 1.5s；击球帧时刻相对规划 deadline 的最大误差为 2.60ms。

## 配置核对

| 项目 | train/play | 当前部署与结论 |
| --- | --- | --- |
| checkpoint | `model_18000.pt` | 本地副本 SHA256 一致：`e22854c523cee8fe8933c198c8885e2216787920d2a89e9da1c6882cc150ffb3` |
| actor | 175→512→256→128→29，ELU | 一致；独立加载 actor，不使用 critic |
| 归一化 | `(x-mean)/(std+0.01)` | 一致；先前与原 RSL-RL 的 990 个输入对照，输出误差 0 |
| 观测顺序/维度 | 保存配置中的 14 项，175 维 | 加载时验证函数、顺序、维度和变换；坐标修正见上文 |
| 参考 command | 29 维位置 + 29 维速度 | 一致，按保存的 Isaac articulation 顺序映射 NPZ |
| 动作/尺度/默认姿态 | 29 维，逐关节 scale + default，clip=null | 一致；动作历史保留原 actor 输出 |
| PD、armature、力矩限制 | 保存的逐关节值 | 按名称读取/校验；执行 PD 表与保存值一致，scale=1 |
| 资产质量/惯量/关节/FK | `g1_29dof_pingpong_fast.urdf` | 与本地来源 URDF 的惯量、质量、关节 origin/axis/limit 对照一致；球拍中心为 wrist+[0.215,0.003,0] |
| 控制频率 | 50Hz | 一致 |
| 物理时间步 | Isaac 0.005s，decimation=4 | 完整部署用 MuJoCo 0.001s，20 子步；受控核对使用 0.005s/4 子步。引擎不相同 |
| 击球时刻 | round(0.46×93)=43 帧 | 一致，TTS=(43-frame)×0.02，击球后保留负值 |
| 目标速度/法向 | 参考法向，正反手速度符号，基准速度 3m/s | 一致；部署用扰动中心值，actor 绝对向量映回训练 world |
| 机身线速度 | Isaac 瞬时速度 | 完整部署从动捕位置差分估计；受控核对用瞬时 root-link 速度，二者不是同一种传感器 |
| 地形/随机化 | 粗糙地形、噪声、摩擦/质量/PD 随机化 | MuJoCo 基线为平地、确定性参数，未逐样本复刻随机化 |
| 自碰撞 | 保存配置开启 | 原共享 MuJoCo 资产关闭自碰撞；本次未改变此项，不能据此声称接触统计与 PhysX 完全一致 |
| 初始化/调度 | motion 首帧 reset，随机起始帧，75 帧回位，随机切下一动作 | 部署从 AMP 接管、固定 home 等球、按来球选动作；调度与 play 不完全相同 |
| 球 | 本 HOPE play 任务追踪期望拍面目标，play marker 不是部署的来球碰撞验收 | 部署额外进行球预测、动作选择和球拍碰撞；必须另测命中率 |

训练/play 的来源文件快照与 SHA256 位于本地
`artifacts/hope_20261012_18000/audit_sources/`；训练保存配置位于同目录 `params/`。
这不是“PhysX 与 MuJoCo 全部相同”的结论；上表明确保留了引擎、传感器和调度差异。

## 定量验证

- 受控名义目标：22/22 动作通过位置 <7.5cm、速度误差 <1m/s、法向误差 <25°
  的组合阈值，平均位置误差 2.14cm。该实验逐动作首帧初始化、关闭目标扰动，
  不经过 AMP 或来球选择，目的在于分离基础跟踪与在线接球错误。
- 同一 seed=42，修正前 0/10 次物理碰撞命中，修正后 10/10。
- 额外 seed=7、123 分别为 10/10；三个种子合计 30/30，均完成回位且未检测到失去站立。
- 隔离坐标转换的对照实验（同 seed=42、相同新时序及其他修正）：仅关闭 actor
  输入的训练 world 逆旋转为 2/10，开启后为 10/10。这支持绝对向量坐标错误是主要原因。
- 修正最后一帧/75 帧回位计数后，额外运行仍为 10/10；10 次挥拍与回位全部完成。
- 命中表示球与拍面接触，不表示每一球都合法越网并落在对方桌面。

机器可读结果：`deploy/recordings/hope_g1/train_play_audit_20261019/report.json`。
原始运行目录分别为 `audit_baseline_10`、`audit_training_world_vectors`、
`audit_fixed_seed7`、`audit_fixed_seed123`，位于 `deploy/recordings/hope_g1/`。
新时序及坐标对照目录为 `audit_timing_fixed` 与 `audit_ablation_world_inverse_off`。
后者未启用命中率门槛，其 `passed=true` 仅表示控制流程完成，实际命中率为 20%。

复现有命中率门槛的接球验收：

```zsh
cd /home/lhy/table-tennis/Adapt-TeleAI
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --render --sim-controls auto \
  --motion-select auto --motion-allow all \
  --episodes 10 --duration 70 --min-hit-rate 0.8
```

`--min-hit-rate` 只用于自动模拟来球；控制台和 summary 同时报告
`flow_passed`、`simulated_hits`、`simulated_balls` 和 `hit_rate`。
完成挥拍/回位但没有命中时，不能通过此门槛。

复现受控名义目标核对：

```zsh
.venv/bin/python -m deploy.hope_g1.audit \
  --receiving-run deploy/recordings/hope_g1/audit_training_world_vectors \
  --output deploy/recordings/hope_g1/nominal_tracking_audit.json
```

本轮不启动真机电机；MuJoCo 命中率不是 `.200` 真机运动或接球验收结果。
本轮回归：`pytest deploy/tests -q` 为 154 passed（18 条已有资产 attach 警告）；
HOPE 源码与对应测试的 Ruff 检查通过。

## 有效回球验真与发球约束（后续测试）

用户要求保留策略真实效果，不通过外部控制补偿。此轮未改变 checkpoint、actor 输入/
输出语义、3m/s 的参考目标速度、参考拍面法向、PD、球拍/球半径、恢复系数或碰撞响应。
没有出球方向覆盖、速度重置、落点控制、按成功结果筛球或筛动作。

新增独立 `hope_g1.feeder.MotionPlaneFeeder`，共享运行时通过 `feeder_factory` 注入，
旧任务默认继续使用原 `RandomBallFeeder`。发球仅使用进入策略时冻结的参考动作：

1. 第 43 帧的拍面中心和 wrist +Y 法向确定击球平面。
2. 在平面内施加原配置的横向/竖向小扰动，法向偏移为零。
3. 入射水平角优先沿拍面法向，从对面朝向机器人；有限球网的范围约束允许斜入射。
4. 垂直运动仍由原重力和球台反弹决定，使用真实模拟球验证来球跨过网顶、
   在己方台面恰好弹一次并到达参考接触点。不要求来球速度完整平行拍面法向。
5. 不读取 actor 运行后的拍面或出球质量生成下一球；发球间隔仍为 4s。
   为入射合法性做的拒绝采样会影响动作分布，故结果不代表 22 个动作均被测试。

有效回球必须按顺序满足：碰拍 → 球体从网顶上方越网 → 第一次向下穿过桌面高度的
球心交点在对方半台内；打网、先在己方侧反弹、侧出界、打长均不通过。
对整个桌面高度平面检测第一次交点，避免忽略台外掉落后再把后续反弹计为成功。
新增测试覆盖成功回球、己方先落台、过长、侧出界、打网、发球平面与过网走廊，
以及流程/碰拍成功但有效回球不足时验收失败。

| seed | 发球 | 碰拍 | 有效回球 | 挥拍/回位 | 80% 回球门槛 |
| --- | --- | --- | --- | --- | --- |
| 42 | 10 | 10 | 0 | 10/10 | 不通过 |
| 7 | 10 | 10 | 0 | 10/10 | 不通过 |
| 123 | 10 | 10 | 0 | 10/10 | 不通过 |

30 球覆盖 16 个发球参考动作，均未检测到失去站立；入射过网最低净余量为 14.57cm，
过网横向最大绝对坐标为 0.7126m（桌宽的一半为 0.7625m）。失败分布为：
17 球的第一次桌面高度交点在对方半台之外，11 球没有对方落台且未记录到该向下交点，
2 球打网；后一组中包括接触点已经低于桌面且继续向下的情况。

**进一步定位：** 保存的训练奖励是拍面位置/速度/法向、机身位置与动作跟踪等，
不是实际来球的越网/对方落台奖励。参考目标是沿动作法向的固定 3m/s 速度，
没有从入射速度和落点求解回球目标。30 球接触时拍面平面相对目标最大夹角仅约 5.76°，
但小角度/速度误差仍可能明显改变落点。一个 seed=7 的动作 9 样本出球速度为
`[7.964,-0.602,1.670]m/s`，首次桌面高度交点 x≈3.646m，超过远端 3.39m。
不能把良好的目标跟踪解释为可靠的实际回球，也不能把有限来球分布的失败推广到所有场景。

另做了**仅用于分析、不输入控制器**的理想跟踪计算：保留每球实际入射速度，假设拍面
完全达到原目标法向和速度，再用相同恢复系数计算出球并作自由飞行投影。
30 个样本中只有 1 个投影满足有效回球，其余目标本身在这些来球条件下也未产生合法落点。
这不是实测成功率，不包含完整碰撞仿真；它支持“训练目标和有效回球指标存在差距”的判断，
不用于补偿策略，也不能据此宣称该 checkpoint 在所有来球下都不能得分。

最终严格测试原始目录为 `legal_return_plane_overnet_seed42`、
`legal_return_plane_overnet_seed7`、`legal_return_plane_overnet_seed123`。
早期 `legal_return_motion_plane_seed*` 尚未强制有限球网内的来球走廊，只是中间诊断结果，
不应与上述最终来球分布混合统计。

汇总与出球轨迹图：
`deploy/recordings/hope_g1/legal_return_audit_20261019/report.json`、
`deploy/recordings/hope_g1/legal_return_audit_20261019/return_trajectories.png`。

复现严格验收（当前结果不通过，退出码 1 是预期的验收结论）：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --render --sim-controls auto \
  --motion-select auto --motion-allow all --seed 42 \
  --episodes 10 --duration 70 --min-return-rate 0.8
```

复现轨迹汇总：

```zsh
.venv/bin/python -m deploy.hope_g1.validate_returns \
  --receiving-run deploy/recordings/hope_g1/legal_return_plane_overnet_seed42 \
  --receiving-run deploy/recordings/hope_g1/legal_return_plane_overnet_seed7 \
  --receiving-run deploy/recordings/hope_g1/legal_return_plane_overnet_seed123 \
  --output deploy/recordings/hope_g1/legal_return_audit_20261019
```

新增有效回球验收后的完整回归为 **162 passed，18 条已有资产 attach 警告**；
本次修改源码与对应测试的 Ruff 检查通过。未执行真机运动或修改训练 checkpoint。
