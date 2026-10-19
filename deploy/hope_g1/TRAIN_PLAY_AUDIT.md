# HOPE G1：train/play 与 MuJoCo 部署核对（2026-10-19）

已确认并修复部署中的参考坐标系和目标构造错误。本次验证的是实际球拍碰撞，
没有调整球拍半径、球半径、碰撞判定或 checkpoint。

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
- 命中表示球与拍面接触，不表示每一球都合法越网并落在对方桌面。

机器可读结果：`deploy/recordings/hope_g1/train_play_audit_20261019/report.json`。
原始运行目录分别为 `audit_baseline_10`、`audit_training_world_vectors`、
`audit_fixed_seed7`、`audit_fixed_seed123`，位于 `deploy/recordings/hope_g1/`。

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
