# G1 model_18000 与 A3 部署对标（2026-10-20）

**本文件是修改前的核对记录。完整 A3 目标链路已在本机接入，当前实现与 30/30
有效回球验收见 [A3_ALIGNMENT_RESULT.md](A3_ALIGNMENT_RESULT.md)。以下“缺失/未接入”
描述保留为原问题的证据，不代表当前状态。**

当前 G1 已通过实际 Isaac 观测/actor 数值对拍，但还不是 A3 的完整接球部署：
缺少 planner 到策略的目标拍速/法向命令链路。之前的有效回球 0/30 仅描述固定参考目标
baseline，不能单独据此判定 checkpoint 的实际回球能力。

## 对标来源

- 用户指定仓库 `handsomehhh/HOPE-TeleAI`；网页访问返回 404，使用本机已有 checkout
  与 `.200:/home/lxz/HOPE` 实际代码核对，没有用另一个上游版本代替。
- 两个 checkout 的 HEAD 均为 `67a1a720e985b01ec157821c9809e63bcbb13d9a`，
  `.200` 有大量工作区修改，故以实际工作区快照为准，不能只依据 commit 判断代码相同。
- A3：`README-TeleAI.md`、`hope_deploy_main.cpp`、`hope_deploy_adapter.hpp`、
  `hope_dynamic_obs_builder.cpp`、`hope_motion_clock.hpp`、`HOPE_OBSERVATION_SCHEMA.md`。
- G1：实际 `scripts/play.py`、`HOPEPingPongMultiMotionG1.yaml`、mdp 实现、
  指定 run 的 `params/env.yaml`、`params/agent.yaml`、TensorBoard 与 checkpoint。
- 原始代码、实际 Isaac 数据和校验值保存在
  `artifacts/hope_20261012_18000/a3_audit_20261020/`，含 `remote_sources.tar.gz` 和 manifest。
  本轮未改 `.200` 的训练/部署源码、checkpoint 或启动真机电机；诊断在 `/tmp` 中运行。

## 已确认的问题与修正

| 项目 | A3 实现/实际训练 | 当前 G1 与结论 |
| --- | --- | --- |
| 目标命令完整性 | `/racket/command` 提供接触位置、目标拍速、目标法向、时刻；锁定后输入 actor | **缺失**。`ReceivePlanner` 只预测位置/时刻，`reference_goal` 固定 `velocity=sign×3×reference_normal`、`normal=reference_normal`。没有消费 A3 的完整 RacketCommand |
| 扰动范围所属坐标系 | A3 `LatchRacketTargetFromCommand` 先将位置变换回 training world，按训练坐标轴检查/约束 | **本轮修复**。原选择器直接沿 deploy world 轴检查，现 `offset_train=R.T@offset_deploy`。旋转 45° 的接受/拒绝边界测试通过 |
| 已训练扰动尺度 | TensorBoard iteration=18000：`curriculum_scale=1.0`，位置每轴 ±0.25m | **本轮修复配置**。原默认按 play 的起始值 0.15，只允许 ±0.0375m；现显式 `target_perturb_scale: 1.0`，保留已有 0.25m 欧氏距离门槛 |
| 绝对速度/法向坐标 | A3 legacy HOPE 将向量逆变换到训练 world，再组观测 | 前轮已修正，G1 相同。不能改成 raw deploy-world，也不能套用不同 HITTER 观测契约 |
| base target 耦合 | A3 按参考动作 reach offset 耦合接触点与机身目标 | G1 已实现 XY 耦合，使用 NPZ pelvis，而非把 torso 当 pelvis |
| 固定 home / 回位 | A3 支持 fixed-world-home、frame-zero hold、75 帧回位 | G1 实现同类调度；前轮已修复末帧提前进入回位的 20ms 偏差 |

### 1. 最大缺项是完整目标输入，而非碰球后的补偿

A3 的标准链路为：球状态估计 → 轨迹/出球目标 planner → RacketCommand →
按选定动作和正反手约定映射目标 → actor → 关节命令。

当前 G1 为：球状态估计 → 接触位置/时刻预测 → 固定参考拍速/法向 → actor → 关节命令。
`ball.target_xy` 被用于模拟目标/误差统计，但当前 G1 没有用它计算送给 actor 的拍速和法向。
所以即使仿真显示了对方落点，策略也没有收到“用什么拍速/拍面朝那里回球”的目标命令。

这是策略任务输入接口缺失。接入标准目标命令与碰球后覆盖球速、修改策略关节输出是不同的
操作；本轮未加入后两者，也未自动加入新的落点求解器。配置和启动日志现明确标记
`target_source: reference`，不支持的来源会报错，避免把 reference baseline 误认作完整 A3 模式。

A3 至少有两种目标模式，不能混用：

- envelope 模式：位置/法向按训练空间约束，速度保留参考方向、用 planner 的速度大小缩放。
- README 使用的 `--no-target-clamp` 模式：位置、物理拍速向量不替换；法向按正反手转换到
  拍面本地 +Y 的观测约定，速度不能因反手再翻一次符号。

G1 是 `ref_velocity_perturb_mode=scale` 训练，任意方向的 raw planner 拍速未必在训练支持内。
后续如接命令，应分别报告 raw 命令、策略实际收到的目标、越界/拒绝情况与回球结果，
不能以外部替代动作或重置球速掩盖跟踪失败。

### 2. play 启动尺度不等于 checkpoint 已训练尺度

`.200` 该 run 的 TensorBoard 文件为
`events.out.tfevents.1791787662.lxz-System-Product-Name.2886550.0`。
在 step=18000，`Metrics/` 和 `Live/` 下的 `ref_perturb_scale`、`curriculum_scale`
均为 1.0，`curriculum_pos_half_extent_mean` 均为 0.25。
原始查询结果保存于 `curriculum_receipt.json`。

play 只加载 actor/normalizer，不恢复 command curriculum，因此重新从 0.15 开始。
这解释了为何配置文件初始值是 0.15，但不能据此把 ±0.0375m 当作已训练策略的最大支持。
如需专门复现 play 重启初期的目标分布，应显式设 0.15；部署尺度现在明确设为已核实的 1.0。
尺度更大也不保证每个目标都跟得住，不能把“训练采样范围”解释为成功保证。

## 实际 Isaac → 本机部署数值对拍

在 `.200` 使用用户指定 task/checkpoint，2 个环境、220 tick，共 440 个状态/目标快照。
关闭观测噪声和 push，保留其余 play 随机化；无 ONNX 导出，无 checkpoint 写入。

逐项把 Isaac 的实际 root/torso 状态、采样目标、上一 raw action 和源 NPZ 帧交给本机观测器，
再比较 actor。为隔离接口与传感器差异，此诊断使用 Isaac 瞬时线速度，不更改生产动捕差分。

| 指标 | 结果 |
| --- | --- |
| 样本 | 440，覆盖 6 个动作，含 150 个 return 样本 |
| 58 维 motion command / NPZ 顺序最大误差 | 0 |
| 175 维观测最大绝对误差 | 3.7700e-6 |
| 29 维 actor 输出最大绝对误差 | 7.4208e-6 |
| 当前 task 与保存配置 | action、PD、init_state、sim 参数一致；command 仅增加两个值为 null 的配置项 |

有一项必须控制：训练/play startup 会随机改变各环境 `default_joint_pos`，并同时改变
action offset。此次实际最大零位扰动为 0.0095965rad。若拿同一个 Isaac q，但减去本机的
名义零位，`joint_pos_rel` 差约 0.0096、actor 输出可差 0.0662。
使用该环境实际零位后才得到上表的契约误差。本机名义零位是确定性基线，不能把 play 的
随机零位当作应永久复制到真机的标定值，也不能声称两套 plant/传感器逐帧完全相同。

结果：`deploy/recordings/hope_g1/a3_comparison_20261020/isaac_contract_comparison.json`。
复现（本机已迁移数据）：

```bash
.venv/bin/python -m deploy.hope_g1.compare_isaac_contract \
  --isaac-contract deploy/hope_g1/artifacts/hope_20261012_18000/a3_audit_20261020/contract.npz \
  --output deploy/recordings/hope_g1/a3_comparison_20261020/isaac_contract_comparison.json
```

## 仍需区分的部署差异

- A3 的实时选择分支会按当前 torso 对候选动作单独做 yaw rebase；G1 当前让动作库共享
  policy-entry 时冻结的变换。前者会重新居中，后者保留 NPZ 之间的偏移，是不同调度。
  不能为了“对标”直接改掉并宣称一定正确；需在相同来球/目标命令下单独 A/B。
- A3 的 `x_hit` 也是世界 X 常量的虚拟截面，不能误认为默认沿实际拍面法向求交。
  本机发球使用参考拍面构造入射目标；接触预测仍用 X 截面，两者概念不同。
- 本机 court 为 x∈[0.65,3.39]、y∈[-0.7625,0.7625]；仓库 planner 的 canonical
  table 为 x∈[0,2.74]、y∈[-1.525,0]。接它之前必须明确 world 变换。
  只比较这两套几何时，平移为 `[0.65,0.7625,0]`，不能未经现场标定就把它硬加到动捕。
- 仓库 optimized planner 配置的 drag/restitution 与本机零阻力基线不同；应让接入的
  planner 使用当前实验 plant 的参数，而非直接抄 A3 的标定常数。
- G1 当前 MuJoCo 资产关闭自碰撞，训练开启；物理引擎、地形/随机化、传感器速度来源仍有
  明确差异。实际 Isaac 契约对拍通过不表示这些动力学差异已消失。

## 修复后的复测

仅修正检查坐标轴（仍为旧尺度 0.15）的 seed=42 复测：碰拍 10/10，有效回球 0/10，
挥拍/回位 10/10，未失去站立。因此该坐标边界错误并非这组回球失败的主要原因。
记录：`deploy/recordings/hope_g1/a3_comparison_envelope_fixed_seed42/`。

位置检查坐标修正且显式使用已训练尺度 1.0 后，seed=42 再测为：碰拍 10/10、
有效回球 0/10、挥拍/回位 10/10；最大倾角约 7.68°，未检测到失去站立。
记录：`deploy/recordings/hope_g1/a3_comparison_trained_scale_seed42/`。
说明放回正确输入支持范围也不能替代缺失的目标拍速/法向命令链路。

本报告的判断是：基础 actor 接口已获得实际 Isaac 数值证据；已修复两个选择范围问题；
完整 A3 planner 目标链路尚未接通。下一步应对标“同一个完整目标命令，策略能否达到”，
而不是把固定参考模式的回球统计直接解释为 checkpoint 的能力上限。

本轮验证：完整部署测试 **165 passed**（18 条已有资产 attach 警告），Ruff 检查通过。
Isaac 采样和本机契约比较脚本分别为 `isaac_contract_dump.py`、`compare_isaac_contract.py`。
