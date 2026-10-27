# HOPE G1 仿真测试与回看指令

核对日期：2026-10-26。本文汇总当前 `HOPEPingPongMultiMotionG1` 的仿真入口，所有命令均在本机项目根目录运行。旧的 TeleGMR / Hard 任务见文末链接，不与 HOPE G1 混用。

## 1. 选择测试入口

| 目的 | 模块 | planner | motion 选择器 / policy | 机器人是否在每条样本前重置 |
|---|---|---|---|---|
| 第一阶段：sample racket command，检查动作执行 | `deploy.benchmarks.racket_command.run` | 不调用 | 使用 | 是 |
| 第二阶段：随机发球，检查预测、执行与回球 | `deploy.benchmarks.receive.run` | 使用 | 使用 | 是 |
| 连续随机来球 MuJoCo 演示 / 部署流程验证 | `deploy.hope_g1.deploy_agent --env mujoco` | 使用 | 使用 | 否，连续执行 |
| 真机完整状态可视化回看 | `deploy.tools.view_real_log` | 不重算，显示已接受的记录指令 | 不重新执行，显示实测姿态 | 直接定位日志时间 |
| 真机来球轨迹驱动的新仿真 | `deploy.benchmarks.planner.recorded_receive` | 重新运行 | 重新运行 | 球与球之间不重置 |
| 单独检查 planner 的轨迹预测误差 | `deploy.benchmarks.planner.real_like` | 使用 | 不使用 | 无机器人控制 |
| 真机日志中的 planner 数值重算 | `deploy.hope_g1.replay_real` | 使用 | 不使用 | 无机器人控制 |

第一阶段的“policy 测试”包含现有 motion 选择器，因为该策略依赖参考动作；不是脱离 motion 的纯神经网络测试。第二阶段评测与连续部署共享策略等组件，但不是同一个程序。

## 2. 运行目录与配置

```bash
cd /home/lhy/table-tennis/Adapt-TeleAI
```

直接使用 `.venv/bin/python`，无需激活虚拟环境。以下本地仿真不需要启动 ROS、VRPN、桥接、真机程序或连接 .200。每次新测试使用新的 `--output`，冻结 suite 的输入目录与本次结果输出目录应分开。

| 配置 | 修改内容 |
|---|---|
| `deploy/hope_g1/configs/task.yaml` | checkpoint、motion 目录；当前为迁移后的 `model_18000.pt` |
| `deploy/hope_g1/configs/rally_mujoco_random.yaml` | 连续部署的随机发球范围、间隔、planner、球桌及物理参数 |
| `deploy/hope_g1/configs/rally_mujoco_random_a3_target.yaml` | 与上一份仅回球落点不同：`[2.5,0]` |
| `deploy/hope_g1/configs/rally_mujoco.yaml` | 独立评测 adapter 默认读取的 rally 配置；连续演示不使用它的历史 `motion_plane` 发球方式 |
| `deploy/hope_g1/configs/real.yaml` | 仿真复用的控制流程、AMP、PD 等运行参数 |
| `deploy/hope_g1/configs/mocap.yaml` | 估计器和位姿处理配置；默认仿真读取此配置不等于连接动捕 |
| `deploy/benchmarks/racket_command/default.yaml` | 第一阶段范围、采样、方向 / 速度 / TTS、判定阈值 |
| `deploy/hope_g1/configs/command_domain.json` | 冻结的 motion＋训练随机化支持域 |
| `deploy/benchmarks/receive/center.yaml` | 第二阶段随机球初始状态范围、接球区域、网格、每格球数 |

带 `--render` 的命令在桌面终端运行，先点击 MuJoCo 窗口再按键。无窗口评测可去掉 `--render`；无显示器导出图片 / 视频使用 `MUJOCO_GL=egl`。

加载策略和资源、检查配置，但不运行仿真：

```bash
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env check \
  --config deploy/hope_g1/configs/task.yaml \
  --rally-config deploy/hope_g1/configs/rally_mujoco_random.yaml \
  --motion-index 0 --motion-select auto --motion-allow all
```

## 3. 第一阶段：racket command → motion 选择 → policy

### 3.1 可视化网格测试

```bash
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --config deploy/benchmarks/racket_command/default.yaml \
  --command-only --render --playback-rate 0.5 \
  --seed 42 \
  --output "deploy/recordings/racket_command/policy_grid_$(date +%Y%m%d_%H%M%S)"
```

无球、无球轨迹估计、无回球 planner。直接采样位置、拍面法向、速度、TTS，检查指定 deadline 时刻球拍能否完成指令。`default.yaml` 已是 `policy_tracking`，这里显式加 `--command-only` 便于识别。

默认位置支持域先裁掉最低 Z 行，保留 8×5 个几何格；有训练位置支持的格数与最终可生成组合数以该次 `suite.json` 为准。默认每个有效格固定一个点，在该点测试多组方向、速度和 TTS；它不是每次都换位置。正常对打约束在采样时检查理想出球合理性，不调用 planner，也不按实际执行成绩挑样本。

主指标：指定 TTS 时刻，位置误差 ≤5 cm、法向误差 ≤15°、速度向量误差 ≤0.5 m/s 同时满足。`success` 还要求完成回位、没有失败。拒绝指令保留在分母；不能拿其他时刻的最近距离替代 deadline 成绩。

### 3.2 固定一个点，改变方向、速度、TTS

```bash
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --command-only --render \
  --point -0.8 0.2 \
  --directions 3 --speed-levels 3 --tts-levels 3 \
  --seed 42 \
  --output "deploy/recordings/racket_command/policy_point_$(date +%Y%m%d_%H%M%S)"
```

`--point` 是 Y、Z，X 由击球平面确定，示例点为 `[-0.25,-0.8,0.2] m`。最多 3×3×3=27 个组合；满足支持域及正常对打约束的实际数量可能减少，查看采样报告。

### 3.3 按 TTS 分组的粗网格纯策略测试

```bash
MUJOCO_GL=egl .venv/bin/python -m deploy.benchmarks.racket_command.run \
  --config deploy/benchmarks/racket_command/grid_tts.yaml \
  --command-only \
  --tts-values 0.3 0.4 0.5 --samples-per-cell 3 \
  --output "deploy/recordings/racket_command/policy_tts_$(date +%Y%m%d_%H%M%S)"
```

加 `--render` 可观看；各 TTS 档重复同一批位置 / 法向 / 速度 target。`grid_tts.yaml` 原配置是物理球诊断，**本命令通过 `--command-only` 切换为无球指令跟踪**。

### 3.4 全区域、多 TTS 的纯策略压力测试

```bash
MUJOCO_GL=egl .venv/bin/python -m deploy.benchmarks.racket_command.run \
  --config deploy/benchmarks/racket_command/grid_tts_full.yaml \
  --command-only \
  --output "deploy/recordings/racket_command/policy_tts_full_$(date +%Y%m%d_%H%M%S)"
```

此配置覆盖 8×5 的完整矩形，每格每档 50 个 target，TTS 为 0.2/0.3/0.4/0.5/0.6 秒，共 10,000 条。它保留训练支持域空洞和启动窗口外的指令；当前最大启动 TTS=0.5 秒，拒绝也如实统计。因此该压力测试不能直接解释为“正常域内策略成功率”。

### 3.5 冻结目标序列，再执行同一批指令

在同一个终端依次执行：

```bash
stage1_suite="deploy/recordings/racket_command/suite_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --generate-only --command-only --seed 42 \
  --output "$stage1_suite"

.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --suite "$stage1_suite/suite.json" \
  --command-only --render \
  --output "deploy/recordings/racket_command/policy_frozen_$(date +%Y%m%d_%H%M%S)"
```

这是重新执行冻结指令，不是播放历史机器人状态。常用覆盖参数：`--task-config` 换策略资源，`--motion-allow 0,1,2` 限制候选，`--limit 10` 只测前 10 条并标记 partial，`--position-sampling random` 在格内随机换位置，`--capture-first 3` 保存样本截图。

输出：`suite.json`、`results.jsonl`、`report.json`、`report.html`、`capability.png`；TTS 测试另有 `tts_grids/`。直接打开 `report.html` 查看交互热力图。

## 4. 第二阶段：随机球 → estimator → planner → 选择器 / policy

### 4.1 中心区域的物理接球评测

```bash
.venv/bin/python -m deploy.benchmarks.receive.run \
  --config deploy/benchmarks/receive/center.yaml \
  --return-target 2.055 -0.7625 \
  --seed 42 --render --playback-rate 0.5 \
  --output "deploy/recordings/receive/center_$(date +%Y%m%d_%H%M%S)"
```

初始球位置 / 速度独立采样，不从 motion 或 command 反推。球自由积分，按历史位置观测进入估计器和 planner。每条样本恢复同一准备状态，分别评测碰拍、有效回球、command 跟踪及回位。

默认仅覆盖中心区域：X=−0.25 m、Y=[−1.000,−0.525] m、Z=[0.12,0.42] m，3×2 格、每格 5 球，共 30 球。Z 相对于桌面。不要把这组成绩解释成全区域连续部署性能。

此入口默认 rally 来自 `rally_mujoco.yaml`，不提供 `--return-target` 时使用 A3 原配置落点 `[2.5,0,0]`；上面显式改成对方半台中心 `[2.055,-0.7625,0]`。

### 4.2 冻结同一批随机球，对比两种回球落点

```bash
stage2_suite="deploy/recordings/receive/suite_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m deploy.benchmarks.receive.run \
  --generate-only --seed 42 --output "$stage2_suite"

.venv/bin/python -m deploy.benchmarks.receive.run \
  --suite "$stage2_suite/suite.json" \
  --return-target 2.5 0 --render \
  --output "deploy/recordings/receive/a3_target_$(date +%Y%m%d_%H%M%S)"

.venv/bin/python -m deploy.benchmarks.receive.run \
  --suite "$stage2_suite/suite.json" \
  --return-target 2.055 -0.7625 --render \
  --output "deploy/recordings/receive/center_target_$(date +%Y%m%d_%H%M%S)"
```

`--balls-per-cell N` 调整新 suite 的每格球数；`--config` 更换区域、网格、初始 p/v 采样范围。使用冻结 suite 后不能再用 `--seed` 或 `--balls-per-cell` 改采样。`--capture-first N` 保存截图，`--limit N` 做部分测试。

输出：`suite.json`、`metadata.json`、`results.jsonl`、`traces/`、`report.json`、`report.html`、`capability.png`。碰拍率与有效回球率应分别查看，planner 误差与 deadline 跟踪误差也应分别查看。

## 5. 连续随机来球 MuJoCo 部署演示

### 5.1 手动进入策略，连续接球

```bash
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --render --sim-controls manual \
  --config deploy/hope_g1/configs/task.yaml \
  --rally-config deploy/hope_g1/configs/rally_mujoco_random.yaml \
  --motion-index 0 --motion-select auto --motion-allow all \
  --target-source a3_planner --a3-target-mode raw \
  --seed 7 --duration 600 \
  --output "deploy/recordings/hope_g1/continuous_$(date +%Y%m%d_%H%M%S)"
```

按顺序操作：

1. **Enter**：START，等待 `POSE_HOLD`。
2. **L**：平滑下放，等待终端 `support released`。
3. **A**：进入 AMP，等待 `STABLE`。
4. **W**：进入接球策略，等待 `WAIT_BALL` / ARMED。
5. 默认每 4 秒自动发一球。**T** 暂停 / 恢复发球；**F** 额外发一球。停发不影响已经飞出的球。
6. **S** 返回 AMP；**B/X** 停控；**R** 重置仿真；**Q** 退出。重置后重新走进入策略流程。

该配置是独立随机初始球状态的 `random_launch`，不是 `motion_plane` 反推发球；按来球几何筛选覆盖区域，机器人姿态、policy 历史、挥拍与回位连续保留。接球平面 X=−0.25 m，覆盖配置中的 8×5 网格包络，实际可生成来球的覆盖以该次日志为准，不意味着矩形每处均可达。

当前回球目标为对方半台中心。对比 A3 原落点时，只替换为：

```text
--rally-config deploy/hope_g1/configs/rally_mujoco_random_a3_target.yaml
```

### 5.2 自动流程与连续多球统计

```bash
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --sim-controls auto \
  --rally-config deploy/hope_g1/configs/rally_mujoco_random.yaml \
  --motion-index 0 --motion-select auto --motion-allow all \
  --target-source a3_planner --a3-target-mode raw \
  --seed 7 --episodes 10 --duration 90 \
  --output "deploy/recordings/hope_g1/continuous_auto_$(date +%Y%m%d_%H%M%S)"
```

无需按键，添加 `--render` 可看自动流程。`--episodes` 控制自动运行的目标挥拍数，`--duration` 是总时限，不是每球时限；没有接受指令的来球不必然形成一次挥拍，需检查实际发球分母。

可加 `--min-hit-rate 0.8` 或 `--min-return-rate 0.8` 设定退出判据；这些是验收门槛，不会改变来球或策略。未设置门槛时，`passed` / `flow_passed` 不能直接当作有效回球成绩。

### 5.3 只验证进入策略和首帧保持，不发球

```bash
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --render --sim-controls manual --standing-only \
  --rally-config deploy/hope_g1/configs/rally_mujoco_random.yaml \
  --motion-index 0 --motion-select auto --motion-allow all \
  --duration 120 \
  --output "deploy/recordings/hope_g1/posture_$(date +%Y%m%d_%H%M%S)"
```

同样执行 Enter → L → A → W。

连续部署输出重点：`summary.json`、`trace.jsonl`、`mocap_packets.jsonl`、`rally_config.yaml`；统计中查看 `simulated_balls`、`simulated_hits`、`hit_rate`、`simulated_legal_returns`、`legal_return_rate`、`completed_swings`、`recovered_swings`、`failure` / `error`。`ball_returns` 提供逐球回球证据。

## 6. 真机日志的 MuJoCo 状态回看

### 6.1 完整回看

```bash
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261022_121714
```

替换 `--recording`（别名 `--log-dir`）即可读取其他日志目录。该目录需要 `summary.json`、`trace.jsonl`；球观测来自可选的 `mocap_packets.jsonl`。无需 checkpoint。

从 30 秒开始，四分之一速度，打开后先暂停：

```bash
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261022_121714 \
  --start 30 --rate 0.25 --paused
```

回看直接显示实测关节角和根姿态，不重跑策略，不重新积分接触。绿色 / 蓝色 command 标记仅在日志显示机器人已接收指令并执行挥拍时出现，待球 home goal 不显示。橙色球来自已到达的实测样本；球缺测超时隐藏，不补出虚构轨迹。

坐标标定偏差按原日志保留，包括此前观察到的脚底约低于地面 5 cm；该现象不能通过回看自动修复。

### 6.2 导出慢放视频

```bash
MUJOCO_GL=egl .venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261022_121714 \
  --start 30 --end 35 --rate 0.25 --fps 30 \
  --video "deploy/recordings/hope_g1/review_$(date +%Y%m%d_%H%M%S)/swing.mp4"
```

5 秒日志按 0.25 倍速导出约 20 秒视频。视频含场景与标记，交互窗口额外显示状态文字。

仅检查日志字段与样本数量：

```bash
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261022_121714 --inspect
```

## 7. 用真机来球轨迹重新测试 planner＋人形策略

```bash
.venv/bin/python -m deploy.benchmarks.planner.recorded_receive \
  --recording deploy/benchmarks/planner/recorded_ball_cases_20261022.json \
  --render --manual --sensor-clock recorded \
  --repetitions 1 --duration 600 \
  --output "deploy/recordings/planner/recorded_receive_$(date +%Y%m%d_%H%M%S)"
```

执行 Enter → L → A → W，三条真机来球轨迹依次输入，间隔 5 秒，球与球之间不重置机器人。去掉 `--manual` 自动进入策略。

**这里 `--recording` 输入的是提取后的球轨迹 JSON，不是第 6 节的完整日志文件夹。** 原始点位只用于环境重现来球，planner 通过估计器收到因果观测，不能读取未来轨迹；机器人动作由当前策略重新生成。碰到仿真球拍后停止按记录位置驱动球，转为物理碰撞及自由飞行。

`--sensor-clock recorded` 保留记录的消息时间戳；`capture` 使用重建采集时钟做单独诊断。后者不是实测曝光时钟，也不是已上线修复，两者结果须分开。

自动重测并导出视频：

```bash
MUJOCO_GL=egl .venv/bin/python -m deploy.benchmarks.planner.recorded_receive \
  --recording deploy/benchmarks/planner/recorded_ball_cases_20261022.json \
  --sensor-clock recorded --video --duration 45 \
  --output "deploy/recordings/planner/recorded_video_$(date +%Y%m%d_%H%M%S)"
```

输出：部署日志、`diagnosis.json` 和可选的 `humanoid.mp4`。这是新的仿真结果，不是对真机动作的回放。

## 8. planner 单独诊断

### 8.1 相似来球的物理预测 / 时间戳敏感性测试

```bash
.venv/bin/python -m deploy.benchmarks.planner.real_like \
  --render --playback-rate 0.5 \
  --output "deploy/recordings/planner/real_like_$(date +%Y%m%d_%H%M%S)"
```

默认读取已有 `deploy/recordings/hope_g1/diagnostics_20261022/latest_two_runs_diagnosis.json`，从真机来球拟合初始 p/v，再进行物理轨迹测试。可用 `--diagnosis` 替换诊断文件，或 `--suite` 复用此模块生成的 `suite.json`。`--phases` 控制轮询相位数，`--noise-mm` 增加观测噪声。

对比理想状态、采集时间戳、批量回调时间戳，以及反弹参数敏感性。没有 motion 选择器或 policy，不能用其误差达标率代替碰拍率。输出：`results.json`、`suite.json`、`planner_errors.png`、`trajectory_comparison.png`、`physical_trajectories.npz`。

### 8.2 在真机原触发时刻重算 planner

```bash
.venv/bin/python -m deploy.hope_g1.replay_real \
  --summary deploy/recordings/hope_g1/real_20261022_121714/summary.json \
  --ball-samples deploy/recordings/hope_g1/real_20261022_121714/mocap_packets.jsonl \
  --rally-config deploy/hope_g1/configs/rally.yaml \
  --mocap-config deploy/hope_g1/configs/mocap.yaml \
  --output "deploy/recordings/planner/numeric_replay_$(date +%Y%m%d_%H%M%S)/report.json"
```

无窗口、无机器人执行，比较原触发时刻的预测与观测穿平面位置 / 时间。复现历史实验时，rally 和 mocap 参数也需与该次记录一致；直接使用今天的配置会成为“当前算法 / 配置对历史数据的重算”。

### 8.3 第二阶段结果的理想球拍离线诊断

将路径替换为第 4 节已经完成的输出目录：

```bash
.venv/bin/python -m deploy.benchmarks.receive.audit \
  --recording deploy/recordings/receive/你的已完成测试目录
```

输出 `ideal_paddle_audit.json`。这是检查已锁定 command 物理效果的辅助诊断，不修改真实测试成绩，不向策略提供理想球拍控制。

### 8.4 真机回看脚底 / 根位置一致性检查

```bash
.venv/bin/python -m deploy.tools.audit_real_pose \
  --recording deploy/recordings/hope_g1/real_20261022_121714 \
  --recording deploy/recordings/hope_g1/real_20261022_121834 \
  --output "deploy/recordings/hope_g1/pose_audit_$(date +%Y%m%d_%H%M%S)/report.json"
```

只读重放记录中的 UDP 根位姿处理，核对日志与 viewer 的关节 / 根位置 / 球拍 FK，比较静止待球时外观脚底与训练接触盒的高度。不修改记录或标定，不连接机器人。脚底穿地只说明记录位姿与模型 / 地面不一致，不能单凭该结果断言是动捕位置错误。

可选 `--nominal-offset X Y Z` 做**替换下游 offset 的离线假设对照**；它不会修改任何部署配置，且不代表应该把动捕已应用的补偿再加一次。上游已应用 offset 时，下游为零可以是正确设置。`--marker-csv` 可核对十点布局的球心均值；均值的反向量只有在刚体原点为该均值、坐标轴与 pelvis 对齐时才等于实际外参。

## 9. 历史物理球诊断与并行压力测试

以下模式按 sampled command 构造来球，**不等于独立随机发球的第二阶段**，也不是无球的第一阶段。

可视化诊断：

```bash
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --config deploy/benchmarks/racket_command/grid_tts.yaml \
  --render \
  --output "deploy/recordings/racket_command/command_ball_$(date +%Y%m%d_%H%M%S)"
```

全区域 10,000 条并行诊断，在同一终端先保存输出目录再执行：

```bash
batch_output="deploy/recordings/racket_command/physical_full_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/hope-tts-inductor TORCHINDUCTOR_COMPILE_THREADS=1 \
.venv/bin/python -m deploy.benchmarks.racket_command.batch_run \
  --config deploy/benchmarks/racket_command/grid_tts_full.yaml \
  --workers 4 --output "$batch_output"
```

按本机 CPU / 内存增加 `--workers`；默认完整配置启用编译加速，需要本地 C++ 编译工具链。中断后保留同一个目录执行：

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/hope-tts-inductor TORCHINDUCTOR_COMPILE_THREADS=1 \
.venv/bin/python -m deploy.benchmarks.racket_command.batch_run \
  --config deploy/benchmarks/racket_command/grid_tts_full.yaml \
  --workers 4 --resume --output "$batch_output"
```

`batch_run` 没有 `--command-only` 参数；若要并行做纯策略测试，应先另存 YAML 并将 `evaluation_mode` 改为 `policy_tracking`，再通过 `--config` 指定，不能把默认并行物理诊断称为纯策略评测。

## 10. 播放操作对照

| 操作 | 连续 MuJoCo / 真机来球驱动仿真 | 第一 / 第二阶段评测 | 真机状态回看 |
|---|---|---|---|
| 减速 / 加速 | `[` / `]`，也可 `-` / `+` | 相同 | 相同 |
| 恢复 1 倍速 | `\` | `\` | `\` |
| 暂停 / 继续 | 空格 / P | 空格 / P | 空格 |
| 重启 | R：重置仿真，重新走启动流程 | 无通用 R 重启 | R：从日志 0 秒重播，保留倍速 / 视角 |
| 逐步 | N：仿真单步 | 无通用逐步键 | `,` / `.`：上一 / 下一帧并暂停 |
| 前后跳进度 | 不支持倒放物理状态 | 不支持倒放物理状态 | 左 / 右：后退 / 前进 1 秒 |
| 上 / 下一阶段 | 无此回看操作 | 无此回看操作 | P / N |
| 退出 | Q | Q / Escape | 关闭窗口 |

`real_like` 的专用展示支持变速，其余操作以该窗口提示为准。变速只改变墙钟播放等待，不改变物理步长、policy 频率、TTS 或观测时间戳；评测初始化倍速使用 `--playback-rate`，真机状态回看使用 `--rate`，连续部署入口不支持这两个 CLI 参数，运行时用按键变速。

## 11. 进一步说明与历史入口

- [第一阶段详细参数、采样与指标](benchmarks/racket_command/README.md)
- [第二阶段详细参数与输出](benchmarks/receive/README.md)
- [planner 及真机来球驱动仿真说明](benchmarks/planner/README.md)
- [真机状态回看说明](tools/REAL_LOG_VIEWER.md)
- [HOPE G1 部署说明](hope_g1/README.md)
- 旧 TeleGMR / Hard 任务：[play 与 MuJoCo](PLAY_AND_MUJOCO.md)、[旧任务部署与测试说明](README.md)。

`deploy.tools.replay_deployment` 和 `deploy.tools.replay_operator_plant` 是旧 `g1_pingpong` 控制器的诊断工具，不是 HOPE G1 的真机状态 viewer，不能直接把 HOPE task 配置传给它们。`deploy.hope_g1.validate_returns` 的旧离线名义投影当前要求 `drag=0`，与现用 `drag=0.1537` 不兼容；当前回球评分使用测试自身的物理记录，不为运行旧诊断而改球参数。
