# G1 乒乓球真机正式部署

运行主机：`192.168.152.200`  
项目目录：`/home/lxz/deploy_g1_pingpong`  
终端 Shell：**zsh**。所有命令直接在 zsh 中执行，不要先运行 `bash`，ROS 环境统一加载 `setup.zsh`。

本文仅包含正式部署的启动流程，不包含安装、仿真和只读测试命令。终端 3 的命令含 `--enable-motors`，会允许机器人控制。

基础 play 与 deploy MuJoCo 测试命令见 [PLAY_AND_MUJOCO.md](PLAY_AND_MUJOCO.md)。

## 启动前提

- 机器人网卡为 `eno1`，目标地址为 `192.168.123.164`。**2026-10-15 本轮检查时 eno1 为 DOWN、carrier=0、无 IPv4，尚未接通机器人链路；完成接线和机器人开启后，需要恢复该接口的机器人网段地址并重新验证 LowState。** 2026-10-14 的历史地址为 `192.168.123.165/24`，曾测得 ping 3/3、DDS 约 1000Hz、mode_machine=5，这不是当前连通性结论。此前保存的 `g1-direct-eno1` 配置使用 `192.168.123.210/24`，也是同网段有效地址。部署命令继续使用 `eno1`。
- 机器人是与模型匹配的 29 自由度 G1，实际 `mode_machine` 与配置值 5 一致，处于允许 SDK 低层接管的状态，没有其他电机控制程序同时接管。
- `deploy/configs/mocap_g1_live.yaml` 已建立并完成现场标定：桥接的球输入为 `/poses`（`PoseArray.poses[0]`），机器人输入为 `/P1/pose`（`PoseStamped`），单位已为米。标记身份、消息 frame、坐标变换、机器人刚体到 pelvis 的偏移均已核实，球和机器人位姿持续有效。
- 只读联调已通过；现场有人员操作急停并具备防跌落措施。

**2026-10-14 14:28—14:32 实测：新版 AMP 流程的控制代码、模型和本文已同步 .200；AMP 模型可正常加载并完成离线推理。VRPN 客户端和 vrpn_to_pose 正在运行，后者实际参数为 robot_side=P2。12 秒订阅检查中，/poses 与 /P1/pose 均约 180Hz，时间戳持续更新，分别标记 world 和 map；临时启动本文的桥接并用部署 UDP 接收器测试 12 秒，accepted=3752、rejected=0，预热后 37/37 次检查球和机器人位姿均有效。临时桥接和接收器已退出，UDP 15050 端口已释放；已有 VRPN 客户端和转换器保持运行。机器人由操作人员之后开启，本次未验证机器人 DDS 或运动。**

桥接配置按输入分别校验标签，scale=1.0，不重复进行 P2 变换。按用户确认，上游已输出标定后的 pelvis；deploy 保持零偏移、单位旋转，不重复应用安装偏移。以上检查确认消息和桥接链路，不独立证明现场几何标定精度。

## 2026-10-15 最新诊断版

运行版本 `20261015-return-audit-12`。MuJoCo full/standing 与真机共用控制循环。此前新增的收拍双臂目标平滑已完整撤回，无 `return_target_seconds` 混合逻辑。START/A/UP/DOWN 与初始固定计时流程保持不变，默认 Kp/Kd 数值保持此前基准，现由 `deploy/g1_pingpong/pd_gains.py` 逐关节显式定义，不从训练配置/XML/AMP metadata 读取执行值。`policy_pd` 与 `stable_policy.pd` 分别保留两组 scale，详见 [PD_GAINS.md](PD_GAINS.md)。

按用户要求，正式启动不查询、不释放、不切换 MotionSwitcher 的 ai 模式。自动快速振荡停控已关闭，出现抖动由操作人员 SELECT/B/X 停止。START → 默认姿态、A → AMP稳定策略、UP → 乒乓球策略、DOWN → AMP流程保持不变；数据过期、关节范围、速度硬上限及倾角保护保留。

通信进程的队列等待为1ms；实际网络与电机总延迟仍未辨识。真机正运动学只更新位姿和速度，不计算动捕坐标下的虚拟地面接触力；它与原路径的状态结果一致。此次改动允许观察原有抖动并减少计算开销，不代表闭环稳定问题已修复。完整复查见 `deploy/FULL_SHAKING_AUDIT_20261014.md`。

`deploy/configs/real_damping_trial.yaml` 单独保存实验增益：Kp 不变，腿/手腕 Kd 为训练值 1.5 倍，腰为 3 倍，肩/肘为 2 倍。它通过部分长时、延迟和挥拍仿真，但加质量、低摩擦和反馈噪声试验仍暴露失稳或持续运动。正式命令继续使用 `real.yaml`，不自动启用该实验配置；不要把此候选视为已修复真机抖动。

完整本次证据见 `deploy/SHAKING_DIAGNOSIS_20261014.md`。


当前回位计时为 `recovery_seconds: 0`、`ready_seconds: 0`，完成挥拍后下一周期即可响应新球；`return_blend_seconds: 0`，`stage2_tail_mode: full`。START/A/UP 初始接管计时不变。2026-10-15 本轮同步状态与实测链路见 [SYNC_200_20261015.md](SYNC_200_20261015.md)。仿真物理超限只记录的观察模式不应用于真机；真机保留现有实测范围、速度、倾角与数据过期保护。

## 终端分工

| 终端 | 功能 |
| --- | --- |
| 终端 1 | VRPN 动捕客户端，接收球与机器人刚体（当前 ID 为 30310 与 4） |
| 终端 2 | 将动捕数据变换到 G1 坐标并转发至本机 UDP |
| 终端 3 | G1 策略与机器人控制 |

三个终端均在 .200 上运行，依次启动并保持运行。本流程依赖已有 `vrpn_to_pose` 后台转换器，实际链路为 VRPN → vrpn_to_pose → /poses 与 /P1/pose → G1 桥接 → UDP → G1 部署。转换器必须持续运行，不需要旧 HOPE planner 或 A3 控制程序。同一 VRPN 客户端或 G1 桥接已经运行时，不要重复启动。

## 终端 1：VRPN 动捕客户端

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh

export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd /home/lxz/HOPE/hope_ws

ros2 launch ./src/vrpn_mocap/vrpn_mocap/launch/client.launch.yaml \
  server:=192.168.1.101 \
  port:=3883 \
  allowed_topics:="/vrpn_mocap/MCAvatar/pose4,/vrpn_mocap/MCAvatar/pose30310"
```

若后台转换器退出，在额外终端加载相同 ROS 环境后启动（不要重复启动）：

```zsh
ros2 launch vrpn_to_pose vrpn_to_pose.launch.py \
  robot_side:=P2 \
  robot_input_topic:=/vrpn_mocap/MCAvatar/pose4 \
  ball_input_topic:=/vrpn_mocap/MCAvatar/pose30310
```

## 终端 2：G1 动捕桥接

```zsh
source /opt/ros/humble/setup.zsh

export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd /home/lxz/deploy_g1_pingpong

/usr/bin/python3 -m deploy.io.mocap bridge \
  --config deploy/configs/mocap_g1_live.yaml
```

桥接配置的 UDP 目标应为 `127.0.0.1:15050`。终端 3 将绑定该端口，正式运行时不能同时开启 UDP monitor 或第二个部署进程。

## 终端 3：正式部署

使用基础模型、home=19，自动从动作 19 和 24 中选择。该组合已有仿真验证，尚未完成真机运动验证。控制流程为 START 默认姿态、A 稳定策略、UP 乒乓球策略。

```zsh
cd /home/lxz/deploy_g1_pingpong

.venv/bin/python -m deploy.g1_pingpong.deploy_agent \
  --env real \
  --config deploy/configs/g1_tele_gmr.yaml \
  --real-config deploy/configs/real.yaml \
  --mocap-config deploy/configs/mocap_g1_live.yaml \
  --rally-config deploy/configs/mujoco.yaml \
  --network-interface eno1 \
  --ball-source mocap \
  --motion-index 19 \
  --motion-select auto \
  --motion-allow 19,24 \
  --duration 600 \
  --enable-motors \
  --output "deploy/recordings/real_$(date +%Y%m%d_%H%M%S)"
```

使用项目独立环境 `.venv/bin/python`，不需要激活 Conda。`--duration 600` 表示最多运行 600 秒；`--episodes` 不是真机自动停止条件。

首次真机单动作试验时，将 `--motion-select auto --motion-allow 19,24` 替换为 `--motion-select fixed`，固定执行动作 19。

## 手柄操作顺序

1. 启动后为 `WAIT`，不发送位置控制目标。
2. 按 **START**：`POSITIONING`，从当前关节角平滑进入 `deploy_lhy` AMP 模型的默认站姿（默认 3 秒）。使用同一模型的站立 PD，不运行学习策略。
3. 显示 **POSE_HOLD** 后保持该姿态。此阶段可吊起摆姿；由现场人员将机器人放稳到地面。插值完成不能证明机器人可以无支撑站立。
4. 松开后按 **A**：`TO_STABLE → STABLE`，平滑切入 AMP 稳定策略，默认过渡约 0.4 秒。该策略根据 pelvis IMU 和关节反馈保持平衡，速度指令固定为零，摇杆不会触发行走；不依赖动捕。
5. 确认已站稳后重新按 **方向键 UP**：先进入 `POLICY_POSTURE`，约 1.5 秒把腰和双臂移到所选 home 动作第 0 帧，腿部仍由 AMP 保持平衡；固定保持 0.5 秒后进入 `TO_POLICY`，关节跟踪误差、速度与 gyro 仅记录，不阻止切换。此后约 1.5 秒混合 PD 力矩接管，参考从 home 第 0 帧开始，上一动作清零并随后保存原始策略动作。上半身准备计时结束前不运行乒乓球 tracker；进入 tracker 时需要新鲜、已标定的动捕。
6. 初始接管渐变完成后固定等待 **0.5 秒**，显示 **WAIT_BALL** 并自动等待来球；初始渐变和等待期间禁止触发挥拍。有效来球触发 `EXECUTE → RECOVER → WAIT_BALL`，回位和 READY 附加等待均为 **0 秒**，不做实测站稳门控，不需要每次重新按键。
7. **方向键 DOWN**：从乒乓球任意阶段平滑返回 AMP 稳定策略，取消当前挥拍。下一次按 UP 时重新建立乒乓球策略历史和世界参考。
8. **SELECT / B / X**：停止并进入阻尼退出；这些按键优先于模式切换，不承担返回稳定策略的功能。

按键切换要求重新按下对应按键；过早按住 A/UP 不会在之后自动激活。按 UP 后，姿态准备、保持、策略接管自动依次完成，不需要再按一次键。默认姿态、AMP 稳定策略、乒乓球策略是三个不同控制阶段。

稳定模型和固定深度输入来自 `192.168.152.225:~/mjlab-prior/deploy_lhy/ckpts`，football 分支，模型与观测合同保存在 `deploy/assets/stable_amp/metadata.json`。该模型使用固定深度输入，不需要相机。

`deploy/configs/real.yaml` 的 `stable_policy` 控制接管：`stable_transition_seconds=0.4`、`transition_seconds=1.5`、`settle_seconds=0.5`；新增上半身准备时间 `preposition_seconds=1.5`、保持时间 `preposition_hold_seconds=0.5`。返回 AMP 仍使用每个 20ms 周期最大 0.15rad 的目标变化界限。进入乒乓球时通过提前摆姿及 PD 力矩混合完成接管，不对学习策略的虚拟目标再施加逐帧追赶限速；关节速度硬上限、实测关节范围、倾角、数据过期及停止保护保持有效；站立阈值仅作日志诊断。

以下为历史诊断过程，旧版稳定门控不属于当前固定计时流程。2026-10-14 两次真机记录（15:01:56、15:03:55）均未通过当时的 TO_POLICY 稳定检查，没有进入 ARMED/EXECUTE。随后修正过渡中的目标混合：使用 Kp 加权的目标，使限速前的 PD 力矩为两个控制器力矩的平滑混合，避免目标角和 Kp 分别插值造成的交叉放大。该力矩混合仍保留；旧的实测稳定门控已被固定计时替代。

随后 15:31:42、15:33:08 的记录确认力矩混合已生效，但仍振荡。对照 .200 的 HOPE A3 源码后，进一步修正了上半身提前摆姿、策略历史和姿态来源。`root_orientation_source=imu_mocap_yaw` 使用 IMU 倾角与动捕航向，统一用于 pelvis、torso FK 和 projected_gravity；动捕位置和 marker→pelvis 零补偿保持原值。这里参考 HOPE 的接管语义，不复制 A3 的关节表、模型或增益。新增日志保存 `root_quat_raw_mocap`、策略实际使用的 `root_quat`、`stable_source_target_q` 和姿态准备指标。最新实现已通过离线验证，仍需新的真机日志验收；上段延迟失稳结果属于较早实现。

本版使用 `readiness_mode: timed`。这是对照本机 HOPE `67a1a72` 的固定 ramp/hold 逻辑后的更新：前文较早版本的两道实测稳定门控是本项目增加的，不能视为 HOPE 原有要求。START/A/UP/DOWN 按键功能保持不变。实测 `stable`、关节误差、速度、gyro 和高度继续记录；它们不阻止策略接管、等待来球或回位后再次接球。日志内部兼容旧状态名 `ARMED`，终端显示 `WAIT_BALL`；该状态没有实测站稳判定。固定计时修改不等于已修复此前的闭环抖动，PD 数值保持原值。

动捕坐标与 marker→pelvis 标定仍需正确。启动命令不变；退出旧终端 3 进程后重新启动才能加载更新。该流程已有离线仿真验证，尚未完成真机运动验收。

## 延迟诊断日志（2026-10-15）

终端命令不变。新版按命令编号关联SDK反馈、跨进程收发、策略计算、命令入队和每次DDS写入，保存 `command_writes/commands_*.npy` 以及关联编号扩展后的200Hz `lowstate/*.npy`。正常停止或异常退出后自动生成 `latency_summary.json`，按控制阶段汇总主机时序。原始日志采用有界后台写盘，额外开销和丢弃情况会记录在summary。详细字段、时钟限制及可选响应分析见 `deploy/LATENCY_LOGGING.md`。这些记录用于排查，不能把Write成功等同于电机执行。

## 停止与记录

控制中先触发停止，并由现场措施或既定控制方式稳定接管机器人，再退出终端 3。不能把 `Ctrl+C` 或进程退出视为机器人会持续安全站立。之后再停止终端 2 和终端 1。

每次运行生成独立目录：

```text
/home/lxz/deploy_g1_pingpong/deploy/recordings/real_日期_时间/
├── summary.json
├── trace.jsonl
├── mocap_packets.jsonl
├── lowstate/
│   ├── schema.json
│   └── lowstate_*.npy
└── summary.fault.json  # 仅异常退出时生成
```

`summary.json` 记录控制是否启用、命令数、阶段事件、挥拍与回位次数、退出原因和故障；`trace.jsonl` 记录逐周期反馈及控制目标。


## 抖动诊断记录

默认开启 `record_io_diagnostics`。`trace.jsonl` 会记录同帧 LowState tick/`tau_est`、命令时间戳及 DDS Write 时序，以及 `mocap_imu_gravity_error_degrees`。后者比较两种输入在 pelvis 坐标中的重力方向，忽略不同世界航向；持续偏差需结合采样时延与 marker 外参检查。DDS Write 返回不代表电机已经执行。上游动捕已按用户说明输出 pelvis；deploy 的零偏移是有意保留，不能据此判断未标定。AMP阶段站稳不能替代乒乓球策略的验收。

上游已使用 marker→pelvis 位置偏移 `[0.011898400, 0.000002657, 0.086323220]` 米及单位旋转完成标定。该值只在上游应用，`mocap_g1_live.yaml` 中的 `root_marker_to_pelvis` 保持 `[0,0,0]`。机器人输入为 `/P1/pose`；当前 `/poses` 数组用于乒乓球，不要混淆两者。

2026-10-14 补充诊断：启动命令无需增加参数，保持原有 `--output`。退出并重新启动终端 3 后自动生效；现有运行进程不会加载这些更新。采集也支持不带 `--enable-motors` 的只读运行。

| 记录 | 新增内容及用途 |
| --- | --- |
| trace.jsonl，约 50Hz | 无球站立也保存 tracker 的完整输入与动作；AMP 保存 57 维输入、16 维动作、历史重置编号和步号。可区分 Stage1、Stage2 与 AMP，回放策略计算。 |
| trace.jsonl | 原始策略目标、过渡前后目标、混合系数、实际 PD 增益、anchor 与球拍位姿，帮助判断切换冲击和击球参考错误。 |
| trace.jsonl | 循环周期、调度迟到、策略/FK 耗时、上一周期日志序列化及入队耗时、动捕源时间戳和年龄；配合已有 DDS 时序定位延迟。 |
| lowstate/*.npy，默认 200Hz | 同帧 q/dq/tau_est、IMU、源 tick、驱动状态、温度、电压及最近写入命令的生成时间；保留关节顺序和采样定义的 schema.json，帮助分析快速振荡、速度反馈和执行器响应。 |
| mocap_packets.jsonl | UDP 原始球/机器人数据、源时间戳与接收时刻，帮助回放球速度估计，区分测量变化、传输延迟和滤波效果。 |
| summary.fault.json | 异常前最后日志帧、最近可用反馈、尝试发送但可能失败的目标、最后成功提交的目标和策略诊断，补足保护触发前未写入正常 trace 的一帧。 |
| summary.json | 策略/参考动作文件 SHA256、AMP 合同、运行配置、软件版本、日志入队/写入/丢弃计数及写入异常。 |

控制 trace、原始 LowState 和 UDP 包都由有界后台队列写盘；队列满时记录丢弃计数，不等待磁盘。序列化和采样复制仍消耗 CPU，不能声称日志没有开销。默认原始 LowState 文件每 10 分钟约 100MB，另有 trace 和动捕文件。`record_lowstate_hz: 0` 禁用高频采样；200Hz 足以分析已观察到的约 24Hz 分量，但仍是降采样数据，不能排除更高频运动。需要更高频率时上限为 1000Hz，应同时检查 CPU、磁盘及丢帧统计。

`tau_est` 是设备估计，不是独立力矩传感器读数；PD 力矩估算、DDS Write 返回和主机时间戳不能证明电机实际力矩或完整执行延迟。设备源 tick 保留原值，未与主机时钟做同步，不能直接相减计算硬件延迟。`writer_halted` 记录主机写入阻尼状态，不代表机器人已经停止运动。
