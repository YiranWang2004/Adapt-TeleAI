# HOPEPingPongMultiMotionG1 部署

本任务使用独立入口 `deploy.hope_g1.deploy_agent` 和 `HopeController`，直接加载
HOPE 的 `model_18000.pt`，不经过 mjlab Stage1 tracker 或 Stage2 actor。
原任务继续使用 `deploy.g1_pingpong.deploy_agent` 和 `deploy/configs/`。

共享部分是 G1 资产、DDS/UDP、AMP、遥控器接管、保护、记录和 MuJoCo IO。
任务配置全部位于本目录的 `configs/`；默认记录写入
`deploy/recordings/hope_g1/<env>_<时间>/`，summary 明确记录 task、controller 和 checkpoint SHA256。

## 主机分工与迁移

本机 `/home/lhy/table-tennis/Adapt-TeleAI` 负责 MuJoCo；
`.200` 的 `/home/lxz/deploy_g1_pingpong` 负责真机。

本机已迁入 `artifacts/hope_20261012_18000/`：`model_18000.pt`、
`params/env.yaml`、`params/agent.yaml`、22 个 NPZ 动作及 `hit_type.json`。
`migration_manifest.json` 保存来源和逐文件 SHA256；26 个文件均已校验。
本机 AMP 和 G1 资产也已与 `.200` 核对一致。本机测试不依赖 SSH、`.200`、ROS 或 Isaac。
大型资源位于已忽略的 `artifacts/` 目录，不纳入代码提交。

配置按主机分开：`configs/task.yaml` 使用本机仓库内资源；
`configs/task_real.yaml` 使用 `.200` 原训练目录。
入口在 check/MuJoCo 时默认前者，real 时默认后者；也可显式传 `--config`。
检查 `.200` 配置时应显式使用 `--config deploy/hope_g1/configs/task_real.yaml`。

## 本机 MuJoCo 测试

先在本机检查资源（不连接机器人）：

```zsh
cd /home/lhy/table-tennis/Adapt-TeleAI
.venv/bin/python -m deploy.hope_g1.deploy_agent --env check
```

动作库严格遵循保存配置的顺序，共 22 个动作，合法索引为 0..21；
本任务的 19 是 `0804_2_kjp-026.npz`，不能沿用旧任务的 19/24 组合。
默认 home=0（`0623_kjp-008.npz`），固定动作；检查输出包含完整索引表。

打开手动测试窗口：

场景显示蓝色球桌、球网和白色边线；桌面、球网尺寸直接读取同一份
`configs/rally.yaml` 的 ball 参数。默认视角同时覆盖机器人和球桌，
可以在 `viewer` 配置中修改 lookat、distance、elevation、azimuth。

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --render --sim-controls manual \
  --motion-index 0 --motion-select fixed --duration 600
```

在 MuJoCo 窗口依次操作（手动模式必须按 L 放下机器人）：

1. **ENTER**：START，从悬挂姿态进入 `POSITIONING`，等待 `POSE_HOLD`。
2. **L**：1.5s 平滑降低支撑，等待终端显示 `support released`。
3. **A**：切换 AMP，等待 `STABLE`。
4. **W**：相当于真机方向键 UP，上身准备后进入 `TO_POLICY` 和 `WAIT_BALL`。
5. **F**：单次发球，观察 `EXECUTE → RECOVER → WAIT_BALL`；
   **T** 可切换连续发球，默认间隔 4s。
6. **S** 返回 AMP；**B/X** 停控；**R** 重置；**Q** 退出。
   SPACE/P 暂停或继续，N 单步。

自动流程测试不需要按键；加 `--render` 可以观看同一自动流程：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --motion-index 0 --motion-select auto \
  --motion-allow all --episodes 10 --duration 70 --min-hit-rate 0.8
```

`--standing-only` 可检查 AMP → HOPE 首帧保持，不发球。
配置 `--min-hit-rate` 后，退出码 0 同时要求达到指定接触比例、完成回位且没有检测到失去站立。
summary 分别记录 `flow_passed`、`hit_rate`、`simulated_hits`、`simulated_balls`；
未配置命中率门槛时仍只验收流程。物理阈值沿用共享仿真的观察模式，真机保留停控保护。
记录默认在 `deploy/recordings/hope_g1/mujoco_<时间>/`。

## .200 真机测试

在 `.200` 上操作。2026-10-19 本次只读检查时，`eno1` 是 NO-CARRIER 且没有 IPv4，
DDS 报 `eno1: does not match an available interface`，未创建电机控制发布器。
应先开启机器人并接通 `eno1` 网线，再检查机器人网段。
已确认 `.200` 保存了 `g1-direct-eno1` 配置，地址为 `192.168.123.210/24`。
接线后若未自动恢复地址，由操作者执行：

```zsh
sudo nmcli connection up g1-direct-eno1
ip -brief address show eno1
ping -I eno1 -c 3 192.168.123.164
```

不要把当前动捕网络的 USB 网卡 `enx98fc84e70fe2` 当作机器人网口。
本次没有修改任何网络配置。

再检查当前 ROS 节点；已有的进程保持运行，不要重复启动：

```zsh
source /opt/ros/humble/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 node list
ros2 topic info /P1/pose
ros2 topic echo /P1/pose --once --qos-reliability best_effort
```

完整动捕链路需要以下三个常驻进程。仅在对应进程未运行时启动。

终端 1：VRPN 客户端。

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd /home/lxz/HOPE/hope_ws
ros2 launch ./src/vrpn_mocap/vrpn_mocap/launch/client.launch.yaml \
  server:=192.168.1.101 port:=3883 \
  allowed_topics:="/vrpn_mocap/MCAvatar/pose4,/vrpn_mocap/MCAvatar/pose30310"
```

终端 2：转换器，将原始机器人/球数据发布为 `/P1/pose`、`/poses`。

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch vrpn_to_pose vrpn_to_pose.launch.py \
  robot_side:=P2 \
  robot_input_topic:=/vrpn_mocap/MCAvatar/pose4 \
  ball_input_topic:=/vrpn_mocap/MCAvatar/pose30310
```

终端 3：ROS→UDP 桥接。

```zsh
source /opt/ros/humble/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd /home/lxz/deploy_g1_pingpong
/usr/bin/python3 -m deploy.io.mocap bridge \
  --config deploy/hope_g1/configs/mocap.yaml
```

本任务的 `configs/mocap.yaml` 与已标定的 live 链路保持一致：
`/P1/pose`（map）、`/poses`（world）→ `127.0.0.1:15050`。
旧任务的 bridge 可以直接复用。

终端 4：先检查 `.200` 的策略配置：

```zsh
cd /home/lxz/deploy_g1_pingpong
.venv/bin/python -m deploy.hope_g1.deploy_agent --env check \
  --config deploy/hope_g1/configs/task_real.yaml
```

不加 `--enable-motors` 时仅接收/记录：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real --network-interface eno1 --duration 10
```

确认输出 `root_valid=true`、`ball_valid=true`，没有时间戳或坐标系错误。
等待只读进程结束后，再启动电机控制。

真机控制命令（由操作者手动执行）：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real --network-interface eno1 \
  --motion-index 0 --motion-select fixed \
  --duration 600 --enable-motors
```

依次按 START，等待 `POSE_HOLD`；按 A，等待 `STABLE`；确认机器人站稳后按 UP，
经过 `POLICY_POSTURE → TO_POLICY → WAIT_BALL` 后再送球。
挥拍后自动回位，等待下一球；DOWN 返回 AMP；SELECT/B/X 停控。
真机不使用仿真窗口的 L 键。与旧任务电机控制入口、UDP monitor 不能同时运行。
自动选动作可用 `--motion-select auto --motion-allow all` 或本任务的有效索引列表。
真机记录默认在 `deploy/recordings/hope_g1/real_<时间>/`。

## 推理和运行约定

- 175 维输入、29 维动作，ELU MLP，50 Hz，归一化为
  `(obs - saved_mean) / (saved_std + 0.01)`；独立读取 actor，忽略 critic。
- 顺序、默认姿态、默认关节速度、动作 scale、动作文件及回位 75 帧来自保存配置。
  执行 PD 沿用公共显式增益表；加载时核对其与训练 PD 一致，允许独立配置中的显式 scale。
- 动捕位置差分估计机身世界线速度，再旋转到 pelvis 系；首次为零。
  这是真机状态估计，不等同于 Isaac 提供的无噪声速度。
- 关节参考包含位置和速度；anchor 使用完整相对旋转；机身目标和拍面位置使用 yaw-heading 系，
  期望拍面速度/法向在物理世界中用于规划，送入 actor 前逆变换回训练 world；
  击球后的剩余时间保留负值。整个动作库共享进入策略时固定的一个 world 变换。
- 来球预测决定动作和接触点；期望拍面速度沿所选动作击球时的参考法向，
  正反手符号按训练配置，速度为 3m/s。目标在开始挥拍时锁定，后续丢球不更换 actor。
- 部署采用固定 world home、等球时冻结首帧、挥拍后保持 home 1.5s 再接球。
  这是部署调度；训练中的随机换动作和目标扰动不会在部署时随机执行。
  raw actor action 连续保存，AMP 退出/重新进入时清零。

当前验证覆盖 CPU 推理、观测接口和 MuJoCo 接球。完整核对与修复记录见
[TRAIN_PLAY_AUDIT.md](TRAIN_PLAY_AUDIT.md)。真机稳定性与击球效果需另行验收。

## 2026-10-19 验证记录

以下较早记录的 0 次接触已复现并修复：参考 world/绝对向量、动作库变换和机身目标耦合
完成修正后，三个种子合计 30/30 次模拟碰撞命中；名义跟踪组合阈值为 22/22 通过。
后续接球验收使用上面的命中率门槛，详细证据见核对报告。

- 持久迁移后在本机重新测试：默认配置直接加载仓库内 checkpoint 和动作，
  自动选中动作 8、17、13，三次挥拍/回位全部完成；最大倾角 12.62°，
  最低 pelvis 高度 0.750m，没有检测到失去站立，接触次数为 0。
  记录在本机 `deploy/recordings/hope_g1/local_migrated_validation_20261019/`。
- 本机图形窗口启动正常，本次窗口流程在 POLICY_POSTURE 时 HALT 结束；
  完整流程结论来自上述无窗口自动测试，图形交互流程由操作者按文档执行。
- `.200` 的策略配置检查通过；本次只读 DDS 检查受 `eno1` 无 carrier/IP 阻塞，
  没有进行电机控制或修改网络。记录目录为
  `deploy/recordings/hope_g1/real_readonly_20261019_migration/`。
- 本地部署测试：147 项通过；`.200` 的新任务及共享运行时测试：17 项通过。
- 与 `.200` 原 HOPE 环境的 `ActorCritic` + `EmpiricalNormalization` 对照
  990 个输入（含仿真实际观测），最大动作绝对误差为 0。
- `.200` 自动选择模式完成动作 21、19、10 的三次挥拍及回位；
  最大倾角 15.45°，最低 pelvis 高度 0.734m，没有检测到失去站立。
  本次模拟接触次数为 0，因此此结果只证明部署流程与站立保持，不能作为击球效果验收。
- 目标机记录：`deploy/recordings/hope_g1/validation_20261019_auto/`，
  包含 `summary.json`、`trace.jsonl` 和 `policy_parity.json`。
- 验证没有启动真机电机输出，也没有改动原训练目录或 checkpoint。
