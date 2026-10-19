# G1 基础 play 与 MuJoCo 部署测试

本文记录训练侧 `play` 和部署侧 MuJoCo 的基础命令。真机启动流程见
[REAL_DEPLOYMENT.md](REAL_DEPLOYMENT.md)，MuJoCo 参数及交互细节见
[MUJOCO_TESTING.md](MUJOCO_TESTING.md)。

所有命令在项目根目录执行，使用 zsh；本地目录为：

```zsh
cd /home/lhy/table-tennis/Adapt-TeleAI
```

以下统一使用 Tele-GMR Stage2 的 `model_29999.pt`。`play` 需要完整训练工程、
对应依赖和 CUDA 环境；部署侧使用 `deploy/configs/g1_tele_gmr.yaml` 加载同一
checkpoint。更换模型时，分别修改 `--checkpoint-file` 或部署配置中的
`checkpoint`，并确认配套 `tracker.pt`、训练配置和动作库匹配。

## 1. 基本 play：观察训练环境中的策略

```zsh
.venv/bin/python -m mjlab.scripts.play \
  Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive \
  --checkpoint-file logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr/2026-10-04_22-19-39_g1_tele_gmr_stage2_30k_4096_seed42/model_29999.pt \
  --num-envs 1 \
  --viewer native \
  --device cuda:0 \
  --debug-vis False
```

该命令按任务的普通 play 流程运行，不包含部署中的 START → AMP → 乒乓球
接管流程。`--debug-vis False` 关闭调试绘制，`--viewer native` 打开本地窗口。
关闭窗口结束测试。

## 2. play：准备 → 动作 19 → 回位完整测试

```zsh
.venv/bin/python -m mjlab.scripts.play \
  Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive \
  --checkpoint-file logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr/2026-10-04_22-19-39_g1_tele_gmr_stage2_30k_4096_seed42/model_29999.pt \
  --standing-cycle True \
  --cycle-home 19 \
  --cycle-motions 19 \
  --cycle-seed 42 \
  --viewer native \
  --device cuda:0 \
  --debug-vis False \
  --cycle-output "logs/play_cycle/tele_19_$(date +%Y%m%d_%H%M%S)"
```

- `--cycle-home 19`：准备和回位使用动作 19 的首帧参考。
- `--cycle-motions 19`：测试动作 19；可改为 `19,24` 或 `all`。
- `--cycle-seed 42`：来球等随机过程的种子。
- `--cycle-repeats 3`：可选，连续做三组独立测试，种子依次为 42、43、44。
- `--cycle-output`：保存本次循环测试结果。

该模式用于训练侧完整流程检查，仍不等同于部署侧的 AMP 接管测试。

## 3. deploy MuJoCo：手动启动与发球

仅需一个本地终端；使用模拟球，不需要启动 ROS、动捕或 G1 DDS。

```zsh
.venv/bin/python -m deploy.g1_pingpong.deploy_agent \
  --env mujoco \
  --sim-mode full \
  --sim-controls manual \
  --render \
  --config deploy/configs/g1_tele_gmr.yaml \
  --real-config deploy/configs/real.yaml \
  --mocap-config deploy/configs/mocap_g1_live.yaml \
  --rally-config deploy/configs/mujoco.yaml \
  --ball-source sim \
  --motion-index 19 \
  --motion-select auto \
  --motion-allow 19,24 \
  --seed 42 \
  --sim-feed-interval 2.0 \
  --duration 600 \
  --output "deploy/recordings/mujoco_manual_$(date +%Y%m%d_%H%M%S)"
```

将焦点放在 MuJoCo 窗口，按顺序操作：

**Enter → 等待 POSE_HOLD → L 放下 → 等待落地 → A → 等待 STABLE → W → 等待 WAIT_BALL。**

| 按键 | 功能 |
| --- | --- |
| F | 立即发一球 |
| T | 开关连续发球，默认间隔 2 秒 |
| S | 返回 AMP 稳定策略 |
| B / X | 停止策略，物理在阻尼下继续 |
| R | 完整重置；重新执行启动顺序 |
| 空格 / P | 暂停／继续仿真物理和策略 |
| N | 暂停时前进一个策略周期 |
| Q | 退出 |

手动模式默认不自动发球，先观察切换和等待，再按 F 或 T。

## 4. deploy MuJoCo：自动完整测试

自动执行摆姿、放下、AMP 和乒乓球策略接管，之后按固定周期发球。

```zsh
.venv/bin/python -m deploy.g1_pingpong.deploy_agent \
  --env mujoco \
  --sim-mode full \
  --sim-controls auto \
  --render \
  --config deploy/configs/g1_tele_gmr.yaml \
  --real-config deploy/configs/real.yaml \
  --mocap-config deploy/configs/mocap_g1_live.yaml \
  --rally-config deploy/configs/mujoco.yaml \
  --ball-source sim \
  --motion-index 19 \
  --motion-select auto \
  --motion-allow 19,24 \
  --seed 42 \
  --sim-feed-interval 2.0 \
  --episodes 3 \
  --final-hold-seconds 5 \
  --duration 120 \
  --output "deploy/recordings/mujoco_auto_$(date +%Y%m%d_%H%M%S)"
```

不需要窗口时去掉 `--render`，可加速运行。`--episodes 3` 是目标挥拍／回位
次数，不是只发三球；连续发球不等待机器人回位。完成三次回位后再保持
5 秒停止，或先达到 120 秒时长上限。

## 常用修改

| 目的 | 修改方式 |
| --- | --- |
| 固定使用动作 19 | `--motion-select fixed`，保留 `--motion-index 19` |
| 只观察站立／切换 | `--sim-mode standing`，保持 `--ball-source sim` |
| 首球延迟 5 秒 | 自动测试中增加 `--sim-policy-wait 5`，只影响首球 |
| 连续发球每秒一球 | `--sim-feed-interval 1.0` |
| 增加物理步长对照 | 默认 `--sim-physics-dt 0.001`；策略周期仍为 20ms |
| 延长观察时间 | 增大 `--duration`；手动模式不受 `--episodes` 数量限制 |

`--duration 600` 是最多 600 秒仿真时间，即 10 分钟；不渲染时可能比现实
时间快，暂停时不消耗仿真时间。手动窗口在控制到时后可继续展示阻尼物理，
此时策略已停止，可按 R 重试或 Q 退出。

部署侧执行 PD 来自 `deploy/g1_pingpong/pd_gains.py`，倍率来自
`deploy/configs/real.yaml` 的 `policy_pd` 和 `stable_policy.pd`。训练侧 play
使用训练环境的执行器配置，因此修改部署 PD 不会同步修改 play。
`real.yaml` 中的 `stop_policy: hope_67a1a72` 仅对真机入口生效。

部署日志保存在 `--output` 目录的 `trace.jsonl`、`summary.json`；手动 R 后的
记录放入 `reset_001/` 等子目录。MuJoCo 完整测试的物理超限会记录在日志中，
不因倾角／实测关节阈值自动重置；查看结果时同时检查失稳记录和通过状态。
