# G1 乒乓球基础训练流程

项目：`/home/lhy/table-tennis/Adapt-TeleAI`。使用 TeleGMR 数据和当前 `.venv`，每条训练命令独立执行，不需要预先定义变量。

## 1. Stage1：训练动作跟踪器

```zsh
cd /home/lhy/table-tennis/Adapt-TeleAI && \
.venv/bin/python -m mjlab.scripts.train Mjlab-PingPong-Tracking-G1-TeleGMR-Stage1-RandomDt \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --agent.save-interval 2000 \
  --agent.seed 42 \
  --agent.resume False \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.run-name g1_tele_gmr_stage1_30k_seed42_waistrate004
```

产物位于 `logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/<时间>_<run-name>/`：

- `model_N.pt`：训练 checkpoint。
- `jit/modeljit_N.pt`：冻结跟踪器，供 Stage2 使用。

Stage1 学习 29 个关节的动作跟踪。

## 2. Stage2：冻结跟踪器，训练接球策略

**本次完全重训：先训练新 Stage1，再用新 tracker 从头训练 Stage2。** 下方路径是旧 GMR 基线示例；本次运行 Stage2 前，必须将 `--env.actions.joint-pos.tracker-file` 替换成刚完成的新 Stage1 的 `jit/modeljit_N.pt`，不要继续使用下面的旧 tracker。

```zsh
cd /home/lhy/table-tennis/Adapt-TeleAI && \
.venv/bin/python -m mjlab.scripts.train Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --env.actions.joint-pos.tracker-file /home/lhy/table-tennis/Adapt-TeleAI/logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-04_04-32-34_g1_tele_gmr_29dof_30k_save2000_seed42/jit/modeljit_29999.pt \
  --agent.max-iterations 30000 \
  --agent.save-interval 2000 \
  --agent.seed 42 \
  --agent.resume False \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.experiment-name g1_pingpong_receive_stage2 \
  --agent.run-name g1_tele_gmr_stage2_30k_seed42_waistrate004
```

本次 waist004 产物位于普通 Stage2 分组 `logs/rsl_rl/g1_pingpong_receive_stage2/<时间>_<run-name>/`；原 TeleGMR 和 Hard 目录不写入新实验。保留 `model_N.pt`、配套 `tracker.pt` 和 `params/`。

Stage2 只学习关节残差和参考动作速度，**不会更新 Stage1**。如果重训了 Stage1，应使用新跟踪器重新训练匹配的 Stage2。

两条命令中的基本参数：

| 参数 | 含义 |
| --- | --- |
| `--gpu-ids '[0]'` | 使用 GPU 0；当前解析器要求带引号的列表写法 |
| `--env.scene.num-envs 4096` | 并行环境数，显存不足可改为 2048 或 1024 |
| `--agent.max-iterations 30000` | 训练 30000 次 PPO 更新 |
| `--agent.save-interval 2000` | 每 2000 次更新保存模型 |
| `--agent.seed 42` | 随机种子，对照实验保持一致 |
| `--agent.run-name` | 本次实验名称，建议写清奖励修改 |

## 3. Action-rate 奖励在哪里修改

### 奖励函数定义

[src/mjlab/envs/mdp/rewards.py](src/mjlab/envs/mdp/rewards.py:58) 的 `action_rate_l2()`：

```text
penalty = sum((本周期原始动作 - 上周期原始动作)²)
```

该函数支持 `exclude_joint_names`，可以排除非腰部关节，只计算腰部三轴。

### 当前在哪里使用

[src/mjlab/tasks/a3_pingpong/config.py](src/mjlab/tasks/a3_pingpong/config.py:143) 中：

```python
"action_rate": RewardTermCfg(func=common.action_rate_l2, weight=-0.02),
```

G1 的 Stage1、Stage2 都复用这份基础配置。直接将这里的 `-0.02` 改为 `-0.05`，会增大全身动作变化惩罚，也会影响共用该配置的 A3 任务，**不是只改腰部**。

### 单独调整腰部惩罚（已加入）

修改 [src/mjlab/tasks/g1_pingpong/config.py](src/mjlab/tasks/g1_pingpong/config.py) 顶部这两个参数：

```python
WAIST_ACTION_RATE_WEIGHT_STAGE1 = -0.04
WAIST_ACTION_RATE_WEIGHT_STAGE2 = -0.04
```

- 第一个控制 Stage1；第二个控制 Stage2，两阶段可分别调整。
- `0.0`：关闭额外腰部项；`-0.04`：当前值。绝对值越大惩罚越强，保持非正值。
- 原全身 action-rate 仍为 `-0.02`。当前额外腰部项为 `-0.04`，所以腰部原始动作合计权重为 `-0.06`，其他关节保持 `-0.02`。

**具体接入位置：**同一文件的 `_specialize_robot()` 中注册 `cfg.rewards["waist_action_rate"]`，复用现有 `action_rate_l2()`，排除非腰部关节，只计算 `waist_yaw_joint`、`waist_roll_joint`、`waist_pitch_joint`。Stage1、Stage2 的 G1 配置都经过这个函数，因此都会加入；A3 配置不受影响。

Stage1 惩罚基础关节动作变化；Stage2 惩罚腰部残差变化，不包含速度动作，也不等于惩罚最终关节目标。

修改权重后重新启动训练。原 checkpoint 不会自动改变；若重训 Stage1，应使用其新 tracker 训练匹配的 Stage2。本次没有启动训练或修改部署控制参数。

## 4. 与原 GMR 基线对齐检查（2026-10-16）

已与原 TeleGMR Stage1/Stage2 的保存参数比较。按上面命令设置 4096 环境、30000 次更新、2000 保存间隔、seed=42、resume=False 后，排除路径和配置格式差异，训练配置的有效改动仅为新增腰部项 -0.04。其他奖励、网络、PPO、执行器、参考时间与来球分布保持基线设置，不需要切换 Hard 或额外修改 PD。

Stage2 必须使用本次新 Stage1 的产物；训练参数检查不等于新模型已经训练完成。本轮只检查配置，没有启动训练。详细对比保存在 `logs/training_config_audit_20261016/summary.json`。
