# 本机 G1 部署按 A3 对齐后的结果（2026-10-20）

本轮只修改本机 `/home/lhy/table-tennis/Adapt-TeleAI`，没有连接或写入 `.200`。
以后该主机的部署工作范围仅为 `/home/lxz/deploy_g1_pingpong`；不修改 `/home/lxz/HOPE`。
`task_real.yaml` 已在本机改为部署目录内的相对资产路径，但本轮尚未同步远端。

进入策略流程保持为：

`WAIT → POSITIONING → POSE_HOLD → TO_STABLE → STABLE → POLICY_POSTURE → TO_POLICY → ARMED`

START / A / UP、AMP、上身准备、初次策略交接与停控流程保留。MuJoCo 手动模式仍需 L 放下支撑。
三个完整测试的实际阶段记录均验证该顺序。

## 已对标并修正

| 部分 | 当前实现 |
| --- | --- |
| 完整目标命令 | 本地使用仓库 A3 optimized planner 的 Stage 2/3，产生接触点、物理拍速、击球面法向、deadline；不再仅提供接触点/时刻 |
| 默认目标模式 | 对齐 README-TeleAI 的 `--no-target-clamp`：raw 命令通过 actor 输入，不替换为固定 3m/s；可显式选 envelope 模式作对照 |
| 正反手与坐标 | 物理拍速不因反手翻转；反手只将 striking-face normal 映射为拍面本地 +Y；绝对向量在 actor 入口逆旋转一次回训练 world |
| base target | 按选定参考动作的 pelvis-to-racket reach，将接触点与 XY 机身目标耦合；idle/return 同样使用 home strike-frame reach |
| 动作选择 | 按 A3 对候选动作逐个做当前 torso 的纯 yaw rebase、按参考接触点距离选取，使用 5cm hysteresis；保留 G1 的 22 动作及训练顺序 |
| home / 回位 | home 变换在策略入口冻结；挥拍结束立即切回固定 home 的 frame 0，连续保持 75 个控制周期；不重置机器人、不恢复默认关节姿态、不再交接一次 AMP |
| 触发时序 | 对齐 A3 的 25 tick settle、最大命令 TTS=0.5s、deadline 减当前时间、半远离零舍入；G1 最晚起始帧仍为训练的 29，击球帧 43 |
| 球估计 | 二次时间拟合、至少 6 样本、约 90ms 时间窗；反弹时清历史，避免拟合跨速度不连续点。按 G1 的球心反弹高度配置检测 |
| 机身状态 | 按源样本做 pose EMA / 跳变处理；按源时间戳差分、速度限幅和 EMA；位置与速度从同一源样本取，避免回调竞态或网络突发误差 |
| 碰撞模型 | 从迁移的 G1 fast URDF 重建 31 个碰撞几何，处理固定 link 变换，启用训练中的自碰撞；不再继承旧任务的脚底球/禁用自碰撞配置 |
| 记录 | 保存实际 resolved rally 配置、raw planner 命令、actor 实际目标、raw action、joint target、逐球出球速度、过网点及首次落点 |

A3 源码副本、Apache-2.0 LICENSE、源 commit 与 SHA256 位于 `a3_reference/`。
目标映射不是只与另一个 Python 实现自比：`target_golden.cpp` 从实际 A3 C++ helper 提取，
编译生成 raw/envelope、正/反手、不同 yaw/尺度的 golden；Python 映射逐项对拍通过。

## 必要的 G1 适配与物理修正

- 不复制 A3 的 31-DoF、185-D/其他 HITTER 契约。G1 保留实际 checkpoint 的
  29-DoF、175-D、动作 scale、归一化、PD 和 50Hz 控制。
- planner 内部使用 A3 的 near-left/table-surface 坐标，G1 本地 court 是
  centered-Y/floor-Z；明确使用 `[near_x, width/2, table_height]` 变换，球心落台高度含球半径。
  没有把该局部变换擅自加到现场动捕标定。
- 本机模型球参数用于 planner，不抄 A3 的现场 drag/restitution 标定常数。
- A3 源 predictor 会将台外下落的球夹到桌面高度。G1 有低于桌边的接触动作，
  因此修正为只有从桌面上方向下穿过有限桌面时反弹，台外继续自由下落。
- 碰撞几何仍复用已验证的本地关节、惯量和视觉资产，新增 collision 几何不参与质量推断。
  compiled scene 加载时检查碰撞掩码，防止几何被 Entity 的匹配配置静默禁用。
- 这些适配不代表 MuJoCo 与 PhysX/真机动力学相同；地形/随机化和现场时延仍需独立验证。

## 最终 30 球验收

同一模型、配置和发球方法，seed=42、7、123 各 10 球；球持续按 4s 间隔进入，
不根据回球好坏筛选样本。发球只使用冻结的参考动作击球平面和来球合法性，
不使用实际出球质量生成下一球。

| seed | 发球 | 碰拍 | 越网后首次落对方台面 | 完成挥拍/回位 | 验收 |
| --- | --- | --- | --- | --- | --- |
| 42 | 10 | 10 | 10 | 10/10 | 通过 |
| 7 | 10 | 10 | 10 | 10/10 | 通过 |
| 123 | 10 | 10 | 10 | 10/10 | 通过 |
| 合计 | 30 | 30 | 30 | 30/30 | 通过 |

覆盖 16 个发球参考动作、10 个实际选中动作，不代表全部 22 个动作或任意来球都已验收。
球心首次桌面高度交点的 x 范围约 [2.546,3.378]m，y 范围 [-0.382,0.284]m；
对方半台为 x∈(2.02,3.39]、|y|≤0.7625。最小球体网顶净余量约 3.66cm。
三个种子均未检测到失去站立，最大倾角约 6.30°。

验收要求按顺序满足碰拍、球体越过网顶、**第一次**落在对方台面，不会把己方先弹台
之后的过网误算为成功。球与拍面的碰撞仍由实际 MuJoCo 机器人拍面位置、法向及速度计算。

### 没有修改策略输出或球速

- checkpoint SHA256 仍为 `e22854c523cee8fe8933c198c8885e2216787920d2a89e9da1c6882cc150ffb3`。
- 检查了 2258 个实际 SWING tick：下发 target 与 `default + action_scale * raw_actor_action`
  的最大差值为 **0**；没有替代 IK、额外平衡策略或挥拍期动作 blending。
- raw planner 物理拍速与 actor 收到的 deploy-world 速度最大差值为 **0**；
  法向按正反手映射的误差为浮点舍入量级。
- 未覆盖反弹球速、放大拍面、瞬移机器人或球。每轮只有正常的初始模型 reset。
- 初期三球结果和修改前的 0/30 是不同实现/模型条件，不能当作只改变一个变量的因果实验。
  上表只报告最终完整配置的独立测试结果。

## 自动化与证据

- 完整回归：**175 passed**（18 条已有资产 attach 警告），Ruff 检查通过。
- 原始 Isaac 440 样本的观测/actor 对拍复测仍通过：最大观测误差约 3.77e-6，
  actor 最大误差约 7.42e-6。该诊断使用实际采样零位和瞬时速度，不能冒充生产动捕对拍。
- `deploy/recordings/hope_g1/` 下的 `a3_aligned_validation_seed42/`、
  `a3_aligned_validation_seed7/`、`a3_aligned_validation_seed123/` 是三个完整运行目录。
- `deploy/recordings/hope_g1/a3_aligned_validation_20261020/report.json`：逐球结果。
- 同目录 `contract_checks.json`：进入流程、动作无改写、目标映射、碰撞掩码/资产检查。
- 同目录 `isaac_contract.json`：实际 Isaac 数值对拍；`return_trajectories.png`：出球轨迹。

复现自动可视化测试：

```bash
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --render --sim-controls auto \
  --motion-select auto --motion-allow all \
  --target-source a3_planner --a3-target-mode raw \
  --seed 42 --episodes 10 --duration 70 --min-return-rate 0.8
```

手动模式仍按原流程 ENTER → L → A → W，进入 WAIT_BALL 后按 F 发一球、T 连续发球。
本轮没有真机测试或远端同步；上述结果仅限本机模拟来球分布。
