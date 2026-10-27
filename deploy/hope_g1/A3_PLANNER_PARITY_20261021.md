# G1 planner 与 .200 A3 的计算和物理参数统一（2026-10-21）

本轮只修改本机 Adapt-TeleAI。`.200:/home/lxz/HOPE` 仅通过 SSH 执行读取命令、
通过 SFTP 读取源码/配置，未写入文件、启动节点、运行 play、操作电机或同步部署。
用户给定 GitHub URL 当前返回404；本机已有 checkout 与 `.200` HEAD 都为
`67a1a720e985b01ec157821c9809e63bcbb13d9a`，但工作区有差异，因此以只读取得的
`.200` 实际 A3 源码及 launch 配置为准。

## 来源

- `artifacts/a3_planner_readonly_20261021/manifest.json` 保存远端路径、SHA256 和只读说明。
- `.200` 的 `hope_ws/src/hope_planner_optimized/hope_planner_optimized/` 提供 Stage 2/3。
- 实际 launch 读取 `config/hope_planner.yaml`，源目录与 install 目录内容一致。
- 本地 `configs/a3_planner.yaml` 与该 install 配置逐字节一致。
- 同时只读核对用户指定的 G1 task YAML 与 model_18000 run 的 `params/env.yaml`。
  不将 G1 的动作/观测/训练初始化参数当成修改球物理的依据。

## 改动

1. `a3_reference/constants.py`、`ball_trajectory_predictor.py`、`racket_target_planner.py`
   与远端快照逐字节一致，来源记录更新于 `a3_reference/PROVENANCE.json`。
   撤销 G1 特有的台外自由下落预测修正，保留源 A3 的原始高度夹紧行为。
2. real、MuJoCo、reference 三份 rally 配置及随机测试的默认物理统一如下：

|参数|统一值|来源|
|---|---:|---|
|重力|9.81 m/s²|A3 BallPhysics|
|球半径|0.02 m|A3 BallPhysics|
|阻力 k|0.1537|.200 A3 launch YAML|
|桌面水平速度保留比例|0.69|.200 A3 launch YAML|
|桌面竖直恢复系数|1.01|.200 A3 launch YAML，原值保留|
|球拍恢复系数|0.88|.200 A3 launch YAML|
|拍面模型半径|0.075 m|A3 PlannerConfig|
|桌面长度/宽度|2.74 / 1.525 m|A3 TableParams|
|球网高度/每侧延伸|0.1525 / 0.15 m|A3 TableParams|
|回球目标|[2.5,0,0] m|.200 A3 launch YAML|
|回球飞行时间|0.5 s|.200 A3 launch YAML|

`a3_parameters.py` 从只读配置原件及 A3 dataclass 提供默认物理。
一致性测试逐项核对三份 rally、benchmark YAML、PhysicalBall 默认值，避免配置漂移。
`A3Planner` 不再覆写 net_overhang=0，也不再将默认落点抬高一个球半径。
来球方向条件与 A3 一致为 vx<0，开始规划的 X 门槛采用实际配置2.5m。
模拟球的碰撞与网面显示同步使用15cm侧向延伸。

3. 保留 G1 必需的适配：世界坐标/时钟接口、实际球拍body/site、29关节与175维actor、
   motion库、击球平面以及既有进入/回位流程。未修改checkpoint或关节动作输出。
   `.200` A3 的 ROS发布工作区框（x=0、y=-1.5..0、z=0.2..0.6）属于机器人接球工作区，
   不属于球桌物理；G1仍由已有motion选择接口判断可执行性，不直接复制该框。
4. 独立benchmark仍直接采样command，不因这次对齐而改成按motion反推command。
   最低Z行裁剪保持；旧suite回放保留其物理参数和原先无侧向延伸的网面。
   带阻力逆积分出现发散时，仅拒绝该条来球构造候选，不更换command或依赖策略成绩筛选。

## 同输入数值对比

运行：

```bash
.venv/bin/python -m deploy.hope_g1.compare_a3_planner \
  --output deploy/recordings/hope_g1/a3_planner_parity_20261021.json
```

直接加载独立的只读源快照，64组球状态分别在HOPE桌面系与MuJoCo地面系测试，合计128次。
比较同一球状态、击球平面、目标落点、物理参数与测量时间下的四参数command。

|项|最大绝对误差|
|---|---:|
|位置|6.53e-16 m|
|球拍速度|1.04e-13 m/s|
|法向分量|9.55e-15|
|deadline / TTS|1.78e-15 s|

三个核心文件的SHA256逐字节核对通过。全部部署单元测试211 passed，18条已有资产attach警告。

## MuJoCo 运行

- `deploy/recordings/hope_g1/a3_exact_planner_20261021/summary.json`：完整来球planner链路，
  3/3碰球、1/3有效回球，3次均完成挥拍及回位，无失去站立。
- `deploy/recordings/racket_command/a3_shared_physics_20261021/report.json`：随机测试前6条，
  6/6碰球、0/6有效回球、1/6严格command达标，6/6回位。
  `frames/` 保存前两条的接触、deadline等截图。

这些是更换为同一A3参数后的新样本，不代表全区域成功率。历史训练物理30/30和旧现场
标定参数结果仅是历史证据，不再作为当前参数表现。此次未修改只看位置的motion选择器，
此前发现的方向/速度匹配不足仍需单独处理；参数一致不意味着策略一定能完成command。
