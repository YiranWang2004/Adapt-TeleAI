"""Read-only TensorBoard comparisons alongside the independent training monitor."""

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
  "1": ROOT
  / "logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-04_04-32-34_g1_tele_gmr_29dof_30k_save2000_seed42",
  "2": ROOT
  / "logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr/2026-10-04_22-19-39_g1_tele_gmr_stage2_30k_4096_seed42",
}
TAGS = {
  "Episode_Reward/waist_action_rate": "腰部额外奖励（已加权）",
  "Episode_Reward/action_rate": "全身 action-rate 奖励（已加权）",
  "Metrics/motion/error_anchor_pos": "Anchor 位置误差",
  "Metrics/motion/error_anchor_rot": "Anchor 旋转误差",
  "Metrics/motion/error_joint_pos": "关节位置误差",
  "Metrics/motion/error_joint_vel": "关节速度跟踪误差",
  "Metrics/motion/completion_fraction": "采样片段完成度",
  "Episode_Termination/anchor_height": "高度失败终止统计",
  "Episode_Termination/anchor_orientation": "朝向失败终止统计",
  "Episode_Termination/feet_height": "脚部高度失败终止统计",
  "Train/mean_reward": "平均回报（奖励组成不同）",
}


def window_mean(events, start, end):
  values = [event.value for event in events if start <= event.step <= end]
  if not values:
    return {"count": 0, "mean": None, "nonfinite": 0}
  finite = [value for value in values if math.isfinite(value)]
  return {
    "count": len(values),
    "mean": statistics.mean(finite) if finite else None,
    "nonfinite": len(values) - len(finite),
  }


def write_report(output, report):
  temp = output / "latest.json.tmp"
  temp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
  temp.replace(output / "latest.json")
  lines = [
    "# Waist004 训练影响监测",
    "",
    f"更新：{report['updated_at']}；Stage{report['stage']}；训练状态：{report['training_state']}。",
    f"最近已写入 TensorBoard 的迭代：{report['iteration']}；比较区间：{report['window_start']}–{report['iteration']}。",
    f"当前 run：`{report['run_dir']}`。",
    "",
    "| 指标 | 当前窗口均值 | 原 GMR 同迭代窗口均值 |",
    "| --- | ---: | ---: |",
  ]

  def number(value):
    return "未记录" if value is None else f"{value:.6g}"

  for tag, data in report["metrics"].items():
    lines.append(
      f"| {TAGS[tag]} | {number(data['current']['mean'])} | {number(data['baseline']['mean'])} |"
    )
  lines += [
    "",
    f"已保存 checkpoint 编号：{report['saved_checkpoints']}。",
    f"非有限数值样本数：{report['nonfinite_samples']}。",
    "",
    "说明：历史基线没有单独记录腰部 action-rate，不能从全身项推算腰部改善幅度。",
    "腰部项是带权重和回合归一化的训练统计，不是关节速度 RMS，也不等于实际抖动。",
    "关节速度误差是相对参考的误差，不是单独腰部速度。终止项统计不是单回合失败概率；片段完成度不是从首帧完整接球成功率。",
    "同迭代历史对比是观察性参考，不是严格配对 A/B；当前与基线奖励组成不同，不能只凭总回报判断优劣。",
    "训练早期结论需等待后续 checkpoint 的固定条件物理评估；本监测不调整奖励、训练参数或部署参数。",
  ]
  temp = output / "latest.md.tmp"
  temp.write_text("\n".join(lines) + "\n")
  temp.replace(output / "latest.md")
  with (output / "history.jsonl").open("a") as stream:
    stream.write(json.dumps(report, ensure_ascii=False) + "\n")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pipeline", type=Path, required=True)
  parser.add_argument("--interval", type=float, default=300)
  parser.add_argument("--once", action="store_true")
  args = parser.parse_args()
  output = args.pipeline / "waist_effect"
  output.mkdir(exist_ok=True)
  baselines, runs = {}, {}
  while True:
    try:
      status = json.loads((args.pipeline / "status.json").read_text())
      available = [k for k, v in status["stages"].items() if "run_dir" in v]
      if available:
        stage = max(available, key=int)
        run = Path(status["stages"][stage]["run_dir"])
        if stage not in baselines:
          baselines[stage] = EventAccumulator(
            str(BASELINES[stage]), size_guidance={"scalars": 0}
          ).Reload()
        if str(run) not in runs:
          runs[str(run)] = EventAccumulator(str(run), size_guidance={"scalars": 0})
        current = runs[str(run)].Reload()
        baseline = baselines[stage]
        current_tags = current.Tags()["scalars"]
        baseline_tags = baseline.Tags()["scalars"]
        if "Train/mean_reward" in current_tags:
          iteration = max(e.step for e in current.Scalars("Train/mean_reward"))
          start = max(0, iteration - 199)
          metrics = {}
          for tag in TAGS:
            if tag not in current_tags:
              continue
            metrics[tag] = {
              "current": window_mean(current.Scalars(tag), start, iteration),
              "previous_current_window": window_mean(
                current.Scalars(tag), max(0, start - 200), start - 1
              ),
              "baseline": window_mean(
                baseline.Scalars(tag) if tag in baseline_tags else [], start, iteration
              ),
            }
          checkpoints = sorted(
            int(p.stem.split("_")[-1]) for p in run.glob("model_*.pt")
          )
          report = {
            "updated_at": datetime.now().isoformat(),
            "stage": stage,
            "training_state": status["state"],
            "iteration": iteration,
            "window_start": start,
            "run_dir": str(run),
            "baseline_run": str(BASELINES[stage]),
            "metrics": metrics,
            "saved_checkpoints": checkpoints,
            "nonfinite_samples": sum(
              value["current"]["nonfinite"] for value in metrics.values()
            ),
          }
          write_report(output, report)
          print(
            f"{report['updated_at']} Stage{stage} iteration={iteration}, "
            f"nonfinite={report['nonfinite_samples']}; report={output / 'latest.md'}",
            flush=True,
          )
      if args.once or status["state"] in ("complete", "failed", "interrupted"):
        return
    except Exception as exc:
      (output / "monitor_error.json").write_text(
        json.dumps({"time": datetime.now().isoformat(), "error": str(exc)}, indent=2)
      )
      print(f"Effect monitor error: {exc}", flush=True)
      if args.once:
        raise
    time.sleep(args.interval)


if __name__ == "__main__":
  main()
