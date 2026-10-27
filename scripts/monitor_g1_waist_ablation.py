"""Read-only matched-iteration TensorBoard monitor for the 50k ablation."""

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Loss/value",
  "Loss/learning_rate",
  "Episode_Reward/motion_anchor_pos",
  "Episode_Reward/motion_anchor_ori",
  "Episode_Reward/motion_body_pos",
  "Episode_Reward/motion_body_ori",
  "Episode_Reward/motion_wrist",
  "Episode_Reward/motion_hit_arm",
  "Episode_Reward/action_rate",
  "Episode_Reward/waist_action_rate",
  "Metrics/motion/error_anchor_pos",
  "Metrics/motion/error_anchor_rot",
  "Metrics/motion/error_joint_pos",
  "Metrics/motion/error_joint_vel",
  "Metrics/motion/completion_fraction",
  "Metrics/motion/reference_dt",
  "Episode_Termination/anchor_height",
  "Episode_Termination/anchor_orientation",
  "Episode_Termination/feet_height",
]


def window(acc, start, end):
  result = {}
  for tag in TAGS:
    if tag not in acc.Tags()["scalars"]:
      continue
    events = [e.value for e in acc.Scalars(tag) if start <= e.step <= end]
    finite = [v for v in events if math.isfinite(v)]
    result[tag] = dict(
      count=len(events),
      mean=statistics.mean(finite) if finite else None,
      nonfinite=len(events) - len(finite),
    )
  return result


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--pipeline", required=True, type=Path)
  p.add_argument("--interval", type=float, default=300)
  args = p.parse_args()
  folder = args.pipeline / "effect_monitor"
  folder.mkdir(exist_ok=True)
  accumulators = {}
  while True:
    try:
      status = json.loads((args.pipeline / "status.json").read_text())
      available = {}
      for key, job in status["jobs"].items():
        if "run_dir" not in job:
          continue
        run = Path(job["run_dir"])
        if not list(run.glob("events.out.tfevents.*")):
          continue
        if key not in accumulators:
          accumulators[key] = EventAccumulator(str(run), size_guidance={"scalars": 0})
        acc = accumulators[key].Reload()
        if "Train/mean_reward" in acc.Tags()["scalars"] and acc.Scalars(
          "Train/mean_reward"
        ):
          available[key] = max(e.step for e in acc.Scalars("Train/mean_reward"))
      report = dict(
        updated_at=datetime.now().isoformat(),
        pipeline_state=status["state"],
        latest_iterations=available,
        matched_stage_windows={},
        interpretation="Weighted reward compositions differ; total return alone is not a ranking. Waist rate is not measured motor oscillation. Termination scalars are not per-episode failure probabilities.",
      )
      for stage in [1, 2]:
        keys = [key for key, job in status["jobs"].items() if job["stage"] == stage]
        ready = [key for key in keys if key in available]
        if not ready:
          continue
        end = min(available[key] for key in ready)
        start = max(0, end - 499)
        report["matched_stage_windows"][str(stage)] = dict(
          start=start,
          end=end,
          all_variants_available=len(ready) == len(keys),
          missing_variants=[key for key in keys if key not in ready],
          variants={
            key: dict(
              waist_weight=status["jobs"][key]["waist"],
              tracker_job=status["jobs"][key].get("tracker_job"),
              metrics=window(accumulators[key], start, end),
            )
            for key in ready
          },
        )
      tmp = folder / "latest.json.tmp"
      tmp.write_text(json.dumps(report, indent=2))
      tmp.replace(folder / "latest.json")
      with (folder / "history.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
      print(f"{report['updated_at']} {status['state']} {available}", flush=True)
      if status["state"] in ("complete", "failed", "interrupted"):
        return
    except Exception as exc:
      print(
        f"Monitor error (training is not modified): {type(exc).__name__}: {exc}",
        flush=True,
      )
    time.sleep(args.interval)


if __name__ == "__main__":
  main()
