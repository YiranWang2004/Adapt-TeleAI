"""Evaluate each fixed home/swing motion with the full rally controller."""

import argparse
import contextlib
import csv
import json
from pathlib import Path

import torch

from deploy.g1_pingpong.configuration import add_policy_arguments, from_arguments
from deploy.g1_pingpong.simulation import run_rally


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_policy_arguments(parser)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--duration", type=float, default=30)
  parser.add_argument("--extended-duration", type=float, default=120)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if args.episodes < 1 or not 0 < args.duration <= args.extended_duration:
    parser.error("Require positive episodes and 0 < duration <= extended-duration")
  args.output.mkdir(parents=True, exist_ok=False)
  torch.set_num_threads(1)
  deployment = from_arguments(args)
  report = {
    "policy": deployment.report(),
    "seed": args.seed,
    "episodes": args.episodes,
    "duration": args.duration,
    "extended_duration": args.extended_duration,
    "selection": "fixed; same motion for preparation, swing and return",
    "render": False,
    "motions": [],
  }
  for index, motion in enumerate(deployment.motions):
    runs = []
    for label, duration in (
      ("short", args.duration),
      ("extended", args.extended_duration),
    ):
      destination = args.output / f"motion_{index:02d}" / label
      destination.mkdir(parents=True)
      with (destination / "run.log").open("w") as log, contextlib.redirect_stdout(log):
        result = run_rally(
          deployment,
          "deploy/configs/real.yaml",
          "deploy/configs/mocap.yaml",
          motion_index=index,
          motion_select="fixed",
          episodes=args.episodes,
          duration=duration,
          seed=args.seed,
          output=destination,
        )
      runs.append(result)
      if (
        result["passed"] or result["failure"] or args.duration == args.extended_duration
      ):
        break
    result = runs[-1]
    failure = result["failure"] or {}
    status = (
      "passed_requested_recoveries"
      if result["passed"]
      else "failed_" + failure["phase"].lower()
      if failure
      else "incomplete_duration"
    )
    row = {
      "index": index,
      "motion": motion.name,
      "status": status,
      "short_completed": runs[0]["completed_swings"],
      "short_recovered": runs[0]["recovered_swings"],
      "completed": result["completed_swings"],
      "recovered": result["recovered_swings"],
      "max_tilt_degrees": result["max_control_tilt_degrees"],
      "simulation_time": result["simulation_time"],
      "failure_time": failure.get("time"),
      "failure_phase": failure.get("phase"),
      "error": failure.get("error"),
    }
    report["motions"].append(row)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    with (args.output / "report.csv").open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(row))
      writer.writeheader()
      writer.writerows(report["motions"])
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
  main()
