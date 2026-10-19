"""Repeat full fixed-motion rally tests across independent ball random seeds."""

import argparse
import contextlib
import csv
import gzip
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_seed(config, output, seed, episodes, duration):
  import torch
  import yaml

  from deploy.g1_pingpong.configuration import load_deployment
  from deploy.g1_pingpong.simulation import run_rally

  torch.set_num_threads(1)
  output = Path(output)
  rows = []
  with (output / f"seed_{seed}.log").open("w") as log, contextlib.redirect_stdout(log):
    deployment = load_deployment(**yaml.safe_load(Path(config).read_text()))
    for index, motion in enumerate(deployment.motions):
      destination = output / f"motion_{index:02d}" / f"seed_{seed}"
      result = run_rally(
        deployment,
        "deploy/configs/real.yaml",
        "deploy/configs/mocap.yaml",
        motion_index=index,
        motion_select="fixed",
        episodes=episodes,
        duration=duration,
        seed=seed,
        output=destination,
      )
      trace = destination / "trace.jsonl"
      with (
        trace.open("rb") as source,
        gzip.open(trace.with_suffix(".jsonl.gz"), "wb") as target,
      ):
        shutil.copyfileobj(source, target)
      trace.unlink()
      failure = result["failure"] or {}
      selected = sum(e["event"] == "motion_selected" for e in result["events"])
      if result["passed"]:
        outcome = "passed"
      elif failure:
        outcome = (
          "failed_initial_standing"
          if not selected
          else "failed_after_recovery"
          if failure["phase"] == "PREPARE"
          else "failed_" + failure["phase"].lower()
        )
      else:
        outcome = "no_swing" if not selected else "incomplete_duration"
      row = {
        "index": index,
        "motion": motion.name,
        "seed": seed,
        "outcome": outcome,
        "selected": selected,
        "completed": result["completed_swings"],
        "recovered": result["recovered_swings"],
        "max_tilt_degrees": result["max_control_tilt_degrees"],
        "simulation_time": result["simulation_time"],
        "failure_time": failure.get("time"),
        "error": failure.get("error"),
      }
      (destination / "result.json").write_text(json.dumps(row, indent=2))
      rows.append(row)
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--repeats", type=int, default=10)
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--duration", type=float, default=120)
  parser.add_argument("--workers", type=int, default=4)
  args = parser.parse_args()
  if min(args.repeats, args.episodes, args.duration, args.workers) <= 0:
    parser.error("Counts and duration must be positive")
  args.output.mkdir(parents=True, exist_ok=False)
  seeds = list(range(args.seed, args.seed + args.repeats))
  metadata = {
    **vars(args),
    "config": str(args.config),
    "output": str(args.output),
    "seeds": seeds,
  }
  (args.output / "settings.json").write_text(json.dumps(metadata, indent=2))
  for config in (
    args.config,
    *map(
      Path,
      (
        "deploy/configs/real.yaml",
        "deploy/configs/mujoco.yaml",
        "deploy/configs/mocap.yaml",
      ),
    ),
  ):
    shutil.copy2(config, args.output / config.name)
  rows = []
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {
      pool.submit(
        run_seed, str(args.config), str(args.output), seed, args.episodes, args.duration
      ): seed
      for seed in seeds
    }
    for future in as_completed(futures):
      rows.extend(future.result())
      print(f"Finished seed {futures[future]}: {len(rows)} runs", flush=True)
  rows.sort(key=lambda row: (row["index"], row["seed"]))
  aggregates = []
  for index in sorted({row["index"] for row in rows}):
    trials = [row for row in rows if row["index"] == index]
    outcomes = Counter(row["outcome"] for row in trials)
    aggregates.append(
      {
        "index": index,
        "motion": trials[0]["motion"],
        "trials": len(trials),
        "passed": outcomes["passed"],
        "pass_rate": outcomes["passed"] / len(trials),
        "completed": sum(row["completed"] for row in trials),
        "recovered": sum(row["recovered"] for row in trials),
        "min_recovered": min(row["recovered"] for row in trials),
        "max_recovered": max(row["recovered"] for row in trials),
        "max_tilt_degrees": max(row["max_tilt_degrees"] for row in trials),
        "outcomes": json.dumps(dict(outcomes), sort_keys=True),
      }
    )
  for name, records in (("trials", rows), ("summary", aggregates)):
    (args.output / f"{name}.json").write_text(json.dumps(records, indent=2))
    with (args.output / f"{name}.csv").open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(records[0]))
      writer.writeheader()
      writer.writerows(records)
  print(json.dumps(aggregates), flush=True)


if __name__ == "__main__":
  main()
