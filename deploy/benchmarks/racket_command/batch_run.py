"""Run a frozen full-grid suite in independent MuJoCo worker processes."""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

import yaml

from deploy.benchmarks.racket_command.domain import resolve
from deploy.benchmarks.racket_command.sampling import (
  Trial,
  load_suite,
  sample_suite,
  suite_document,
)

_environment = None
_journal = None
_worker_log = None


def initialize(config, output):
  global _environment, _journal, _worker_log
  import torch

  torch.set_num_threads(1)
  from deploy.benchmarks.racket_command.environment import MujocoCommandEnvironment
  from deploy.hope_g1.command_adapter import HopeCommandAdapter

  folder = Path(output) / "workers" / str(os.getpid())
  folder.mkdir(parents=True, exist_ok=True)
  _worker_log = (folder / "worker.log").open("w", buffering=1)
  with redirect_stdout(_worker_log), redirect_stderr(_worker_log):
    agent = HopeCommandAdapter(
      Path("deploy/hope_g1/configs/task.yaml"),
      Path("deploy/hope_g1/configs/rally_mujoco.yaml"),
      Path("deploy/hope_g1/configs/real.yaml"),
      Path("deploy/hope_g1/configs/mocap.yaml"),
      folder,
    )
    _environment = MujocoCommandEnvironment(
      agent,
      config,
      folder,
      False,
      1,
      0,
      show_ball=False,
      command_only=config["evaluation_mode"] == "policy_tracking",
    )
  (folder / "adapter_metadata.json").write_text(json.dumps(agent.metadata, indent=2))
  _journal = (folder / "results.jsonl").open("a", buffering=1)


def execute(trials):
  rows = []
  for document in trials:
    trial = Trial(**document)
    result = _environment.run_trial(trial, 0)
    row = {**result, "command": asdict(trial)}
    _journal.write(json.dumps(row, allow_nan=False) + "\n")
    rows.append(row)
    _environment.completed.clear()
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config", type=Path, default=Path(__file__).with_name("grid_tts_full.yaml")
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
  parser.add_argument("--chunk-size", type=int, default=10)
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  if args.workers < 1 or args.chunk_size < 1:
    parser.error("workers/chunk-size must be positive")
  output = args.output
  output.mkdir(parents=True, exist_ok=True)
  suite_path = output / "suite.json"
  if suite_path.exists():
    if not args.resume:
      parser.error("Existing suite; use a new output directory or --resume")
    suite = json.loads(suite_path.read_text())
    load_suite(suite)
  else:
    if args.resume:
      parser.error("No suite to resume")
    config = resolve(yaml.safe_load(args.config.read_text()))
    suite = suite_document(config, sample_suite(config))
    suite_path.write_text(json.dumps(suite, indent=2))
  config = suite["config"]
  completed = {}
  if args.resume:
    for path in [
      output / "results.jsonl",
      *sorted((output / "workers").glob("*/results.jsonl")),
    ]:
      if path.exists():
        for line in path.read_text().splitlines():
          try:
            row = json.loads(line)
          except json.JSONDecodeError:
            continue  # An interrupted last journal write is not a completed trial.
          if row["trial_id"] in completed and completed[row["trial_id"]] != row:
            raise ValueError("Conflicting duplicate result")
          completed[row["trial_id"]] = row
  remaining = [t for t in suite["trials"] if t["trial_id"] not in completed]
  chunks = [
    remaining[i : i + args.chunk_size]
    for i in range(0, len(remaining), args.chunk_size)
  ]
  start = time.monotonic()
  print(
    f"Frozen suite {suite['sha256']}: {len(suite['trials'])} trials, {len(completed)} already complete, {args.workers} workers",
    flush=True,
  )
  status = "running"
  metadata = dict(
    evaluation_mode=config["evaluation_mode"],
    parallel_workers=args.workers,
    planner_used=False,
    selector_used=True,
    reset_count_per_trial=1,
    spatial_coverage=config.get("spatial_coverage", "support_union"),
    scope="MuJoCo command-conditioned physical-ball benchmark; all rectangular cells retained, no outcome filtering",
  )
  try:
    with (output / "results.jsonl").open("w", buffering=1) as stream:
      for row in completed.values():
        stream.write(json.dumps(row, allow_nan=False) + "\n")
      if chunks:
        with ProcessPoolExecutor(
          max_workers=args.workers,
          mp_context=multiprocessing.get_context("spawn"),
          initializer=initialize,
          initargs=(config, str(output)),
        ) as pool:
          futures = [pool.submit(execute, chunk) for chunk in chunks]
          for future in as_completed(futures):
            for row in future.result():
              completed[row["trial_id"]] = row
              stream.write(json.dumps(row, allow_nan=False) + "\n")
            elapsed = time.monotonic() - start
            done = len(completed)
            progress = dict(
              completed=done,
              scheduled=len(suite["trials"]),
              elapsed_seconds=round(elapsed, 1),
            )
            (output / "progress.json").write_text(json.dumps(progress))
            print(f"{done}/{len(suite['trials'])} | {elapsed:.1f}s", flush=True)
    status = "complete" if len(completed) == len(suite["trials"]) else "partial"
  except KeyboardInterrupt:
    status = "interrupted"
    raise
  except Exception:
    status = "error"
    raise
  finally:
    from deploy.benchmarks.racket_command.visualization import write_reports

    worker_metadata = next((output / "workers").glob("*/adapter_metadata.json"), None)
    if worker_metadata:
      metadata = {**json.loads(worker_metadata.read_text()), **metadata}
    (output / "adapter_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_reports(output, config, suite, list(completed.values()), metadata, status)
    print(f"Report ({status}): {output / 'report.html'}", flush=True)


if __name__ == "__main__":
  main()
