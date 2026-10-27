"""Continuously repeat the three Stage2 ablations on a shared two-GPU FIFO."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_g1_waist_ablation_pipeline as base  # noqa: E402
from scripts.schedule_g1_stage2_fifo import assignments  # noqa: E402


def enqueue_round(jobs, number):
  """Append a full ordered round; never impose a round completion barrier."""
  for offset, spec in enumerate(s for s in base.specifications() if s["stage"] == 2):
    key = f"{spec['id']}_loop{number:04d}"
    if key in jobs:
      raise ValueError(f"Duplicate job {key}")
    jobs[key] = {
      **spec,
      "id": key,
      "gpu": None,
      "state": "pending",
      "dependencies": [spec["tracker_job"]],
      "priority": (number - 1) * 3 + offset,
      "repeat_round": number,
      "task": base.TASKS[2],
      "group": base.GROUPS[2],
    }


def read_progress(job, offsets, tails):
  key = job["id"]
  log = Path(job["console_log"])
  with log.open() as stream:
    stream.seek(offsets.get(key, 0))
    text = tails.get(key, "") + stream.read()
    offsets[key] = stream.tell()
  text = re.sub(r"\x1b\[[0-9;]*m", "", text)
  progress = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
  if progress:
    iteration, total = map(int, progress[-1])
    if total != 50000:
      raise ValueError("Unexpected training length")
    if iteration != job.get("latest_iteration"):
      job["last_progress_unix"] = time.time()
      job["last_progress_at"] = base.stamp()
    job["latest_iteration"] = iteration
    job["progress_percent"] = round((iteration + 1) / 50000 * 100, 3)
  metrics = job.setdefault("latest_console_metrics", {})
  for name, value in re.findall(
    r"^\s*((?:Episode_Reward/|Metrics/|Episode_Termination/)[^:\n]+|Mean reward|Mean episode length|Mean value loss|Mean surrogate loss|Mean entropy loss|Mean action std):\s*([-+\d.eE]+|nan|inf)\s*$",
    text,
    flags=re.MULTILINE | re.IGNORECASE,
  ):
    value = float(value)
    if not math.isfinite(value):
      raise ValueError(f"Nonfinite {name}")
    metrics[name.strip()] = value
  tails[key] = text[-4096:]
  job["warning"] = (
    "No progress for >10 minutes"
    if time.time() - job.get("last_progress_unix", job["started_unix"]) > 600
    else None
  )
  if "run_dir" not in job:
    matches = list((ROOT / "logs/rsl_rl" / job["group"]).glob(f"*_{job['run_name']}"))
    if len(matches) > 1:
      raise ValueError("Ambiguous run directory")
    if matches:
      job["run_dir"] = str(matches[0])
  if "run_dir" in job:
    run = Path(job["run_dir"])
    if (
      not job.get("saved_configuration_validated")
      and (run / "params/agent.yaml").exists()
    ):
      base.validate_saved_config(run, job, tracker=job["tracker"])
      job["saved_configuration_validated"] = True
    job["saved_checkpoints"] = sorted(
      int(p.stem.removeprefix("model_")) for p in run.glob("model_*.pt")
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--previous", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  os.chdir(ROOT)
  # A machine-wide lock prevents two infinite loops competing for the same GPUs.
  lock = (ROOT / "logs/training_pipelines/g1_stage2_continuous.lock").open("a")
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  parent = json.loads((args.previous / "status.json").read_text())
  if any(j["state"] == "running" for j in parent["jobs"].values()):
    raise RuntimeError("Previous pipeline still has running jobs")
  jobs = {}
  for key in ("s1_gmr_noextra", "s1_waist004"):
    job = parent["jobs"][key]
    if job["state"] != "complete":
      raise ValueError("Assigned Stage1 incomplete")
    artifact = job["final_artifacts"]
    if base.digest(artifact["jit"]) != artifact["jit_sha256"]:
      raise ValueError("Stage1 tracker changed")
    jobs[key] = copy.deepcopy(job)
  output = args.output.absolute()
  output.mkdir(parents=True, exist_ok=False)
  frozen = base.source_hashes()
  own_hash = base.digest(__file__)
  base.atomic_json(output / "source_sha256.json", frozen)
  state = dict(
    state="running",
    pipeline_pid=os.getpid(),
    created_at=base.stamp(),
    previous_pipeline=str(args.previous),
    continuous=True,
    gpu_policy="FIFO GPU0/GPU1, one experiment per card; repeat indefinitely",
    iterations_each_stage=50000,
    expected_final_iteration=49999,
    environments=4096,
    seed=42,
    rounds_enqueued=0,
    jobs=jobs,
  )
  children, offsets, tails = {}, {}, {}
  history_at = 0
  while True:
    for key, child in list(children.items()):
      job = jobs[key]
      try:
        read_progress(job, offsets, tails)
        code = child.poll()
        if code is None:
          continue
        job["exit_code"] = code
        if code != 0:
          raise RuntimeError(f"Training exited with code {code}")
        if not job.get("saved_configuration_validated"):
          raise ValueError("Missing validated configuration")
        job["final_artifacts"] = base.finish_artifacts(
          Path(job["run_dir"]),
          job,
          job["tracker"],
          50000,
        )
        job["state"] = "complete"
        print(f"COMPLETE {key} GPU{job['gpu']}; model_49999 validated", flush=True)
      except Exception as exc:
        # Do not kill a live training child on a monitoring error.
        if child.poll() is None:
          job["monitor_error"] = f"{type(exc).__name__}: {exc}"
          continue
        job["state"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {key}: {job['error']}; continue FIFO", flush=True)
      job["finished_at"] = base.stamp()
      del children[key]
      offsets.pop(key, None)
      tails.pop(key, None)
    if not any(j["state"] == "pending" for j in jobs.values()):
      state["rounds_enqueued"] += 1
      enqueue_round(jobs, state["rounds_enqueued"])
    problem = None
    if base.source_hashes() != frozen or base.digest(__file__) != own_hash:
      problem = (
        "Training source/data changed; new launches held to prevent mixed versions"
      )
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
      problem = (
        "Less than 10 GiB free disk; new launches held; no existing logs removed"
      )
    state["launch_warning"] = problem
    if problem is None:
      for key, gpu in assignments(jobs):
        job = jobs[key]
        artifact = jobs[job["tracker_job"]]["final_artifacts"]
        tracker = artifact["jit"]
        if base.digest(tracker) != artifact["jit_sha256"]:
          raise ValueError("Assigned Stage1 tracker changed")
        job.update(gpu=gpu, tracker=tracker, tracker_sha256=artifact["jit_sha256"])
        command, name = base.build_command(job, output.name, tracker=tracker)
        log = output / (key + ".log")
        env = dict(
          os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
        )
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with log.open("w") as stream:
          child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
          )
        children[key] = child
        job.update(
          state="running",
          pid=child.pid,
          run_name=name,
          command=command,
          console_log=str(log),
          started_at=base.stamp(),
          started_unix=time.time(),
        )
        print(f"START {key} GPU{gpu} PID={child.pid}", flush=True)
    state["heartbeat"] = base.stamp()
    if time.time() - history_at >= 60:
      state["gpu_snapshot"] = base.gpus()
      # Keep historical summaries compact; full console/TensorBoard logs remain per run.
      with (output / "history.jsonl").open("a") as stream:
        stream.write(
          json.dumps(
            dict(
              time=state["heartbeat"],
              gpus=state["gpu_snapshot"],
              active={
                k: {
                  n: j.get(n) for n in ("state", "gpu", "latest_iteration", "warning")
                }
                for k, j in jobs.items()
                if j["state"] in ("running", "pending")
              },
            )
          )
          + "\n"
        )
      history_at = time.time()
    base.atomic_json(output / "status.json", state)
    time.sleep(15)


if __name__ == "__main__":
  main()
