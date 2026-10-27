"""Repeat Stage2 training on two GPUs without recording logs or checkpoints.

Only the current process state is exposed in /dev/shm (RAM); no history is kept.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_g1_waist_ablation_pipeline as base  # noqa: E402

STATUS = Path(f"/dev/shm/g1_stage2_occupy_{os.getuid()}.json")


def command(spec, gpu, tracker):
  job = {**spec, "gpu": gpu}
  cmd, _ = base.build_command(job, "no_logs", tracker=tracker)
  return [*cmd, "--no-logs", "True"]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--previous", required=True, type=Path)
  parser.add_argument("--start-index", type=int, default=0)
  args = parser.parse_args()
  os.chdir(ROOT)
  lock = Path(f"/dev/shm/g1_stage2_occupy_{os.getuid()}.lock").open("a")
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  parent = json.loads((args.previous / "status.json").read_text())
  specs = [s for s in base.specifications() if s["stage"] == 2]
  artifacts = {}
  for spec in specs:
    job = parent["jobs"][spec["tracker_job"]]
    artifact = job["final_artifacts"]
    if (
      job["state"] != "complete"
      or base.digest(artifact["jit"]) != artifact["jit_sha256"]
    ):
      raise ValueError("Stage1 tracker not complete or changed")
    artifacts[spec["tracker_job"]] = artifact
  children = {}
  state = dict(
    state="running",
    scheduler_pid=os.getpid(),
    no_logs=True,
    no_checkpoints=True,
    created_at=base.stamp(),
    started=args.start_index,
    completed=0,
    failed=0,
    active={},
    next_variant=specs[args.start_index % len(specs)]["id"],
  )

  def stop(signum, frame):
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, stop)
  signal.signal(signal.SIGINT, stop)
  try:
    while True:
      for gpu, child in list(children.items()):
        code = child.poll()
        if code is not None:
          state["completed" if code == 0 else "failed"] += 1
          del children[gpu]
          del state["active"][str(gpu)]
      for gpu in (0, 1):
        if gpu in children:
          continue
        spec = specs[state["started"] % len(specs)]
        artifact = artifacts[spec["tracker_job"]]
        if base.digest(artifact["jit"]) != artifact["jit_sha256"]:
          raise ValueError("Assigned tracker changed")
        env = dict(
          os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
        )
        env.pop("CUDA_VISIBLE_DEVICES", None)
        child = subprocess.Popen(
          command(spec, gpu, artifact["jit"]),
          cwd=ROOT,
          env=env,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
          start_new_session=True,
        )
        children[gpu] = child
        state["active"][str(gpu)] = dict(
          variant=spec["id"],
          pid=child.pid,
          gpu=gpu,
          round=state["started"] // 3 + 1,
          started_at=base.stamp(),
          waist=spec["waist"],
          tracker=artifact["jit"],
        )
        state["started"] += 1
      state["next_variant"] = specs[state["started"] % len(specs)]["id"]
      state["heartbeat"] = base.stamp()
      base.atomic_json(STATUS, state)
      time.sleep(15)
  finally:
    for child in children.values():
      if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    for child in children.values():
      try:
        child.wait(timeout=30)
      except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    state.update(state="stopped", heartbeat=base.stamp(), active={})
    base.atomic_json(STATUS, state)


if __name__ == "__main__":
  main()
