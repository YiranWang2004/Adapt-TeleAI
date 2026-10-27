"""Run a shared two-GPU FIFO while existing training processes finish unchanged.

The previous scheduler keeps supervising/reaping its current children. Its
source guard disables its unstarted job after a scheduler-only handoff marker;
this queue assumes that job and all repeats. Training source/data are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_g1_waist_ablation_pipeline as base  # noqa: E402


def process_state(pid):
  try:
    raw = Path(f"/proc/{pid}/stat").read_text()
  except FileNotFoundError:
    return "missing", None
  fields = raw[raw.rfind(")") + 2 :].split()
  state = fields[0]  # Linux proc field 3; field 52 is the wait-format exit status.
  code = os.waitstatus_to_exitcode(int(fields[49])) if state == "Z" else None
  return state, code


def make_jobs(previous):
  jobs = copy.deepcopy(previous["jobs"])
  pending = [j for j in jobs.values() if j["stage"] == 2 and j["state"] == "pending"]
  for priority, j in enumerate(pending):
    j["priority"] = priority
    j["gpu"] = None
    j["queue_origin"] = "unfinished_previous_round"
    j["dependencies"] = [j["tracker_job"]]
  start = len(pending)
  for offset, spec in enumerate(j for j in base.specifications() if j["stage"] == 2):
    key = spec["id"] + "_repeat1"
    jobs[key] = {
      **spec,
      "id": key,
      "gpu": None,
      "state": "pending",
      "stage": 2,
      "dependencies": [spec["tracker_job"]],
      "priority": start + offset,
      "queue_origin": "repeat_round_1",
      "repeat_round": 1,
      "task": base.TASKS[2],
      "group": base.GROUPS[2],
    }
  for j in jobs.values():
    if j["state"] == "running":
      j["adopted"] = True
  return jobs


def assignments(jobs):
  busy = {j["gpu"] for j in jobs.values() if j["state"] == "running"}
  free = [gpu for gpu in [0, 1] if gpu not in busy]
  queued = sorted(
    (j for j in jobs.values() if j["state"] == "pending"), key=lambda j: j["priority"]
  )
  result = []
  for j in queued:
    if not free:
      break
    if not all(jobs[d]["state"] == "complete" for d in j["dependencies"]):
      break  # Preserve FIFO; a later repeat cannot bypass unfinished Stage1.
    result.append((j["id"], free.pop(0)))
  return result


def disable_previous_pending(previous, output):
  # The loaded old scheduler already refuses new launches if its own source
  # fingerprint changes. A comment-only edit relinquishes pending launches
  # without stopping, killing, or restarting either training process.
  source = Path(base.__file__).resolve()
  marker = f"\n# Pending Stage2 scheduling transferred to {output}.\n"
  original = source.read_text()
  temporary = source.with_suffix(".fifo-handoff.tmp")
  temporary.write_text(original + marker)
  temporary.replace(source)
  base.atomic_json(
    previous / "scheduler_handoff.json",
    dict(
      new_status=str(output / "status.json"),
      method="scheduler-only comment fingerprint; current children remain supervised",
      timestamp=base.stamp(),
      training_source_and_data_changed=False,
    ),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--previous", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  os.chdir(ROOT)
  previous = args.previous.resolve()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  parent = json.loads((previous / "status.json").read_text())
  old_pid = parent["pipeline_pid"]
  frozen = base.source_hashes()
  own_hash = base.digest(__file__)
  if frozen != json.loads((previous / "source_sha256.json").read_text()):
    raise RuntimeError("Training source changed before scheduler takeover")
  if not all(
    parent["jobs"][key]["state"] == "complete"
    for key in ["s1_gmr_noextra", "s1_waist004"]
  ):
    raise RuntimeError("Both assigned Stage1 models must already be complete")
  for key in ["s1_gmr_noextra", "s1_waist004"]:
    a = parent["jobs"][key]["final_artifacts"]
    if base.digest(a["jit"]) != a["jit_sha256"]:
      raise ValueError("Assigned Stage1 tracker hash mismatch")
  owned = False
  children = {}
  configs = {}
  offsets = {}
  tails = {}
  state = {}

  def save():
    state["heartbeat"] = base.stamp()
    base.atomic_json(output / "status.json", state)

  def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by {signum}")

  signal.signal(signal.SIGTERM, interrupt)
  signal.signal(signal.SIGINT, interrupt)
  try:
    parent = json.loads((previous / "status.json").read_text())
    if parent["pipeline_pid"] != old_pid:
      raise RuntimeError("Previous scheduler ownership changed")
    base.atomic_json(output / "previous_status_at_takeover.json", parent)
    jobs = make_jobs(parent)
    disable_previous_pending(previous, output)
    frozen = base.source_hashes()
    state.update(
      state="running",
      pipeline_pid=os.getpid(),
      created_at=base.stamp(),
      previous_pipeline=str(previous),
      previous_scheduler_pid=old_pid,
      previous_pending_launches_disabled=True,
      training_processes_restarted=False,
      gpu_policy="FIFO, first idle card, one experiment per GPU",
      iterations_each_stage=50000,
      expected_final_iteration=49999,
      environments=4096,
      seed=42,
      jobs=jobs,
    )
    base.atomic_json(output / "source_sha256.json", frozen)
    base.atomic_json(
      output / "queue_plan.json",
      [
        dict(
          id=j["id"],
          priority=j.get("priority"),
          state=j["state"],
          waist=j["waist"],
          tracker_job=j.get("tracker_job"),
          gpu=j["gpu"],
        )
        for j in jobs.values()
        if j["stage"] == 2
      ],
    )
    owned = True
    save()
    print(
      "TAKEOVER: current training PIDs unchanged; pending launch ownership transferred",
      flush=True,
    )
    history_at = 0
    while True:
      observed_parent = json.loads((previous / "status.json").read_text())
      for key, j in jobs.items():
        if j["state"] != "running":
          continue
        try:
          log = Path(j["console_log"])
          if "run_dir" not in j:
            matches = list(
              (ROOT / "logs/rsl_rl" / j["group"]).glob(f"*_{j['run_name']}")
            )
            if len(matches) > 1:
              raise RuntimeError("Ambiguous run directory")
            if matches:
              j["run_dir"] = str(matches[0])
          if (
            "run_dir" in j
            and key not in configs
            and (Path(j["run_dir"]) / "params/agent.yaml").is_file()
          ):
            configs[key] = base.validate_saved_config(
              Path(j["run_dir"]), j, tracker=j["tracker"]
            )
            j["saved_configuration_validated"] = True
          if key not in offsets:
            offsets[key] = max(0, log.stat().st_size - 24000) if j.get("adopted") else 0
          with log.open() as f:
            f.seek(offsets[key])
            text = tails.get(key, "") + f.read()
            offsets[key] = f.tell()
          text = re.sub(r"\x1b\[[0-9;]*m", "", text)
          progress = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
          if progress:
            iteration, total = map(int, progress[-1])
            if total != 50000:
              raise ValueError("Wrong training length")
            if iteration != j.get("latest_iteration"):
              j["last_progress_at"] = base.stamp()
              j["last_progress_unix"] = time.time()
            j["latest_iteration"] = iteration
            j["progress_percent"] = round((iteration + 1) / 50000 * 100, 3)
          metrics = j.setdefault("latest_console_metrics", {})
          for name, value in re.findall(
            r"^\s*((?:Episode_Reward/|Metrics/|Episode_Termination/)[^:\n]+|Mean reward|Mean episode length|Mean value loss|Mean surrogate loss|Mean entropy loss|Mean action std):\s*([-+\d.eE]+|nan|inf)\s*$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
          ):
            val = float(value)
            if not -float("inf") < val < float("inf"):
              raise ValueError(f"Non-finite statistic {name}")
            metrics[name.strip()] = val
          tails[key] = text[-4096:]
          j["seconds_since_log_write"] = round(time.time() - log.stat().st_mtime, 1)
          j["warning"] = (
            "No progress for >10 minutes"
            if time.time() - j.get("last_progress_unix", j["started_unix"]) > 600
            else None
          )
          j["saved_checkpoints"] = (
            sorted(
              int(p.stem.removeprefix("model_"))
              for p in Path(j["run_dir"]).glob("model_*.pt")
            )
            if "run_dir" in j
            else []
          )
          if j.get("adopted"):
            observed = observed_parent["jobs"][key]
            ended = observed["state"] in ("complete", "failed", "interrupted")
            code = observed.get("exit_code") if ended else None
            if ended and code is None:
              raise RuntimeError("Previous supervisor did not provide an exit status")
          else:
            code = children[key].poll()
            ended = code is not None
          if ended:
            j["exit_code"] = code
            if code != 0:
              raise RuntimeError(f"Training exited with code {code}")
            if not j.get("saved_configuration_validated"):
              raise RuntimeError("Missing validated saved config")
            j["final_artifacts"] = base.finish_artifacts(
              Path(j["run_dir"]), j, j["tracker"], 50000
            )
            j["state"] = "complete"
            j["finished_at"] = base.stamp()
            print(f"COMPLETE {key} GPU{j['gpu']}; model_49999 validated", flush=True)
        except Exception as exc:
          # Leave an adopted, still-running training untouched if observation fails.
          if j.get("adopted") and process_state(j["pid"])[0] not in ("Z", "missing"):
            raise RuntimeError(
              f"Cannot safely observe adopted job {key}: {exc}"
            ) from exc
          if key in children and children[key].poll() is None:
            os.killpg(children[key].pid, signal.SIGINT)
            try:
              children[key].wait(timeout=20)
            except subprocess.TimeoutExpired:
              os.killpg(children[key].pid, signal.SIGKILL)
              children[key].wait()
          j["state"] = "failed"
          j["error"] = f"{type(exc).__name__}: {exc}"
          print(f"FAILED {key}: {j['error']}", flush=True)
      for j in jobs.values():
        if j["state"] == "pending" and any(
          jobs[d]["state"] != "complete" for d in j["dependencies"]
        ):
          if any(
            jobs[d]["state"] in ("failed", "dependency_failed")
            for d in j["dependencies"]
          ):
            j["state"] = "dependency_failed"
            j["error"] = "Assigned Stage1 unavailable"
      for key, gpu in assignments(jobs):
        j = jobs[key]
        if base.source_hashes() != frozen or base.digest(__file__) != own_hash:
          raise RuntimeError("Training source/data changed since queue takeover")
        parent_artifact = jobs[j["tracker_job"]]["final_artifacts"]
        tracker = parent_artifact["jit"]
        if base.digest(tracker) != parent_artifact["jit_sha256"]:
          raise ValueError("Assigned tracker changed")
        j["gpu"] = gpu
        j["tracker"] = tracker
        j["tracker_sha256"] = parent_artifact["jit_sha256"]
        batch = (
          previous.name
          if j["queue_origin"] == "unfinished_previous_round"
          else output.name
        )
        cmd, name = base.build_command(j, batch, tracker=tracker)
        log = output / (key + ".log")
        env = {
          **os.environ,
          "PYTHONUNBUFFERED": "1",
          "OMP_NUM_THREADS": "1",
          "MKL_NUM_THREADS": "1",
        }
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with log.open("w") as stream:
          child = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
          )
        children[key] = child
        j.update(
          state="running",
          pid=child.pid,
          run_name=name,
          command=cmd,
          console_log=str(log),
          started_at=base.stamp(),
          started_unix=time.time(),
        )
        print(
          f"START FIFO {key}: GPU{gpu}, priority={j['priority']}, PID={child.pid}",
          flush=True,
        )
      for suffix in ["", "_repeat1"]:
        pair = [
          "s2_waist004_from_waist004" + suffix,
          "s2_noextra_from_waist004" + suffix,
        ]
        if all(k in configs for k in pair):
          if base.normalized_pair_config(
            *configs[pair[0]]
          ) != base.normalized_pair_config(*configs[pair[1]]):
            raise ValueError("Waist pair differs beyond intended weight")
      if all(
        j["state"] in ("complete", "failed", "dependency_failed") for j in jobs.values()
      ):
        state["state"] = (
          "complete"
          if all(j["state"] == "complete" for j in jobs.values())
          else "failed"
        )
        state["finished_at"] = base.stamp()
        save()
        return
      if time.time() - history_at >= 60:
        state["gpu_snapshot"] = base.gpus()
        with (output / "history.jsonl").open("a") as f:
          f.write(
            json.dumps(dict(time=base.stamp(), jobs=jobs, gpus=state["gpu_snapshot"]))
            + "\n"
          )
        print(
          " | ".join(
            f"{k} GPU{j['gpu']} {j['state']} {j.get('latest_iteration', '-')}/50000"
            for k, j in jobs.items()
            if j["stage"] == 2
          ),
          flush=True,
        )
        history_at = time.time()
      save()
      time.sleep(15)
  except BaseException as exc:
    if owned:
      # Do not kill or restart training children when a monitoring exception occurs.
      state["state"] = "monitor_failed_training_left_running"
      state["error"] = f"{type(exc).__name__}: {exc}"
      save()
    raise


if __name__ == "__main__":
  main()
