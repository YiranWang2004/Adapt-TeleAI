"""Persistent 2×Stage1 / 3×Stage2 TeleGMR waist ablation, without global config edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
  1: "Mjlab-PingPong-Tracking-G1-TeleGMR-Stage1-RandomDt",
  2: "Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive",
}
GROUPS = {1: "g1_pingpong_tracking_stage1_random_dt", 2: "g1_pingpong_receive_stage2"}
PROTECTED = (
  "g1_pingpong_receive_stage2_tele_gmr",
  "g1_pingpong_receive_stage2_tele_gmr_hard",
)


def stamp():
  return datetime.now().isoformat(timespec="seconds")


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
  path = Path(path)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
  tmp.replace(path)


def specifications():
  # Queue the GPU1 Stage2 pair on one card. Both use the SAME final Stage1 JIT.
  s1 = ["s1_gmr_noextra", "s1_waist004"]
  return [
    dict(id=s1[0], stage=1, gpu=0, waist=0.0, dependencies=[]),
    dict(id=s1[1], stage=1, gpu=1, waist=-0.04, dependencies=[]),
    dict(
      id="s2_gmr_noextra_from_gmr",
      stage=2,
      gpu=0,
      waist=0.0,
      tracker_job=s1[0],
      dependencies=s1,
    ),
    dict(
      id="s2_waist004_from_waist004",
      stage=2,
      gpu=1,
      waist=-0.04,
      tracker_job=s1[1],
      dependencies=s1,
    ),
    dict(
      id="s2_noextra_from_waist004",
      stage=2,
      gpu=1,
      waist=0.0,
      tracker_job=s1[1],
      dependencies=s1 + ["s2_waist004_from_waist004"],
    ),
  ]


def build_command(job, batch_name, *, iterations=50000, tracker=None):
  stage = job["stage"]
  group = GROUPS[stage]
  if group in PROTECTED:
    raise ValueError("Protected migrated experiment cannot be an output group")
  if stage == 2 and tracker is None:
    raise ValueError("Stage2 requires its validated Stage1 tracker")
  name = f"g1_tele_gmr_{job['id']}_50k_4096_seed42_{batch_name}"
  cmd = [
    str(ROOT / ".venv/bin/python"),
    "-u",
    "-m",
    "mjlab.scripts.train",
    TASKS[stage],
    "--gpu-ids",
    f"[{job['gpu']}]",
    "--env.scene.num-envs",
    "4096",
    "--env.rewards.waist-action-rate.weight",
    str(job["waist"]),
    "--agent.max-iterations",
    str(iterations),
    "--agent.save-interval",
    "2000",
    "--agent.seed",
    "42",
    "--agent.resume",
    "False",
    "--agent.logger",
    "tensorboard",
    "--agent.upload-model",
    "False",
    "--agent.run-name",
    name,
    "--agent.experiment-name",
    group,
  ]
  if tracker is not None:
    cmd += ["--env.actions.joint-pos.tracker-file", str(tracker)]
  return cmd, name


def source_hashes():
  files = list((ROOT / "src").rglob("*.py"))
  files += [
    Path(__file__).resolve(),
    ROOT / "scripts/train_g1_tele_gmr_waist_pipeline.py",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
  ]
  data = ROOT / "dataset/g1_pingpong_tele_gmr"
  files += [data / "manifest.json", *sorted((data / "motions").glob("*.npz"))]
  return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def validate_saved_config(run, job, *, iterations=50000, tracker=None):
  from deploy.g1_pingpong.policy import training_config

  env = training_config(run / "params/env.yaml")
  agent = training_config(run / "params/agent.yaml")
  expected = [
    (env["rewards"]["waist_action_rate"]["weight"], job["waist"], "waist reward"),
    (env["rewards"]["action_rate"]["weight"], -0.02, "whole body reward"),
    (env["scene"]["num_envs"], 4096, "environments"),
    (agent["max_iterations"], iterations, "iterations"),
    (agent["save_interval"], 2000, "save interval"),
    (agent["seed"], 42, "seed"),
    (agent["resume"], False, "fresh training"),
    (agent["experiment_name"], GROUPS[job["stage"]], "output group"),
  ]
  for actual, desired, label in expected:
    if actual != desired:
      raise ValueError(f"{job['id']}: wrong {label}: {actual!r} != {desired!r}")
  if (
    job["stage"] == 2
    and Path(env["actions"]["joint_pos"]["tracker_file"]).resolve()
    != Path(tracker).resolve()
  ):
    raise ValueError("Saved tracker does not match the assigned Stage1")
  return env, agent


def normalized_pair_config(env, agent):
  # Other than the intended waist weight, Stage1 paired configs must be equal.
  env = json.loads(json.dumps(env, default=str))
  agent = json.loads(json.dumps(agent, default=str))
  env["rewards"]["waist_action_rate"]["weight"] = "<EXPERIMENT_WEIGHT>"
  agent.pop("run_name", None)
  return env, agent


def finish_artifacts(run, job, tracker, iterations):
  import torch
  from scripts.train_g1_tele_gmr_waist_pipeline import validate_final_artifacts

  final = validate_final_artifacts(
    run,
    iterations - 1,
    132 if job["stage"] == 1 else 186,
    29 if job["stage"] == 1 else 30,
  )
  if job["stage"] == 2:
    actual = digest(tracker)
    if digest(run / "tracker.pt") != actual:
      raise ValueError("Stage2 bundled tracker differs from assigned Stage1")
    infos = (
      torch.load(final["checkpoint"], map_location="cpu", weights_only=False).get(
        "infos"
      )
      or {}
    )
    if infos.get("pingpong_tracker_sha256", infos.get("a3_tracker_sha256")) != actual:
      raise ValueError("Final Stage2 checkpoint tracker dependency differs")
    final["tracker_sha256"] = actual
  saved = sorted(int(p.stem.removeprefix("model_")) for p in run.glob("model_*.pt"))
  expected = sorted(set(range(0, iterations, 2000)) | {iterations - 1})
  if saved != expected:
    raise ValueError(f"Checkpoint sequence mismatch: expected {expected}, got {saved}")
  final["saved_checkpoint_iterations"] = saved
  return final


def ready_jobs(jobs):
  busy = {j["gpu"] for j in jobs.values() if j["state"] == "running"}
  ready = []
  for j in jobs.values():
    if (
      j["state"] == "pending"
      and j["gpu"] not in busy
      and all(jobs[d]["state"] == "complete" for d in j["dependencies"])
    ):
      ready.append(j)
      busy.add(j["gpu"])
  return ready


def gpus():
  result = subprocess.run(
    [
      "nvidia-smi",
      "--query-gpu=index,memory.used,utilization.gpu",
      "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
    timeout=10,
  )
  return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  os.chdir(ROOT)
  sys.path.insert(0, str(ROOT))
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  specs = specifications()
  jobs = {
    j["id"]: {
      **j,
      "state": "pending",
      "task": TASKS[j["stage"]],
      "group": GROUPS[j["stage"]],
    }
    for j in specs
  }
  status = dict(
    pipeline_pid=os.getpid(),
    created_at=stamp(),
    state="preflight",
    iterations_each_stage=50000,
    expected_final_iteration=49999,
    environments=4096,
    seed=42,
    whole_body_action_rate=-0.02,
    gpu1_stage2_schedule="sequential: waist004 then noextra; identical waist004 Stage1 tracker",
    jobs=jobs,
  )
  children = {}
  configs = {}
  tails = {}
  offsets = {}
  frozen = source_hashes()
  atomic_json(output / "source_sha256.json", frozen)

  def save():
    status["heartbeat"] = stamp()
    atomic_json(output / "status.json", status)

  def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")

  signal.signal(signal.SIGINT, interrupted)
  signal.signal(signal.SIGTERM, interrupted)
  save()
  try:
    # Parse every job through the real CLI before consuming a GPU.
    import tyro

    import mjlab
    from mjlab.scripts.train import TrainConfig

    for job in jobs.values():
      placeholder = (
        ROOT / "ckpts/g1_pingpong_tele_gmr/tracker.pt" if job["stage"] == 2 else None
      )
      cmd, name = build_command(job, output.name, tracker=placeholder)
      cfg = tyro.cli(
        TrainConfig,
        args=cmd[5:],
        default=TrainConfig.from_task(job["task"]),
        config=mjlab.TYRO_FLAGS,
      )
      if (
        cfg.env.rewards["waist_action_rate"].weight != job["waist"]
        or cfg.agent.max_iterations != 50000
      ):
        raise ValueError("CLI override failed")
      job["run_name"] = name
      job["planned_command"] = cmd
    atomic_json(output / "plan.json", list(jobs.values()))
    status["state"] = "running"
    save()
    print(f"Preflight passed; batch={output}", flush=True)
    history_at = 0.0
    while True:
      for key, child in list(children.items()):
        j = jobs[key]
        log = Path(j["console_log"])
        if "run_dir" not in j:
          matches = list((ROOT / "logs/rsl_rl" / j["group"]).glob(f"*_{j['run_name']}"))
          if len(matches) > 1:
            raise RuntimeError("Ambiguous run directory")
          if matches:
            j["run_dir"] = str(matches[0])
        try:
          if (
            "run_dir" in j
            and not j.get("saved_configuration_validated")
            and (Path(j["run_dir"]) / "params/agent.yaml").is_file()
          ):
            configs[key] = validate_saved_config(
              Path(j["run_dir"]), j, tracker=j.get("tracker")
            )
            j["saved_configuration_validated"] = True
          with log.open() as f:
            f.seek(offsets.get(key, 0))
            text = tails.get(key, "") + f.read()
            offsets[key] = f.tell()
          text = re.sub(r"\x1b\[[0-9;]*m", "", text)
          progress = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
          if progress:
            iteration, total = map(int, progress[-1])
            if total != 50000:
              raise ValueError(f"Unexpected training length {total}")
            if iteration != j.get("latest_iteration"):
              j["last_progress_at"] = stamp()
              j["last_progress_unix"] = time.time()
            j["latest_iteration"] = iteration
            j["progress_percent"] = round((iteration + 1) / total * 100, 3)
          metrics = j.setdefault("latest_console_metrics", {})
          for name, value in re.findall(
            r"^\s*((?:Episode_Reward/|Metrics/|Episode_Termination/)[^:\n]+|Mean reward|Mean episode length|Mean value loss|Mean surrogate loss|Mean entropy loss|Mean action std):\s*([-+\d.eE]+|nan|inf)\s*$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
          ):
            parsed = float(value)
            if not (-float("inf") < parsed < float("inf")):
              raise ValueError(f"Non-finite training statistic {name}={value}")
            metrics[name.strip()] = parsed
          tails[key] = text[-4096:]
          j["seconds_since_log_write"] = round(time.time() - log.stat().st_mtime, 1)
          j["warning"] = (
            "No training progress for >10 minutes"
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
          code = child.poll()
          if code is not None:
            j["exit_code"] = code
            if code:
              raise RuntimeError(f"Child exited with code {code}; see {log}")
            if not j.get("saved_configuration_validated"):
              raise RuntimeError("Run did not save validated configs")
            j["final_artifacts"] = finish_artifacts(
              Path(j["run_dir"]), j, j.get("tracker"), 50000
            )
            j["state"] = "complete"
            j["finished_at"] = stamp()
            print(
              f"{key} complete; model_49999/JIT and dependency validated", flush=True
            )
            del children[key]
        except Exception as exc:
          if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
              child.wait(timeout=20)
            except subprocess.TimeoutExpired:
              os.killpg(child.pid, signal.SIGKILL)
              child.wait()
          j["state"] = "failed"
          j["error"] = f"{type(exc).__name__}: {exc}"
          j["finished_at"] = stamp()
          del children[key]
          print(f"FAILED {key}: {j['error']}", flush=True)
      # Validate the experimentally intended pair after both configs exist.
      if all(k in configs for k in ["s1_gmr_noextra", "s1_waist004"]):
        if normalized_pair_config(*configs["s1_gmr_noextra"]) != normalized_pair_config(
          *configs["s1_waist004"]
        ):
          raise ValueError(
            "Stage1 paired configurations differ beyond waist weight/run name"
          )
        status["stage1_pair_configuration_validated"] = True
      pair = ["s2_waist004_from_waist004", "s2_noextra_from_waist004"]
      if all(k in configs for k in pair):
        if normalized_pair_config(*configs[pair[0]]) != normalized_pair_config(
          *configs[pair[1]]
        ):
          raise ValueError("GPU1 Stage2 configs differ beyond waist weight/run name")
        status["stage2_waist_pair_configuration_validated"] = True
      for j in jobs.values():
        if j["state"] == "pending" and any(
          jobs[d]["state"] in ("failed", "dependency_failed") for d in j["dependencies"]
        ):
          j["state"] = "dependency_failed"
          j["error"] = (
            "Required upstream job did not complete; no mismatched tracker substituted"
          )
      for j in ready_jobs(jobs):
        if source_hashes() != frozen:
          j["state"] = "failed"
          j["error"] = (
            "Training source/data changed since batch start; refusing mixed experiment versions"
          )
          continue
        tracker = None
        if j["stage"] == 2:
          parent = jobs[j["tracker_job"]]["final_artifacts"]
          tracker = Path(parent["jit"]).resolve()
          if digest(tracker) != parent["jit_sha256"]:
            raise ValueError("Upstream tracker changed after validation")
          j["tracker"] = str(tracker)
          j["tracker_sha256"] = parent["jit_sha256"]
        command, _ = build_command(j, output.name, tracker=tracker)
        log = output / (j["id"] + ".log")
        env = {
          **os.environ,
          "PYTHONUNBUFFERED": "1",
          "OMP_NUM_THREADS": "1",
          "MKL_NUM_THREADS": "1",
        }
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
        j.update(
          state="running",
          pid=child.pid,
          command=command,
          console_log=str(log),
          started_at=stamp(),
          started_unix=time.time(),
        )
        children[j["id"]] = child
        print(
          f"START {j['id']}: GPU{j['gpu']}, waist={j['waist']}, PID={child.pid}",
          flush=True,
        )
      if not children and all(
        j["state"] in ("complete", "failed", "dependency_failed") for j in jobs.values()
      ):
        status["state"] = (
          "complete"
          if all(j["state"] == "complete" for j in jobs.values())
          else "failed"
        )
        status["finished_at"] = stamp()
        save()
        return
      if time.time() - history_at >= 60:
        status["gpu_snapshot"] = gpus()
        with (output / "history.jsonl").open("a") as f:
          f.write(
            json.dumps(
              dict(
                time=stamp(),
                gpus=status["gpu_snapshot"],
                jobs={
                  k: {
                    n: j.get(n)
                    for n in (
                      "state",
                      "latest_iteration",
                      "saved_checkpoints",
                      "warning",
                      "latest_console_metrics",
                    )
                  }
                  for k, j in jobs.items()
                },
              )
            )
            + "\n"
          )
        print(
          " | ".join(
            f"{k}: {j['state']} {j.get('latest_iteration', '-')}/50000"
            for k, j in jobs.items()
          ),
          flush=True,
        )
        history_at = time.time()
      save()
      time.sleep(15)
  except BaseException as exc:
    status["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    status["error"] = f"{type(exc).__name__}: {exc}"
    for child in children.values():
      if child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
    for key, child in children.items():
      try:
        child.wait(timeout=20)
      except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
      jobs[key]["state"] = "interrupted"
    save()
    raise


if __name__ == "__main__":
  main()

# Pending Stage2 scheduling transferred to /home/lhy/table-tennis/Adapt-TeleAI/logs/training_pipelines/g1_stage2_fifo_repeat1_20261020_191401.
