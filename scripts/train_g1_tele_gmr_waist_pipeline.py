"""Monitor fresh Stage1/Stage2 jobs and hand off only validated final artifacts."""

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
STAGES = {
  1: (
    "Mjlab-PingPong-Tracking-G1-TeleGMR-Stage1-RandomDt",
    "g1_pingpong_tracking_stage1_random_dt",
  ),
  2: (
    "Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive",
    "g1_pingpong_receive_stage2",
  ),
}


def sha256(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_final_artifacts(run, final_iteration, input_dim, output_dim):
  """A 30000-update fresh run ends at 29999, never at 30000."""
  import torch

  run = Path(run)
  checkpoint = run / f"model_{final_iteration}.pt"
  tracker = run / "jit" / f"modeljit_{final_iteration}.pt"
  metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
  if metadata.get("iter") != final_iteration:
    raise ValueError(f"Wrong final checkpoint iteration: {metadata.get('iter')}")
  for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
    if key not in metadata:
      raise ValueError(f"Incomplete training checkpoint: missing {key}")
  model = torch.jit.load(str(tracker), map_location="cpu").eval()
  with torch.inference_mode():
    result = model(torch.zeros(1, input_dim))
  if result.shape != (1, output_dim) or not torch.isfinite(result).all():
    raise ValueError("Final JIT output is invalid")
  return {
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "jit": str(tracker),
    "jit_sha256": sha256(tracker),
    "iteration": final_iteration,
  }


def source_hashes():
  paths = list((ROOT / "src").rglob("*.py"))
  paths += [Path(__file__).resolve(), ROOT / "pyproject.toml", ROOT / "uv.lock"]
  return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def validate_run_config(run, stage, tracker):
  from deploy.g1_pingpong.policy import training_config

  env = training_config(run / "params/env.yaml")
  agent = training_config(run / "params/agent.yaml")
  assert env["rewards"]["waist_action_rate"]["weight"] == -0.04
  assert env["rewards"]["action_rate"]["weight"] == -0.02
  assert env["scene"]["num_envs"] == 4096
  assert agent["seed"] == 42 and agent["resume"] is False
  assert agent["max_iterations"] == 30000
  assert agent["save_interval"] == 2000
  assert agent["experiment_name"] == STAGES[stage][1]
  if stage == 2:
    assert Path(env["actions"]["joint_pos"]["tracker_file"]).resolve() == tracker


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu", type=int, default=1)
  args = parser.parse_args()
  os.chdir(ROOT)
  sys.path.insert(0, str(ROOT))
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  status = {
    "pipeline_pid": os.getpid(),
    "created_at": datetime.now().isoformat(),
    "state": "preflight",
    "gpu": args.gpu,
    "waist_action_rate": -0.04,
    "whole_body_action_rate": -0.02,
    "iterations_each_stage": 30000,
    "expected_final_iteration": 29999,
    "stages": {},
  }
  child = None

  def save():
    status["heartbeat"] = datetime.now().isoformat()
    temp = output / "status.json.tmp"
    temp.write_text(json.dumps(status, indent=2))
    temp.replace(output / "status.json")

  def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Pipeline interrupted by signal {signum}")

  signal.signal(signal.SIGTERM, interrupt)
  signal.signal(signal.SIGINT, interrupt)
  save()
  try:
    from mjlab.tasks.g1_pingpong.config import (
      WAIST_ACTION_RATE_WEIGHT_STAGE1,
      WAIST_ACTION_RATE_WEIGHT_STAGE2,
    )

    if (WAIST_ACTION_RATE_WEIGHT_STAGE1, WAIST_ACTION_RATE_WEIGHT_STAGE2) != (
      -0.04,
      -0.04,
    ):
      raise ValueError("Expected the agreed extra waist weight -0.04 in both stages")
    frozen_sources = source_hashes()
    (output / "source_sha256.json").write_text(json.dumps(frozen_sources, indent=2))
    tracker = None
    for stage in (1, 2):
      if source_hashes() != frozen_sources:
        raise RuntimeError(
          "Training source changed; refusing to mix experiment versions"
        )
      task, experiment = STAGES[stage]
      if experiment.split("/")[0] in (
        "g1_pingpong_receive_stage2_tele_gmr",
        "g1_pingpong_receive_stage2_tele_gmr_hard",
      ):
        raise RuntimeError("Refusing to write into a protected migrated experiment")
      run_name = f"g1_stage{stage}_waist_extra004_30k_4096_seed42_{output.name}"
      command = [
        sys.executable,
        "-u",
        "-m",
        "mjlab.scripts.train",
        task,
        "--gpu-ids",
        f"[{args.gpu}]",
        "--env.scene.num-envs",
        "4096",
        "--agent.max-iterations",
        "30000",
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
        run_name,
        "--agent.experiment-name",
        experiment,
      ]
      if stage == 2:
        command += ["--env.actions.joint-pos.tracker-file", str(tracker)]
      log = output / f"stage{stage}.log"
      entry = {"command": command, "console_log": str(log), "run_name": run_name}
      status["stages"][str(stage)] = entry
      status["state"] = f"stage{stage}_running"
      run = None
      offset = 0
      tail = ""
      validated_config = False
      with log.open("w") as stream:
        child = subprocess.Popen(
          command,
          cwd=ROOT,
          stdout=stream,
          stderr=subprocess.STDOUT,
          start_new_session=True,
          env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"},
        )
        entry["pid"] = child.pid
        save()
        print(f"Stage{stage} started: PID={child.pid}, log={log}", flush=True)
        while True:
          if run is None:
            matches = list((ROOT / "logs/rsl_rl" / experiment).glob(f"*_{run_name}"))
            if len(matches) > 1:
              raise RuntimeError("Ambiguous training run directory")
            if matches:
              run = matches[0]
              entry["run_dir"] = str(run)
          if run and (run / "params/agent.yaml").is_file() and not validated_config:
            validate_run_config(run, stage, tracker)
            validated_config = True
            entry["saved_configuration_validated"] = True
          with log.open() as reader:
            reader.seek(offset)
            text = tail + reader.read()
            offset = reader.tell()
          progress = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
          if progress:
            iteration, total = map(int, progress[-1])
            if entry.get("latest_iteration") != iteration:
              print(f"Stage{stage}: {iteration}/{total}", flush=True)
              entry["last_progress_at"] = datetime.now().isoformat()
            entry["latest_iteration"] = iteration
          tail = text[-512:]
          entry["console_bytes"] = offset
          entry["seconds_since_log_write"] = round(time.time() - log.stat().st_mtime, 1)
          entry["warning"] = (
            "No console update for over five minutes; process is still monitored"
            if entry["seconds_since_log_write"] > 300
            else None
          )
          code = child.poll()
          save()
          if code is not None:
            entry["exit_code"] = code
            if code != 0:
              raise RuntimeError(f"Stage{stage} exited with code {code}; see {log}")
            break
          time.sleep(15)
      if run is None or not validated_config:
        raise RuntimeError(f"Stage{stage} did not save a validated configuration")
      artifacts = validate_final_artifacts(
        run, 29999, 132 if stage == 1 else 186, 29 if stage == 1 else 30
      )
      entry["final_artifacts"] = artifacts
      entry["state"] = "complete"
      if stage == 1:
        tracker = Path(artifacts["jit"]).resolve()
      else:
        if sha256(run / "tracker.pt") != sha256(tracker):
          raise ValueError("Stage2 bundled tracker does not match the new Stage1")
      print(f"Stage{stage} completed and model_29999.pt/JIT validated", flush=True)
      save()
      child = None
    status["state"] = "complete"
    status["finished_at"] = datetime.now().isoformat()
    save()
    print("Both stages completed successfully", flush=True)
  except BaseException as exc:
    status["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    status["error"] = f"{type(exc).__name__}: {exc}"
    if child is not None and child.poll() is None:
      os.killpg(child.pid, signal.SIGINT)
      try:
        child.wait(timeout=20)
      except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    save()
    raise


if __name__ == "__main__":
  main()
