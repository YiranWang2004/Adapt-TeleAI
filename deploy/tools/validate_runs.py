"""Direct checkpoint evaluation and training-input replay acceptance."""

import argparse
import contextlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from deploy.g1_pingpong.commands import CommandGuard
from deploy.g1_pingpong.configuration import load_deployment
from deploy.tools.compare_traces import compare
from deploy.tools.evaluate import rollout_summary, run_mujoco
from deploy.tools.traces import capture_training, replay_inputs


def audit_guard_traces(deployment, directory, real_config="deploy/configs/real.yaml"):
  """Evaluate the configured real command guard on saved sim observations."""
  cfg = yaml.safe_load(Path(real_config).read_text())
  rejected, reasons, episodes = [], Counter(), 0
  limits = np.asarray(deployment.settings["robot"]["joint_limits"])
  target_outside_limits = 0
  for path in sorted(Path(directory).glob("trace_*.npz")):
    guard = CommandGuard(limits, cfg["guard"])
    episodes += 1
    with np.load(path, allow_pickle=False) as trace:
      targets = trace["target_q"]
      if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
        target_outside_limits += 1
      for step, (target, q, dq) in enumerate(
        zip(targets, trace["q"], trace["dq"], strict=True)
      ):
        try:
          guard.check(target, q, dq, deployment.settings["control_dt"])
        except ValueError as exc:
          reason = str(exc)
          reasons[reason.split(":")[0]] += 1
          rejected.append({"trace": path.name, "step": step, "error": reason})
          break
  return {
    "episodes": episodes,
    "episodes_rejected": len(rejected),
    "episodes_with_out_of_range_targets": target_outside_limits,
    "passed_all_guards": episodes > 0 and not rejected,
    "first_rejection_reasons": dict(reasons),
    "rejections": rejected,
    "note": "Desired PD targets checked; this does not imply measured joints exceeded limits.",
  }


def resolve_checkpoint(path):
  path = Path(path).resolve()
  if path.is_file():
    return path
  candidates = [
    candidate
    for candidate in path.rglob("model_*.pt")
    if candidate.stem.removeprefix("model_").isdigit()
    and (candidate.parent / "params/env.yaml").is_file()
    and (candidate.parent / "params/agent.yaml").is_file()
    and (candidate.parent / "tracker.pt").is_file()
  ]
  if not candidates:
    raise FileNotFoundError(f"No complete checkpoint run under {path}")
  return max(
    candidates, key=lambda p: (int(p.stem.removeprefix("model_")), p.stat().st_mtime)
  )


def validate_runs(
  baseline,
  hard,
  output,
  *,
  motion_dir=None,
  episodes=4,
  seed=42,
  reference_steps=80,
  capture_device="cpu",
  warp_cache="/tmp/adapt-teleai-warp",
):
  if episodes < 1 or reference_steps < 0:
    raise ValueError("episodes must be positive; reference_steps must be nonnegative")
  output = Path(output).resolve()
  output.mkdir(parents=True, exist_ok=False)
  torch.set_num_threads(1)
  report = {
    "seed": seed,
    "episodes_per_motion": episodes,
    "device": "cpu",
    "torch_threads": 1,
    "observation_noise": False,
    "robot_commands_sent": False,
    "experiments": {},
  }
  for name, source, task in (
    ("tele_gmr", baseline, "Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive"),
    ("tele_gmr_hard", hard, "Mjlab-PingPong-Receive-G1-TeleGMR-Hard-Stage2-Adaptive"),
  ):
    checkpoint = resolve_checkpoint(source)
    destination = output / name
    destination.mkdir()
    with (
      (destination / "run.log").open("w") as stream,
      contextlib.redirect_stdout(stream),
    ):
      deployment = load_deployment(
        checkpoint,
        task=task,
        tracker=checkpoint.parent / "tracker.pt",
        motion_dir=motion_dir,
      )
      rollouts = run_mujoco(
        deployment,
        all_motions=True,
        episodes=episodes,
        seed=seed,
        output=destination / "rollouts",
      )
      reference_report = None
      comparison = None
      if reference_steps:
        import warp as wp

        wp.config.kernel_cache_dir = warp_cache
        reference = destination / "training_reference.npz"
        replayed = destination / "training_replayed.npz"
        count = capture_training(
          deployment, reference, steps=reference_steps, seed=seed, device=capture_device
        )
        replay_inputs(deployment, reference, replayed)
        comparison = compare(
          reference,
          replayed,
          [
            "tracker_obs",
            "actor_obs",
            "base_action",
            "high_action",
            "joint_action",
            "target_q",
            "reference_time",
            "reference_delta",
          ],
        )
        reference_report = {
          "samples": count,
          "device": capture_device,
          "fields": comparison,
        }
        (destination / "comparison.json").write_text(
          json.dumps(reference_report, indent=2)
        )
    row = {
      "checkpoint": str(checkpoint),
      "policy": deployment.report(),
      "rollouts": rollout_summary(rollouts),
      "training_input_replay": reference_report,
      "real_guard_audit": audit_guard_traces(deployment, destination / "rollouts"),
    }
    report["experiments"][name] = row
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    print(name, json.dumps(row["rollouts"]), flush=True)
    if comparison is not None and not all(
      field["passed"] for field in comparison.values()
    ):
      raise RuntimeError(f"{name}: deployment replay differs from training inputs")
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--baseline",
    type=Path,
    default=Path("logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr"),
  )
  parser.add_argument(
    "--hard",
    type=Path,
    default=Path("logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr_hard"),
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--motion-dir", type=Path, default=Path("dataset/g1_pingpong_tele_gmr/motions")
  )
  parser.add_argument("--episodes", type=int, default=4)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--reference-steps", type=int, default=80)
  parser.add_argument("--capture-device", default="cpu")
  args = parser.parse_args()
  validate_runs(
    args.baseline,
    args.hard,
    args.output,
    motion_dir=args.motion_dir,
    episodes=args.episodes,
    seed=args.seed,
    reference_steps=args.reference_steps,
    capture_device=args.capture_device,
  )


if __name__ == "__main__":
  main()
