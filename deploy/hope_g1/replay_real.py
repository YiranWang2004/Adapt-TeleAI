"""Replay measured HOPE ball packets through the live estimator/planner, no IO.

Evaluate at the recorded trigger epochs; this is prediction validation, not a
counterfactual robot success rate. Recorded motor trajectories are never altered.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from deploy.g1_pingpong.perception import BallEstimator
from deploy.hope_g1.a3_planner import A3Planner


def load_ball_samples(path):
  if path.suffix == ".npz":
    return np.load(path)["samples"][:, :4]
  samples = {}
  with path.open() as stream:
    for line in stream:
      record = json.loads(line)
      packet = record["packet"]
      if packet["kind"] == "ball":
        samples[float(packet["stamp"])] = packet["position"]
  return np.asarray([[stamp, *position] for stamp, position in sorted(samples.items())])


def replay(summary, samples, rally, mocap):
  events = [e for e in summary["events"] if e["event"] == "motion_selected"]
  fit = BallEstimator(**mocap["estimator"])
  planner = A3Planner(rally["ball"], rally["a3"])
  base = samples[0, 0]
  index = 0
  results = []
  for event in events:
    old = event["raw_command"]
    stamp = old["estimate_stamp"]
    while index < len(samples) and samples[index, 0] <= stamp + 1e-7:
      sample = samples[index]
      fit.update(sample[1:4], sample[0], received=sample[0] - base)
      index += 1
    now = stamp - base + old["age_seconds"]
    estimate = fit.get(now)
    plane = old["hit_plane_x"]
    command = planner.plan(estimate, now, plane) if estimate else None
    window = samples[(samples[:, 0] >= stamp) & (samples[:, 0] <= stamp + 1.5)]
    crossing = None
    for a, b in zip(window, window[1:], strict=False):
      if a[1] > plane >= b[1] and b[0] - a[0] < 0.05:
        fraction = (a[1] - plane) / (a[1] - b[1])
        crossing = a + fraction * (b - a)
        break
    item = dict(
      motion_index=event["motion_index"],
      estimate_stamp=stamp,
      recorded_position=old["position"],
      recorded_remaining=old["remaining"],
      fit_samples=len(fit.samples),
      replay_valid=command is not None,
      observed_crossing=crossing.tolist() if crossing is not None else None,
    )
    if command:
      remaining = command.deadline - now
      item.update(
        replay_position=command.position.tolist(),
        replay_remaining=remaining,
        replay_velocity_estimate=estimate.velocity.tolist(),
        predicted_bounces=command.diagnostic["predicted_bounces"],
        within_trigger_window=bool(
          0 < remaining <= rally["a3"]["max_command_time_to_strike"]
        ),
      )
      if crossing is not None:
        actual_remaining = crossing[0] - stamp - old["age_seconds"]
        item.update(
          recorded_position_error_m=float(
            np.linalg.norm(np.asarray(old["position"]) - crossing[1:4])
          ),
          replay_position_error_m=float(
            np.linalg.norm(command.position - crossing[1:4])
          ),
          recorded_time_error_s=old["remaining"] - actual_remaining,
          replay_time_error_s=remaining - actual_remaining,
        )
    results.append(item)
  metrics = {}
  for key in (
    "recorded_position_error_m",
    "replay_position_error_m",
    "recorded_time_error_s",
    "replay_time_error_s",
  ):
    values = [abs(row[key]) for row in results if key in row]
    if values:
      metrics[key] = dict(median=float(np.median(values)), max=float(max(values)))
  return dict(
    world_frame="hope_table",
    physics_profile=rally["physics_profile"],
    robot_commands_sent=False,
    validation_scope="Prediction at original trigger epochs; not robot contact/return validation",
    rally_config=rally,
    mocap_config=mocap,
    metrics=metrics,
    swings=results,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--summary", type=Path, required=True)
  parser.add_argument(
    "--ball-samples",
    type=Path,
    required=True,
    help="mocap_packets.jsonl or raw_ball_samples.npz, already in HOPE table world",
  )
  parser.add_argument(
    "--rally-config", type=Path, default=Path("deploy/hope_g1/configs/rally.yaml")
  )
  parser.add_argument(
    "--mocap-config", type=Path, default=Path("deploy/hope_g1/configs/mocap.yaml")
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = replay(
    json.loads(args.summary.read_text()),
    load_ball_samples(args.ball_samples),
    yaml.safe_load(args.rally_config.read_text()),
    yaml.safe_load(args.mocap_config.read_text()),
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2))
  print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
  main()
