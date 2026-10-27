"""Fit A3 ball physics from measured flights/bounces, without actor/contact scores.

Use the first N recorded rallies for fitting and reserve the others for prediction
validation. Local polynomial derivatives suppress camera differentiation noise.
Restitution is the median observed pre/post velocity ratio, as in A3 calibration.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from deploy.hope_g1.replay_real import load_ball_samples, replay


def bounce_indices(samples):
  z, t = samples[:, 3], samples[:, 0]
  result = []
  for i in np.flatnonzero((z[1:-1] < z[:-2]) & (z[1:-1] < z[2:])) + 1:
    if not -0.04 <= z[i] <= 0.04:
      continue
    before = samples[(t >= t[i] - 0.06) & (t <= t[i] - 0.01)]
    after = samples[(t >= t[i] + 0.01) & (t <= t[i] + 0.06)]
    if min(len(before), len(after)) < 4:
      continue
    v0 = np.polyfit(before[:, 0] - t[i], before[:, 1:4], 1)[0]
    v1 = np.polyfit(after[:, 0] - t[i], after[:, 1:4], 1)[0]
    if v0[2] < -0.4 and v1[2] > 0.4:
      result.append((i, v0, v1))
  return result


def fit_physics(events, samples):
  drag = []
  horizontal, vertical, diagnostics = [], [], []
  # A 21-point cubic local fit on a 180 Hz resampling grid; no learned policy.
  grid = np.arange(-10, 11) / 180
  inverse = np.linalg.pinv(np.column_stack([grid**i for i in range(4)]))
  for event in events:
    stamp = event["raw_command"]["estimate_stamp"]
    segment = samples[(samples[:, 0] >= stamp - 0.6) & (samples[:, 0] <= stamp + 0.8)]
    if len(segment) < 40:
      raise ValueError("Insufficient measured flight samples")
    relative = segment[:, 0] - segment[0, 0]
    t = np.arange(0, relative[-1], 1 / 180)
    xyz = np.column_stack([np.interp(t, relative, segment[:, i]) for i in range(1, 4)])
    windows = np.lib.stride_tricks.sliding_window_view(xyz, 21, axis=0)
    velocity = windows @ inverse[1]
    acceleration = 2 * (windows @ inverse[2])
    positions, times = xyz[10:-10], t[10:-10]
    speed = np.linalg.norm(velocity, axis=1)
    mask = (
      (positions[:, 2] > 0.04)
      & (speed > 1)
      & (positions[:, 0] > -0.25)
      & (positions[:, 0] < 2.6)
    )
    bounces = bounce_indices(segment)
    for index, v0, v1 in bounces:
      mask &= abs(times - relative[index]) > 0.1
      horizontal.append(float(np.linalg.norm(v1[:2]) / np.linalg.norm(v0[:2])))
      vertical.append(float(abs(v1[2] / v0[2])))
    design = -speed[:, None] * velocity
    residual = acceleration + [0.0, 0.0, 9.81]
    k = float(np.sum(design[mask] * residual[mask]) / np.sum(design[mask] ** 2))
    if np.isfinite(k) and k >= 0:
      drag.append(k)
    diagnostics.append(
      dict(
        source_stamp=stamp,
        flight_samples=int(mask.sum()),
        drag=k,
        bounce_count=len(bounces),
      )
    )
  if not drag or not horizontal or not vertical:
    raise ValueError("Insufficient flights/bounces for calibration")
  return dict(
    drag=float(np.median(drag)),
    table_tangential_damping=float(np.median(horizontal)),
    table_restitution=float(np.median(vertical)),
  ), diagnostics


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--summary", type=Path, required=True)
  parser.add_argument("--ball-samples", type=Path, required=True)
  parser.add_argument("--train-swings", type=int, default=3)
  parser.add_argument(
    "--base-rally",
    type=Path,
    default=Path("deploy/hope_g1/configs/rally_a3_reference.yaml"),
  )
  parser.add_argument(
    "--mocap-config", type=Path, default=Path("deploy/hope_g1/configs/mocap.yaml")
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--output-rally", type=Path, required=True)
  args = parser.parse_args()
  summary = json.loads(args.summary.read_text())
  events = [e for e in summary["events"] if e["event"] == "motion_selected"]
  if not 2 <= args.train_swings < len(events):
    parser.error("Reserve at least one rally for validation; fit at least two")
  samples = load_ball_samples(args.ball_samples)
  physics, diagnostics = fit_physics(events[: args.train_swings], samples)
  rally = yaml.safe_load(args.base_rally.read_text())
  rally["ball"].update(physics)
  rally["physics_calibration"] = dict(
    source_recording=Path(
      summary.get("real_runtime_config", {}).get(
        "diagnostic_output_dir", args.summary.parent
      )
    ).name,
    method="measured flights and bounce velocity ratios",
    fitting_swings=args.train_swings,
    validation_swings=len(events) - args.train_swings,
    actor_and_hit_outcomes_used=False,
  )
  result = replay(
    summary, samples, rally, yaml.safe_load(args.mocap_config.read_text())
  )
  result.update(
    fitted_physics=physics,
    flight_diagnostics=diagnostics,
    fitting_swings=args.train_swings,
    validation_swings=len(events) - args.train_swings,
  )
  # Report held-out prediction quality independently of the fitting rallies.
  held = result["swings"][args.train_swings :]
  result["held_out_metrics"] = {
    key: dict(
      median=float(np.median([abs(r[key]) for r in held if key in r])),
      max=float(max(abs(r[key]) for r in held if key in r)),
    )
    for key in (
      "recorded_position_error_m",
      "replay_position_error_m",
      "recorded_time_error_s",
      "replay_time_error_s",
    )
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2))
  args.output_rally.write_text(yaml.safe_dump(rally, sort_keys=False))
  print(
    json.dumps({k: result[k] for k in ("fitted_physics", "held_out_metrics")}, indent=2)
  )


if __name__ == "__main__":
  main()
