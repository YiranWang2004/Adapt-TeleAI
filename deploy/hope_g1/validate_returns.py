"""Audit recorded physical returns without changing actor goals or ball dynamics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from deploy.hope_g1.deploy_agent import CONFIG_DIR


def first_flight(position, velocity, cfg):
  """Diagnostic ballistic projection only; never used by the controller."""
  p, v = np.asarray(position), np.asarray(velocity)
  plane = cfg["table_height"] + cfg["ball_radius"]
  gravity = cfg["gravity"]
  discriminant = v[2] ** 2 + 2 * gravity * (p[2] - plane)
  if discriminant <= 0:
    return dict(legal_return=False, reason="no_downward_table_height_crossing")
  t = (v[2] + np.sqrt(discriminant)) / gravity
  if t <= 0:
    return dict(legal_return=False, reason="already_below_table_and_descending")
  landing = p + v * t + np.array([0, 0, -0.5 * gravity * t**2])
  net_x = (cfg["table_near_x"] + cfg["table_far_x"]) / 2
  net_t = (net_x - p[0]) / v[0] if v[0] > 0 else -1
  crossing = p + v * net_t + np.array([0, 0, -0.5 * gravity * net_t**2])
  clear = bool(
    0 < net_t < t
    and abs(crossing[1]) <= cfg["table_width"] / 2
    and crossing[2] > cfg["table_height"] + cfg["net_height"] + cfg["ball_radius"]
  )
  on_opponent = bool(
    net_x < landing[0] <= cfg["table_far_x"]
    and abs(landing[1]) <= cfg["table_width"] / 2
  )
  return dict(
    legal_return=clear and on_opponent, landing=landing.tolist(), net_clear=clear
  )


def summarize(path, cfg):
  path = Path(path)
  summary = json.loads((path / "summary.json").read_text())
  rows = [json.loads(line) for line in (path / "trace.jsonl").open()]
  feeds = {
    e["index"]: e["time"]
    for e in summary["simulation"]["events"]
    if e["event"] == "simulated_ball"
  }
  balls = []
  for ball in summary["ball_returns"]:
    result = {k: v for k, v in ball.items() if k != "trajectory"}
    contact = ball["contact"]
    if contact:
      now = feeds[ball["feed_id"]] + contact["time"]
      row = min(
        (r for r in rows if r.get("command_policy") == "hope_actor"),
        key=lambda r: abs(r["time"] - now),
      )
      incoming = np.asarray(contact["incoming_velocity"])
      normal = np.asarray(row["goal_normal"])
      normal /= np.linalg.norm(normal)
      velocity = np.asarray(row["goal_velocity"])
      ideal = (
        incoming
        - (1 + cfg["racket_restitution"]) * ((incoming - velocity) @ normal) * normal
      )
      measured_normal = np.asarray(contact["racket_normal"])
      result.update(
        selected_motion=row["motion_index"],
        goal_velocity=velocity.tolist(),
        goal_normal=normal.tolist(),
        plane_angle_error_degrees=float(
          np.degrees(np.arccos(np.clip(abs(normal @ measured_normal), 0, 1)))
        ),
        ideal_tracking_outgoing_velocity=ideal.tolist(),
        ideal_tracking_projection=first_flight(contact["position"], ideal, cfg),
      )
    balls.append(result)
  return dict(
    path=str(path),
    seed=summary["simulation"]["seed"],
    served=summary["simulated_balls"],
    hits=summary["simulated_hits"],
    legal_returns=summary["simulated_legal_returns"],
    return_rate=summary["legal_return_rate"],
    gate_passed=summary["passed"],
    flow_passed=summary["flow_passed"],
    physics_validation=summary["physics_validation"],
    outcomes=dict(Counter(b["outcome"] for b in balls)),
    balls=balls,
  )


def plot_runs(paths, cfg, output):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Rectangle

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
  near, far, width, height = (
    cfg[k] for k in ("table_near_x", "table_far_x", "table_width", "table_height")
  )
  net_x = (near + far) / 2
  axes[0].add_patch(
    Rectangle(
      (near, -width / 2), far - near, width, facecolor="#d9e9e0", edgecolor="black"
    )
  )
  axes[0].plot([net_x, net_x], [-width / 2, width / 2], "k--", label="Net")
  axes[1].plot([near, far], [height, height], "k-", label="Table")
  axes[1].plot([net_x, net_x], [height, height + cfg["net_height"]], "k--", label="Net")
  labels = set()
  for path in paths:
    summary = json.loads((Path(path) / "summary.json").read_text())
    for ball in summary["ball_returns"]:
      landing = ball["first_landing_plane_crossing"]
      trajectory = [
        s for s in ball["trajectory"] if landing is None or s["time"] <= landing["time"]
      ]
      if not trajectory:
        continue
      points = np.asarray([s["position"] for s in trajectory])
      color = "#18864b" if ball["legal_return"] else "#c14b45"
      label = "Legal return" if ball["legal_return"] else "Failed return"
      for axis, coordinate in zip(axes, [1, 2], strict=True):
        axis.plot(
          points[:, 0],
          points[:, coordinate],
          color=color,
          alpha=0.45,
          label=label if label not in labels else None,
        )
        axis.scatter(points[0, 0], points[0, coordinate], s=9, color="black")
        if landing:
          p = landing["position"]
          axis.scatter(p[0], p[coordinate], s=15, color=color, marker="x")
      labels.add(label)
  axes[0].set(
    xlabel="Court x (m)",
    ylabel="Court y (m)",
    title="Top view: outgoing first flights",
    xlim=(-0.2, 5.2),
    ylim=(-2.5, 2.5),
  )
  axes[1].set(
    xlabel="Court x (m)",
    ylabel="Ball centre z (m)",
    title="Side view: outgoing first flights",
    xlim=(-0.2, 5.2),
    ylim=(0, 2.7),
  )
  for axis in axes:
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
  fig.suptitle(
    "Measured MuJoCo returns; black dots = racket contact; crosses = first table-height crossing"
  )
  fig.tight_layout()
  fig.savefig(output, dpi=160)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--receiving-run", type=Path, action="append", required=True)
  parser.add_argument("--rally-config", type=Path, default=CONFIG_DIR / "rally.yaml")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  cfg = yaml.safe_load(args.rally_config.read_text())["ball"]
  if cfg["drag"] != 0:
    parser.error("The diagnostic nominal projection currently requires drag=0")
  runs = [summarize(path, cfg) for path in args.receiving_run]
  balls = [ball for run in runs for ball in run["balls"]]
  served = sum(run["served"] for run in runs)
  result = dict(
    criterion="racket contact, then clear net, then first bounce within opponent table; no return net contact",
    actual_robot_commands_sent=False,
    ball_model=cfg,
    served=served,
    hits=sum(run["hits"] for run in runs),
    legal_returns=sum(run["legal_returns"] for run in runs),
    outcomes=dict(Counter(ball["outcome"] for ball in balls)),
    intended_motion_counts=dict(
      Counter(str(ball["feed"]["motion_index"]) for ball in balls)
    ),
    ideal_tracking_projection_passes=sum(
      ball.get("ideal_tracking_projection", {}).get("legal_return", False)
      for ball in balls
    ),
    nominal_projection_note="Offline thought experiment assuming exact target velocity/normal; not a controller or measured result",
    runs=runs,
  )
  args.output.mkdir(parents=True, exist_ok=True)
  (args.output / "report.json").write_text(json.dumps(result, indent=2))
  plot_runs(args.receiving_run, cfg, args.output / "return_trajectories.png")
  print(
    json.dumps(
      {
        key: value for key, value in result.items() if key not in ("runs", "ball_model")
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
