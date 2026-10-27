"""Reproduce nominal HOPE tracking and summarize actual receiving contacts.

This is a controlled MuJoCo diagnostic, not an Isaac/PhysX replacement: reset
to clip frame zero, no target/noise/domain randomization, exact root velocity.
The production receiving loop continues to use measured mocap odometry.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from deploy.backends.mujoco import MujocoBackend
from deploy.g1_pingpong.motions import RebasedMotion
from deploy.hope_g1.configuration import load_deployment
from deploy.hope_g1.controller import HopeActorController, HopeController
from deploy.hope_g1.deploy_agent import CONFIG_DIR


def nominal_tracking(deployment, runtime, options):
  engine = HopeController(deployment, runtime, options)
  dt = deployment.settings["control_dt"]
  results = []
  thresholds = deployment.settings["hope"]["racket"]
  for index, clip in enumerate(engine.library.motions):
    robot = MujocoBackend(copy.copy(deployment.model), deployment.settings)
    robot.reset(clip)
    reference = RebasedMotion(clip, *clip.sample(0)[1:])
    goal = engine.reference_goal(reference, index)
    actor = HopeActorController(deployment.settings, deployment.actor)
    strike = round(clip.strike_time(deployment.settings["ball"]) * clip.fps)
    for frame in range(strike + 1):
      state, now = robot.state(), frame * dt
      if frame == strike:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
          robot.model, robot.data, mujoco.mjtObj.mjOBJ_SITE, robot.racket, velocity, 0
        )
        pos_error = float(np.linalg.norm(state.racket_pos - goal.position))
        vel_error = float(np.linalg.norm(velocity[3:] - goal.velocity))
        angle = float(
          np.degrees(np.arccos(np.clip(state.racket_normal @ goal.normal, -1, 1)))
        )
        results.append(
          dict(
            motion_index=index,
            motion=deployment.motions[index].name,
            position_error_m=pos_error,
            velocity_error_m_s=vel_error,
            normal_error_degrees=angle,
            position_passed=pos_error < thresholds["strike_success_pos_thresh"],
            composite_passed=(
              pos_error < thresholds["strike_success_pos_thresh"]
              and vel_error < thresholds["strike_success_vel_thresh"]
              and angle < thresholds["strike_success_normal_thresh_deg"]
            ),
          )
        )
        break
      # Seed the same observation builder with instantaneous root-link
      # velocity for this diagnostic, rather than its production finite diff.
      actor.previous_time = now - dt
      actor.previous_position = (
        state.root_pos - dt * robot.data.qvel[robot.free_qvel : robot.free_qvel + 3]
      )
      target = actor.step(state, reference, frame, goal, (strike - frame) * dt, now)
      robot.set_target(target)
      for _ in range(deployment.settings["decimation"]):
        robot.substep()
  return dict(
    conditions="clip-frame-zero reset; nominal goal; exact root-link velocity; no AMP, mocap, noise, terrain or domain randomization",
    physics_dt=deployment.settings["physics_dt"],
    control_dt=dt,
    position_passed=sum(r["position_passed"] for r in results),
    composite_passed=sum(r["composite_passed"] for r in results),
    total=len(results),
    mean_position_error_m=float(np.mean([r["position_error_m"] for r in results])),
    motions=results,
  )


def receiving_metrics(path):
  path = Path(path)
  summary = json.loads((path / "summary.json").read_text())
  strikes = []
  rows = []
  for line in (path / "trace.jsonl").open():
    row = json.loads(line)
    rows.append(row)
    if (
      row.get("command_policy") == "hope_actor"
      and abs(row.get("time_to_strike", 1)) < 1e-6
    ):
      strikes.append(
        dict(
          time=row["time"],
          motion_index=row["motion_index"],
          position_error_m=float(
            np.linalg.norm(np.asarray(row["racket_pos"]) - row["goal_position"])
          ),
        )
      )
  hits, balls = summary["simulated_hits"], summary["simulated_balls"]
  timing = []
  events = summary["events"]
  dt = summary["simulation"]["policy_dt"]
  for selected in (e for e in events if e["event"] == "motion_selected"):
    finished = next(
      (
        e
        for e in events
        if e["event"] == "swing_finished" and e["time"] > selected["time"]
      ),
      None,
    )
    if finished is None:
      continue
    recovered = next(
      (e for e in events if e["event"] == "recovered" and e["time"] > finished["time"]),
      None,
    )
    swing = [
      r
      for r in rows
      if selected["time"] <= r["time"] < finished["time"]
      and r.get("state") == "EXECUTE"
    ]
    # Old recordings omit the frame-count/index fields. Keep their contact
    # metrics, but only claim a frame audit when the diagnostic fields exist.
    if not swing or "reference_frame_count" not in swing[0] or recovered is None:
      continue
    returning = [
      r
      for r in rows
      if finished["time"] <= r["time"] < recovered["time"]
      and r.get("state") == "RECOVER"
    ]
    count = swing[0]["reference_frame_count"]
    frames = [r["reference_frame"] for r in swing]
    return_indices = [r["return_frame_index"] for r in returning]
    strike_rows = [r for r in swing if abs(r["time_to_strike"]) < 1e-6]
    timing.append(
      dict(
        motion_index=selected["motion_index"],
        start_frame=selected["start_frame"],
        frame_count=count,
        commanded_frames=len(frames),
        consecutive_frames=frames == list(range(selected["start_frame"], count)),
        expected_hold_seconds=(count - selected["start_frame"]) * dt,
        actual_hold_seconds=finished["time"] - selected["time"],
        last_frame_hold_seconds=finished["time"] - swing[-1]["time"],
        return_commands=len(returning),
        consecutive_return_indices=return_indices == list(range(len(returning))),
        return_hold_seconds=recovered["time"] - finished["time"],
        strike_deadline_error_seconds=(strike_rows[0]["time"] - selected["deadline"])
        if strike_rows
        else None,
      )
    )
  return dict(
    path=str(path),
    hits=hits,
    balls=balls,
    hit_rate=hits / balls if balls else 0.0,
    physics=summary["physics_validation"],
    strike_errors=strikes,
    playback_timing=timing,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, default=CONFIG_DIR / "task.yaml")
  parser.add_argument("--receiving-run", type=Path, action="append", default=[])
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  config = yaml.safe_load(args.config.read_text())
  config.pop("task")
  options = yaml.safe_load((CONFIG_DIR / "rally.yaml").read_text())
  runtime = yaml.safe_load((CONFIG_DIR / "real.yaml").read_text())
  deployment = load_deployment(**config, ball_config=options["ball"])
  result = dict(
    checkpoint=str(deployment.checkpoint),
    checkpoint_sha256=hashlib.sha256(deployment.checkpoint.read_bytes()).hexdigest(),
    mujoco_version=mujoco.__version__,
    nominal_tracking=nominal_tracking(deployment, runtime, options),
    receiving=[receiving_metrics(p) for p in args.receiving_run],
    actual_robot_commands_sent=False,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2))
  print(
    json.dumps(
      dict(
        nominal_position_passed=result["nominal_tracking"]["position_passed"],
        nominal_composite_passed=result["nominal_tracking"]["composite_passed"],
        mean_position_error_m=result["nominal_tracking"]["mean_position_error_m"],
        receiving=[
          {k: r[k] for k in ("path", "hits", "balls", "hit_rate")}
          for r in result["receiving"]
        ],
        output=str(args.output),
      ),
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
