"""Replay recorded operator/ball inputs against a free MuJoCo plant (no SDK)."""

import argparse
import bisect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from deploy.g1_pingpong import simulation_runtime as sim
from deploy.g1_pingpong.configuration import load_deployment
from deploy.g1_pingpong.core import Controller, rotation
from deploy.g1_pingpong.motions import RebasedMotion
from deploy.g1_pingpong.perception import BallEstimate
from deploy.g1_pingpong.rally import RallyController


def run(
  recording,
  output,
  *,
  no_ball=False,
  duration=None,
  return_mode="fixed_world",
  return_blend=None,
  tracker_tail=False,
  terminal_tail=False,
):
  recording, output = Path(recording), Path(output)
  output.mkdir(parents=True, exist_ok=True)
  source = json.loads((recording / "summary.json").read_text())
  rows = [
    json.loads(line) for line in (recording / "trace.jsonl").read_text().splitlines()
  ]
  times = [row["time"] for row in rows]
  config = yaml.safe_load(Path("deploy/configs/g1_tele_gmr.yaml").read_text())
  deployment = load_deployment(**{k: v for k, v in config.items() if v is not None})
  runtime = dict(source["real_runtime_config"])
  runtime.pop("diagnostic_output_dir", None)
  cfg_path = output / "source_runtime.yaml"
  cfg_path.write_text(yaml.safe_dump(runtime))
  options = source.get("rally_runtime_config") or yaml.safe_load(
    Path("deploy/configs/mujoco.yaml").read_text()
  )
  if (
    "rally_runtime_config" not in source
    and "realtime-feed" not in source["software"]["runtime_revision"]
  ):
    options["ready_seconds"] = 0.5
  if terminal_tail:
    options["stage2_tail_mode"] = "observed_terminal"
  if return_blend is not None:
    options["return_blend_seconds"] = return_blend
  options_path = output / "source_rally.yaml"
  options_path.write_text(yaml.safe_dump(options))
  landings = [
    e["time"]
    for e in source["simulation"]["events"]
    if e["event"] == "operator_lowering_fixture"
  ]
  probe = dict(max_q_error_before_policy=0.0, first_q_error_over_1e_4=None)
  original_mocap, original_bind, original_record = (
    sim.MujocoRuntime.open_mocap,
    sim.MujocoRuntime.bind,
    sim.MujocoRuntime.record,
  )

  def source_row(now):
    return rows[max(0, min(len(rows) - 1, bisect.bisect_right(times, now + 1e-8) - 1))]

  def buttons(session):
    row = source_row(session.monotonic())
    return row["buttons"] if session.monotonic() <= times[-1] else 0

  def ball(now):
    row = source_row(now)
    if no_ball or row.get("ball_position") is None:
      return None
    age = (row.get("ball_age") or 0) + max(0, now - row["time"])
    if age > 0.1:
      return None
    return BallEstimate(
      np.array(row["ball_position"]),
      np.array(row["ball_velocity"]),
      row.get("ball_source_stamp", now),
      now - age,
    )

  def mocap(session, cfg):
    result = original_mocap(session, cfg)
    result.ball = SimpleNamespace(get=ball)
    session.continuous_feed = False
    return result

  def bind(session, engine):
    original_bind(session, engine)
    step = session.robot.substep

    def substep():
      if (
        landings
        and session.landing_started is None
        and session.monotonic() >= landings[0] - 1e-9
      ):
        session.landing_started = landings[0]
      step()

    session.robot.substep = substep

  def record(session, row):
    original_record(session, row)
    expected = source_row(session.monotonic())
    error = float(np.max(abs(np.array(row["q"]) - np.array(expected["q"]))))
    if row["state"] in (
      "WAIT",
      "POSITIONING",
      "POSE_HOLD",
      "STABLE",
      "TO_STABLE",
      "POLICY_POSTURE",
    ):
      probe["max_q_error_before_policy"] = max(
        probe["max_q_error_before_policy"], error
      )
    if error > 1e-4 and probe["first_q_error_over_1e_4"] is None:
      probe["first_q_error_over_1e_4"] = session.monotonic()

  original_step = RallyController.step

  def engine_step(engine, state, estimate, now):
    if return_mode != "fixed_world" and engine.lifecycle.state == "RECOVER":
      if getattr(engine, "_rebased_return", None) != engine.lifecycle.transition_start:
        engine._rebased_return = engine.lifecycle.transition_start
        _, point, quat = engine.home.sample(0)
        point = point.copy()
        point[:2] = state.anchor_pos[:2]
        source = engine.library.motions[engine.home_index]
        engine.home = RebasedMotion(
          source, point, state.anchor_quat if return_mode == "measured_xyyaw" else quat
        )
        engine.standing.motion = engine.home
        engine.standing.controller.motion = engine.home
    return original_step(engine, state, estimate, now)

  original_controller_step = Controller.step

  def controller_step(controller, state, ball=None, **kwargs):
    if (
      tracker_tail
      and controller.actor is not None
      and controller.t >= controller.motion.strike_time(controller.cfg["ball"])
    ):
      kwargs["tracker_only"] = True
    return original_controller_step(controller, state, ball, **kwargs)

  with (
    patch.object(Controller, "step", controller_step),
    patch.object(RallyController, "step", engine_step),
    patch.object(sim.MujocoRuntime, "buttons", buttons),
    patch.object(sim.MujocoRuntime, "open_mocap", mocap),
    patch.object(sim.MujocoRuntime, "bind", bind),
    patch.object(sim.MujocoRuntime, "record", record),
  ):
    result = sim.run_deployment(
      deployment,
      cfg_path,
      "deploy/configs/mocap_g1_live.yaml",
      simulation_config=options_path,
      motion_index=source["home_motion"],
      motion_select=source["motion_select"],
      motion_allow=source["motion_allow"],
      duration=duration or times[-1] + 0.02,
      episodes=1000,
      standing_only=True,
      seed=901,
      output=output,
    )
  replay = [json.loads(x) for x in (output / "trace.jsonl").read_text().splitlines()]
  falling = [
    r
    for r in replay
    if r.get("root_quat") is not None
    and np.degrees(np.arccos(np.clip(rotation(r["root_quat"])[2, 2], -1, 1))) > 45
  ]
  report = dict(
    source=str(recording),
    return_mode=return_mode,
    return_blend=return_blend,
    tracker_tail=tracker_tail,
    terminal_tail=terminal_tail,
    no_ball=no_ball,
    first_fall_time=falling[0]["time"] if falling else None,
    completed=result["completed_swings"],
    error=result["error"],
    **probe,
    actual_robot_commands_sent=False,
  )
  (output / "plant_replay_comparison.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report), flush=True)
  return report


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--recording", required=True, type=Path)
  p.add_argument("--output", required=True, type=Path)
  p.add_argument("--no-ball", action="store_true")
  p.add_argument("--duration", type=float)
  p.add_argument(
    "--return-mode",
    choices=["fixed_world", "measured_xy", "measured_xyyaw"],
    default="fixed_world",
  )
  p.add_argument("--return-blend", type=float)
  p.add_argument("--tracker-tail", action="store_true")
  p.add_argument("--terminal-tail", action="store_true")
  a = p.parse_args()
  torch.set_num_threads(1)
  run(
    a.recording,
    a.output,
    no_ball=a.no_ball,
    duration=a.duration,
    return_mode=a.return_mode,
    return_blend=a.return_blend,
    tracker_tail=a.tracker_tail,
    terminal_tail=a.terminal_tail,
  )


if __name__ == "__main__":
  main()
