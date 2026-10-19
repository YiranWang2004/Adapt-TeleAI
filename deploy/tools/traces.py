"""Capture inputs and replay them through deployment without a second rollout."""

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import numpy as np

from deploy.g1_pingpong.configuration import add_policy_arguments, from_arguments
from deploy.g1_pingpong.core import BallState, Controller, Motion, RobotState


def input_snapshot(controller, state, ball=None):
  row = {field.name: getattr(state, field.name).copy() for field in fields(RobotState)}
  row.update(
    reference_time=controller.t,
    reference_delta=controller.delta_dt,
    previous_joint_action=controller.previous_joint.copy(),
    previous_high_action=controller.previous_high.copy(),
  )
  if ball is not None:
    row.update(
      {"ball_" + field.name: getattr(ball, field.name) for field in fields(BallState)}
    )
  return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in row.items()}


def save_trace(path, rows, motion_name, stage):
  if not rows:
    raise ValueError("Cannot save an empty trace")
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  arrays: dict[str, Any] = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
  arrays.update(
    trace_version=np.array(2), motion_name=np.array(motion_name), stage=np.array(stage)
  )
  np.savez_compressed(path, **arrays)


def replay_inputs(deployment, reference, output):
  cfg = deployment.settings
  rows = []
  with np.load(reference, allow_pickle=False) as trace:
    if int(trace["trace_version"]) != 2 or int(trace["stage"]) != cfg["stage"]:
      raise ValueError("Trace version/stage differs from selected policy")
    name = str(trace["motion_name"])
    if name not in cfg["motions"]:
      raise ValueError("Trace motion is not in the selected policy")
    motion = Motion(
      deployment.motions[cfg["motions"].index(name)],
      cfg["robot"]["joint_names"],
      cfg["anchor"],
    )
    tracker, actor = deployment.tracker, deployment.actor
    controller = Controller(cfg, motion, tracker, actor)
    for i in range(len(trace["reference_time"])):
      state = RobotState(
        **{field.name: trace[field.name][i] for field in fields(RobotState)}
      )
      ball = (
        BallState(
          **{field.name: trace["ball_" + field.name][i] for field in fields(BallState)}
        )
        if actor
        else None
      )
      controller.t = float(trace["reference_time"][i])
      controller.delta_dt = float(trace["reference_delta"][i])
      controller.previous_joint = trace["previous_joint_action"][i].copy()
      controller.previous_high = trace["previous_high_action"][i].copy()
      row = input_snapshot(controller, state, ball)
      controller.step(state, ball)
      rows.append({**row, **controller.debug})
  save_trace(output, rows, name, cfg["stage"])
  return len(rows)


def capture_training(
  deployment, output, *, motion_index=0, steps=100, seed=42, device="cuda:0"
):
  """Optional reference capture; requires the training environment and a selected policy.

  This command is provided for later checkpoint acceptance, not run during the
  initial deployment build. Reference observations come from the real managers.
  """
  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.mdp.actions import JointPositionActionCfg
  from mjlab.sim import MujocoCfg
  from mjlab.tasks.a3_pingpong.actions import (
    FrozenTrackerAction,
    FrozenTrackerActionCfg,
    OrderedJointPositionAction,
  )
  from mjlab.tasks.a3_pingpong.ball import ReceivingBallCommand, ReceivingBallCommandCfg
  from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommand, MotionCommandCfg
  from mjlab.tasks.registry import load_env_cfg

  manifest = deployment.settings
  cfg = load_env_cfg(manifest["task"], play=True)
  cfg.scene.num_envs = 1
  cfg.seed = seed
  cfg.sim.mujoco = MujocoCfg(**manifest["sim_options"])
  motion_cfg = cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_files = tuple(str(path) for path in deployment.motions)
  motion_cfg.motion_file = motion_cfg.motion_directory = ""
  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  action_cfg.scale = dict(
    zip(
      manifest["robot"]["joint_names"], manifest["robot"]["action_scale"], strict=True
    )
  )
  if manifest["stage"] == 2:
    assert isinstance(action_cfg, FrozenTrackerActionCfg)
    action_cfg.tracker_file = str(deployment.tracker_path)
    action_cfg.min_speed = manifest["action"]["min_speed"]
    action_cfg.max_speed = manifest["action"]["max_speed"]
    action_cfg.residual_clip = manifest["action"]["residual_clip"]
    ball_cfg = cfg.commands["ball"]
    assert isinstance(ball_cfg, ReceivingBallCommandCfg)
    for field in fields(ReceivingBallCommandCfg):
      if field.name in manifest["ball"]:
        setattr(ball_cfg, field.name, manifest["ball"][field.name])
    ball_cfg.position_noise = ball_cfg.velocity_noise = 0.0
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  # Hard curricula must be evaluated at the saved iteration, not iteration 0.
  cast(Any, env).current_learning_iteration = manifest.get("iteration", 0)
  rows = []
  try:
    env.reset(seed=seed)
    robot = env.scene["robot"]
    motion = env.command_manager.get_term("motion")
    assert isinstance(motion, MotionCommand)
    ids = torch.zeros(1, device=device, dtype=torch.long)
    motion.motion_ids[:] = motion_index
    motion.reset_to_frame(ids, 0)
    env.action_manager.reset(ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    motion.update_relative_body_poses()
    if manifest["stage"] == 2:
      ball_command = env.command_manager.get_term("ball")
      assert isinstance(ball_command, ReceivingBallCommand)
      ball_command._resample_command(ids)
    env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    selected_policy = deployment.actor or deployment.tracker
    policy = selected_policy.model.to(device).eval()
    term = env.action_manager.get_term("joint_pos")
    assert isinstance(term, OrderedJointPositionAction)
    frozen = term if isinstance(term, FrozenTrackerAction) else None
    if manifest["stage"] == 2:
      assert frozen is not None
    anchor = robot.body_names.index(manifest["anchor"])
    racket = robot.find_sites(("right_racket_surface",), preserve_order=True)[0][0]

    def array(tensor):
      return tensor[0].detach().cpu().numpy().copy()

    for _ in range(steps):
      actor_obs = env.obs_buf["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      actor_obs = actor_obs.clone()
      tracker_obs = env.obs_buf["tracker"] if frozen else actor_obs
      assert isinstance(tracker_obs, torch.Tensor)
      tracker_obs = tracker_obs.clone()
      state = RobotState(
        array(robot.data.joint_pos),
        array(robot.data.joint_vel),
        array(robot.data.root_link_pos_w),
        array(robot.data.root_link_quat_w),
        array(robot.data.root_link_ang_vel_b),
        array(robot.data.body_link_pos_w[:, anchor]),
        array(robot.data.body_link_quat_w[:, anchor]),
        array(robot.data.site_pos_w[:, racket]),
        np.zeros(3),
      )
      row = {field.name: getattr(state, field.name) for field in fields(RobotState)}
      row.update(
        reference_time=float(motion.current_time_seconds(ids)[0]),
        reference_delta=float(motion.last_dt[0]),
        previous_joint_action=array(frozen.joint_action if frozen else term.raw_action),
        previous_high_action=array(term.raw_action)
        if manifest["stage"] == 2
        else np.zeros(30),
        tracker_obs=array(tracker_obs),
        actor_obs=array(actor_obs) if manifest["stage"] == 2 else np.array([]),
      )
      if manifest["stage"] == 2:
        ball = env.command_manager.get_term("ball")
        assert isinstance(ball, ReceivingBallCommand)
        row.update(
          ball_position=array(ball.pos_w),
          ball_velocity=array(ball.vel_w),
          ball_intercept=array(ball.intercept_w),
          ball_remaining=float((ball.intercept_time - ball.elapsed)[0]),
          ball_target=array(ball.target_w),
          ball_hit=bool(ball.hit[0]),
        )
      with torch.inference_mode():
        actions = policy(actor_obs)
        base = frozen.tracker(tracker_obs) if frozen else actions
      _, _, terminated, timed_out, _ = env.step(actions)
      if bool((terminated | timed_out)[0]):
        break  # Auto-reset changes action caches; exclude that final step.
      row.update(
        base_action=array(base),
        high_action=array(actions) if manifest["stage"] == 2 else np.zeros(30),
        joint_action=array(frozen.joint_action if frozen else term.raw_action),
        target_q=array(term._processed_actions),
        speed=float(frozen.speed[0]) if frozen else 1.0,
      )
      rows.append(row)
    save_trace(output, rows, manifest["motions"][motion_index], manifest["stage"])
  finally:
    env.close()
  return len(rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="mode", required=True)
  replay = subparsers.add_parser("replay")
  replay.add_argument("--reference", type=Path, required=True)
  capture = subparsers.add_parser("capture")
  capture.add_argument("--motion-index", type=int, default=0)
  capture.add_argument("--steps", type=int, default=100)
  capture.add_argument("--device", default="cuda:0")
  for command in (replay, capture):
    add_policy_arguments(command)
    command.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  deployment = from_arguments(args)
  if args.mode == "replay":
    count = replay_inputs(deployment, args.reference, args.output)
  else:
    count = capture_training(
      deployment,
      args.output,
      motion_index=args.motion_index,
      steps=args.steps,
      device=args.device,
    )
  print(f"Saved {count} samples to {args.output}")


if __name__ == "__main__":
  main()
