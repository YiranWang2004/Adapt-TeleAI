"""Continuous recovery diagnostic in the actual train/play environment.

Only command scheduling and termination handling differ from ordinary play.
Robot dynamics, observations, frozen tracker and residual actions are unchanged.
Each environment is initialized once at home; reference switches never write
robot state or clear action history. This is a diagnostic, not a trained skill.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.tasks.a3_pingpong.actions import FrozenTrackerActionCfg
from mjlab.tasks.adapt_tennis.mdp.commands import _resolve_motion_paths


def selected_motions(spec, count):
  ids = list(range(count)) if spec == "all" else [int(x) for x in spec.split(",")]
  if not ids or len(set(ids)) != len(ids) or any(x < 0 or x >= count for x in ids):
    raise ValueError(f"Expected unique motion indices in [0, {count})")
  return ids


def configure_cycle(env_cfg, cfg):
  if cfg.cycle_repeats < 1:
    raise ValueError("cycle-repeats must be positive")
  if cfg.cycle_repeats > 1 and (cfg.video or cfg.video_headless):
    raise ValueError(
      "Record one cycle at a time; repeated trials are headless without video"
    )
  if cfg.agent != "trained" or cfg.no_terminations or cfg.motion_warmup_steps:
    raise ValueError(
      "standing-cycle requires trained policy and its own phase scheduling"
    )
  if not cfg.checkpoint_file:
    raise ValueError(
      "standing-cycle requires --checkpoint-file for reproducible provenance"
    )
  if cfg.onnx_file or cfg.collect_dof_pos_n or cfg.collect_per_motion_directory:
    raise ValueError("standing-cycle cannot be combined with ONNX or collection mode")
  if not isinstance(env_cfg.actions.get("joint_pos"), FrozenTrackerActionCfg):
    raise ValueError("standing-cycle requires a frozen-tracker Stage-2 task")
  paths = _resolve_motion_paths(env_cfg.commands["motion"])
  ids = selected_motions(cfg.cycle_motions, len(paths))
  if not 0 <= cfg.cycle_home < len(paths):
    raise ValueError("cycle-home outside motion library")
  if cfg.num_envs is not None and cfg.num_envs != len(ids):
    raise ValueError("standing-cycle uses one environment per selected motion")
  env_cfg.scene.num_envs = len(ids)
  env_cfg.seed = cfg.cycle_seed
  # Preserve physical continuity beyond ball_finished / motion end. Explicit
  # physical failure checks and phase deadlines below replace auto-reset.
  env_cfg.terminations = {}
  for command in env_cfg.commands.values():
    command.resampling_time_range = (1e9, 1e9)


def stability_metrics(robot, anchor, home_height):
  quat = robot.data.root_link_quat_w
  tilt = torch.rad2deg(torch.acos((1 - 2 * (quat[:, 1:3] ** 2).sum(-1)).clamp(-1, 1)))
  return torch.stack(
    (
      robot.data.joint_vel.abs().amax(-1),
      robot.data.root_link_ang_vel_b.abs().amax(-1),
      tilt,
      torch.linalg.vector_norm(robot.data.body_link_lin_vel_w[:, anchor], dim=-1),
      (robot.data.body_link_pos_w[:, anchor, 2] - home_height).abs(),
    ),
    dim=-1,
  )


@torch.inference_mode()
def run_cycle(env, policy, cfg, task_id, *, viewer=None):
  uw = env.unwrapped
  device, dt = uw.device, float(uw.step_dt)
  motion = uw.command_manager.get_term("motion")
  ball = uw.command_manager.get_term("ball")
  robot = uw.scene["robot"]
  paths = _resolve_motion_paths(motion.cfg)
  selected = selected_motions(cfg.cycle_motions, len(paths))
  count = len(selected)
  ids = torch.arange(count, device=device)
  swings = torch.tensor(selected, device=device)
  # PREPARE=0, SWING=1, RETURN=2, HOLD=3, PASS=4, FAIL=5.
  phase = torch.zeros(count, dtype=torch.long, device=device)
  phase_start = torch.zeros(count, device=device)
  stable_time = torch.zeros(count, device=device)
  limits = torch.tensor([0.1, 0.1, 10.0, 0.05, 0.1], device=device)
  names = ["PREPARE", "SWING", "RETURN", "HOLD", "PASS", "FAIL"]
  rows = [
    {
      "motion_index": index,
      "motion": str(paths[index]),
      "events": [{"time": 0.0, "phase": "PREPARE"}],
    }
    for index in selected
  ]
  output = Path(cfg.cycle_output)
  output.mkdir(parents=True, exist_ok=True)

  uw.reset(seed=cfg.cycle_seed)
  motion.motion_ids[:] = cfg.cycle_home
  motion.reset_to_frame(ids, 0)  # The only robot state initialization in the rollout.
  uw.action_manager.reset(ids)
  uw.episode_length_buf.zero_()
  uw.scene.write_data_to_sim()
  uw.sim.forward()
  uw.sim.sense()
  motion.update_relative_body_poses()
  anchor = robot.body_names.index(motion.cfg.anchor_body_name)
  home_height = motion.body_pos_w[
    :, motion.cfg.body_names.index(motion.cfg.anchor_body_name), 2
  ].clone()
  # Use MotionCommand's existing freeze path after every simulation step.
  original_warmup_mask = motion._motion_warmup_mask
  motion._motion_warmup_mask = lambda: phase != 1
  motion._freeze_motion_timeline_at_start()
  uw.obs_buf = uw.observation_manager.compute(update_history=True)
  history = {
    key: []
    for key in ("time", "phase", "metrics", "reference_time", "joint_pos", "root_pos")
  }

  def change(mask, new_phase, now, reason=None):
    changed = mask.nonzero().flatten()
    for index in changed.tolist():
      event = {"time": round(now, 4), "phase": names[new_phase]}
      if reason:
        event["reason"] = reason
      if new_phase in (4, 5):
        rows[index]["hit"] = bool(ball.hit[index])
        rows[index]["return_success"] = bool(ball.success[index])
      rows[index]["events"].append(event)
      print(
        f"[CYCLE] motion={selected[index]} t={now:.2f} {names[new_phase]} {reason or ''}",
        flush=True,
      )
    phase[changed] = new_phase
    phase_start[changed] = now
    stable_time[changed] = 0
    return changed

  stop_reason = "40s overall deadline"
  with torch.inference_mode():
    for step in range(int(40 / dt)):
      frame_started = time.monotonic()
      obs = TensorDict(uw.obs_buf, batch_size=[count])
      actions = policy(obs)
      # Zero high-level residual outside SWING leaves the real frozen tracker
      # running with its previous joint action and a frozen frame-zero reference.
      actions[phase != 1] = 0
      env.step(actions)
      now = (step + 1) * dt
      metrics = stability_metrics(robot, anchor, home_height)
      stable = (metrics <= limits).all(-1) & torch.isfinite(metrics).all(-1)
      stable_time = torch.where(stable, stable_time + dt, 0.0)
      live = phase < 4
      bad = live & (
        (metrics[:, 2] > 45) | (metrics[:, 4] > 0.25) | ~torch.isfinite(metrics).all(-1)
      )
      change(bad, 5, now, "tilt>45deg / anchor height error>0.25m / nonfinite")
      history["time"].append(now)
      history["phase"].append(phase.cpu().numpy().copy())
      history["metrics"].append(metrics.cpu().numpy().copy())
      history["reference_time"].append(
        motion.current_time_seconds().cpu().numpy().copy()
      )
      history["joint_pos"].append(robot.data.joint_pos.cpu().numpy().copy())
      history["root_pos"].append(robot.data.root_link_pos_w.cpu().numpy().copy())

      # Final hold must be continuously stable for five seconds, not just end
      # upright; a loss of stability is a failed acceptance trial.
      change((phase == 3) & ~stable, 5, now, "stability lost during final hold")
      change((phase == 3) & (now - phase_start >= 5), 4, now)
      change((phase == 2) & (now - phase_start >= 3) & (stable_time >= 1), 3, now)
      elapsed = now - phase_start
      change(
        ((phase == 0) | (phase == 2)) & (elapsed >= 15), 5, now, "standing timeout"
      )

      duration = (
        motion.motion.lengths[motion.motion_ids] - 1
      ) / motion.motion.fps_values[motion.motion_ids]
      returning = change(
        (phase == 1) & (motion.current_time_seconds() >= duration), 2, now
      )
      motion.motion_ids[returning] = cfg.cycle_home
      if len(returning):
        motion._freeze_motion_timeline_at_start(returning)
      starting = change(
        (phase == 0) & (now - phase_start >= 3) & (stable_time >= 1), 1, now
      )
      motion.motion_ids[starting] = swings[starting]
      if len(starting):
        motion._freeze_motion_timeline_at_start(starting)
        ball._resample_command(starting)  # Ball only; never reset robot.
        for index in starting.tolist():
          rows[index]["incoming_ball"] = {
            "position": ball.pos_w[index].cpu().tolist(),
            "velocity": ball.vel_w[index].cpu().tolist(),
            "intercept": ball.intercept_w[index].cpu().tolist(),
            "intercept_time": float(ball.intercept_time[index]),
          }
      if len(starting) or len(returning):
        motion.update_relative_body_poses()
        uw.obs_buf = uw.observation_manager.compute(update_history=False)
      if viewer is not None:
        viewer.sync_env_to_viewer()
        time.sleep(max(0.0, dt - (time.monotonic() - frame_started)))
        if not viewer.is_running():
          stop_reason = "viewer closed before completion"
          break
      if bool((phase >= 4).all()):
        break
    change(phase < 4, 5, now, stop_reason)

  np.savez_compressed(
    output / "trace.npz", **{k: np.asarray(v) for k, v in history.items()}
  )
  metric_values = np.asarray(history["metrics"])
  for i, row in enumerate(rows):
    row["passed"] = bool(phase[i] == 4)
    row["final_phase"] = names[int(phase[i])]
    end_step = min(int(round(row["events"][-1]["time"] / dt)), len(metric_values))
    row["max_tilt_degrees"] = float(metric_values[:end_step, i, 2].max())
    row["final_metrics"] = metric_values[end_step - 1, i].tolist()
  report = {
    "task": task_id,
    "checkpoint": cfg.checkpoint_file,
    "checkpoint_sha256": hashlib.sha256(
      Path(cfg.checkpoint_file).read_bytes()
    ).hexdigest(),
    "tracker": uw.cfg.actions["joint_pos"].tracker_file,
    "tracker_sha256": hashlib.sha256(
      Path(uw.cfg.actions["joint_pos"].tracker_file).read_bytes()
    ).hexdigest(),
    "training_iteration": uw.current_learning_iteration,
    "ball_difficulty_scale": ball._difficulty_scale(),
    "seed": cfg.cycle_seed,
    "home": cfg.cycle_home,
    "physics": "ManagerBasedRlEnv train/play; no robot resets after initialization",
    "schedule": "tracker home >=3s and stable >=1s; Stage2 full clip; tracker home >=3s and stable >=1s; stable hold 5s",
    "metric_names": [
      "max_joint_velocity",
      "max_gyro",
      "tilt_degrees",
      "anchor_speed",
      "anchor_height_error",
    ],
    "stability_limits": limits.cpu().tolist(),
    "passed": sum(row["passed"] for row in rows),
    "total": count,
    "trials": rows,
  }
  (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
  print(f"[CYCLE] {report['passed']}/{count} passed; results: {output}", flush=True)
  motion._motion_warmup_mask = original_warmup_mask
  return report
