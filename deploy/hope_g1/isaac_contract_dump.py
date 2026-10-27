"""Run inside the training Isaac environment to capture actual G1 observation inputs.

Writes only to dump_dir (default /tmp/hope_g1_contract); no ONNX export or motors.
Uses play.py's task/algo composition and actor-only checkpoint loader. Noise and
pushes are disabled so raw observation arithmetic can be compared independently.
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

TRAINING_ROOT = Path(
  os.environ.get(
    "HOPE_TRAINING_ROOT", "/home/lxz/HOPE/hope_training/whole_body_tracking"
  )
)
sys.path.insert(0, str(TRAINING_ROOT / "scripts"))


def run(cfg):
  import gymnasium as gym
  import numpy as np
  import torch
  import whole_body_tracking.tasks  # noqa: F401
  import yaml
  from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
  from isaaclab_tasks.utils import parse_env_cfg
  from rsl_rl.runners import OnPolicyRunner
  from train import _apply_task_overrides, _resolve_motion_paths, _set_motion_paths
  from whole_body_tracking.utils.onnx_checkpoint import (
    load_actor_and_normalizer_for_onnx_export,
  )
  from whole_body_tracking.utils.ppo_cfg import runner_kwargs

  env_cfg = parse_env_cfg(
    str(cfg.task.gym_task), device=str(cfg.device), num_envs=int(cfg.num_envs)
  )
  _apply_task_overrides(env_cfg, cfg.task)
  env_cfg.sim.device = str(cfg.device)
  env_cfg.viewer.origin_type = "world"
  env_cfg.observations.policy.enable_corruption = False
  for name in ("push_robot", "push_by_setting_velocity"):
    if hasattr(env_cfg.events, name):
      setattr(env_cfg.events, name, None)
  motions = (
    cfg.get("motion_files") or cfg.task.get("motion_files") or cfg.get("motion_file")
  )
  _set_motion_paths(
    env_cfg,
    _resolve_motion_paths(motions),
    source="contract audit",
    log_prefix="[audit]",
  )
  algo = cfg.algo
  if cfg.task.get("algo_overrides") is not None:
    algo = OmegaConf.merge(algo, cfg.task.algo_overrides)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    **runner_kwargs(
      OmegaConf.to_container(algo, resolve=True), str(cfg.task.experiment_name)
    )
  )
  agent_cfg.device = str(cfg.device)
  output = Path(str(cfg.get("dump_dir", "/tmp/hope_g1_contract")))
  output.mkdir(parents=True, exist_ok=True)
  (output / "effective_env.yaml").write_text(yaml.dump(env_cfg.to_dict()))
  (output / "effective_agent.yaml").write_text(yaml.dump(agent_cfg.to_dict()))
  env = RslRlVecEnvWrapper(
    gym.make(str(cfg.task.gym_task), cfg=env_cfg), clip_actions=agent_cfg.clip_actions
  )
  runner = OnPolicyRunner(
    env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
  )
  load_actor_and_normalizer_for_onnx_export(runner, str(cfg.checkpoint))
  policy = runner.get_inference_policy(device=env.unwrapped.device)
  robot = env.unwrapped.scene["robot"]
  motion = env.unwrapped.command_manager.get_term("motion")
  racket = env.unwrapped.command_manager.get_term("racket_target")
  anchor = robot.body_names.index(env_cfg.commands.motion.anchor_body_name)

  def policy_obs(value):
    if isinstance(value, tuple):
      value = value[0]
    return value["policy"] if isinstance(value, dict) else value

  records = {}

  def save(name, value):
    records.setdefault(name, []).append(value.detach().cpu().numpy().copy())

  obs = policy_obs(env.get_observations())
  with torch.inference_mode():
    for step in range(int(cfg.get("dump_steps", 220))):
      action = policy(obs)
      for name, value in dict(
        obs=obs,
        action=action,
        q=robot.data.joint_pos,
        dq=robot.data.joint_vel,
        default_q=robot.data.default_joint_pos,
        default_dq=robot.data.default_joint_vel,
        root_pos=robot.data.root_pos_w,
        root_quat=robot.data.root_quat_w,
        gyro=robot.data.root_ang_vel_b,
        root_velocity=robot.data.root_lin_vel_w,
        anchor_pos=robot.data.body_pos_w[:, anchor],
        anchor_quat=robot.data.body_quat_w[:, anchor],
        goal_position=racket.racket_target_pos_w,
        goal_velocity=racket.racket_target_vel_w,
        goal_normal=racket.racket_target_normal_w,
        goal_base=racket.base_target_pos_w,
        tts=racket.time_to_strike,
        frame=motion.reference_time_steps,
        motion_index=motion.reference_motion_idx,
        returning=motion.return_to_start_active,
        reference_command=motion.command,
      ).items():
        save(name, value)
      if step % 50 == 0:
        print(
          f"[audit] {step}: obs={tuple(obs.shape)} frame={motion.reference_time_steps.tolist()}",
          flush=True,
        )
      obs = policy_obs(env.step(action))
  np.savez_compressed(
    output / "contract.npz",
    **{k: np.stack(v) for k, v in records.items()},
    env_origins=env.unwrapped.scene.env_origins.cpu().numpy(),
    joint_names=np.asarray(robot.joint_names),
  )
  print(f"[audit] saved {output}", flush=True)
  env.close()


@hydra.main(
  config_path=str(TRAINING_ROOT / "cfg"), config_name="play.yaml", version_base=None
)
def main(cfg):
  from isaaclab.app import AppLauncher

  app = AppLauncher(headless=True, device=str(cfg.device)).app
  try:
    run(cfg)
  finally:
    app.close()


if __name__ == "__main__":
  main()
