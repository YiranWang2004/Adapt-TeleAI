"""Compare deployment observation/actor math on actual Isaac snapshots.

This isolates the software contract from plant dynamics and odometry: actual
Isaac instantaneous root velocity is supplied for this diagnostic only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from deploy.g1_pingpong.core import RobotState
from deploy.g1_pingpong.motions import MotionLibrary, RebasedMotion
from deploy.hope_g1.configuration import OBSERVATIONS, load_deployment
from deploy.hope_g1.controller import HopeActorController, RacketGoal
from deploy.hope_g1.deploy_agent import CONFIG_DIR


def compare(deployment, path):
  data = np.load(path)
  cfg = deployment.settings
  if data["joint_names"].tolist() != cfg["robot"]["joint_names"]:
    raise ValueError("Actual Isaac articulation order differs from saved joint order")
  library = MotionLibrary(deployment)
  actor = HopeActorController(cfg, deployment.actor)
  nominal_default = actor.default.copy()
  slices, offset = {}, 0
  for name, size in OBSERVATIONS.items():
    slices[name] = slice(offset, offset + size)
    offset += size
  term_errors = {name: 0.0 for name in slices}
  action_error = reference_error = 0.0
  nominal_observation_error = nominal_action_error = default_shift = 0.0
  worst = None
  errors = []
  for tick in range(len(data["obs"])):
    for env in range(data["obs"].shape[1]):
      frame, mid = int(data["frame"][tick, env]), int(data["motion_index"][tick, env])
      clip = library.motions[mid]
      _, position, quaternion = clip.sample(0)
      reference = RebasedMotion(clip, position + data["env_origins"][env], quaternion)
      state = RobotState(
        data["q"][tick, env],
        data["dq"][tick, env],
        data["root_pos"][tick, env].astype(float),
        data["root_quat"][tick, env],
        data["gyro"][tick, env],
        data["anchor_pos"][tick, env],
        data["anchor_quat"][tick, env],
        np.zeros(3),
        np.array([1.0, 0, 0]),
      )
      goal = RacketGoal(
        data["goal_position"][tick, env],
        data["goal_velocity"][tick, env],
        data["goal_normal"][tick, env],
        data["goal_base"][tick, env],
      )
      now = (tick + 1) * cfg["control_dt"]
      actor.previous_time = now - cfg["control_dt"]
      actor.previous_position = (
        state.root_pos - cfg["control_dt"] * data["root_velocity"][tick, env]
      )
      expected = data["obs"][tick, env]
      actor.previous_joint = expected[slices["actions"]].copy()
      # Startup randomization changes both joint_pos_rel's zero and the action
      # offset in Isaac. Compare arithmetic with that actual sampled zero;
      # retain nominal-zero differences separately instead of hiding them.
      if "default_q" in data:
        actor.default = data["default_q"][tick, env]
        actor.default_dq = data["default_dq"][tick, env]
      default_shift = max(
        default_shift, float(np.max(np.abs(actor.default - nominal_default)))
      )
      actual, _ = actor.observation(
        state, reference, frame, goal, float(data["tts"][tick, env]), now
      )
      delta = np.abs(actual - expected)
      maximum = float(delta.max())
      errors.append(maximum)
      if worst is None or maximum > worst["error"]:
        worst = dict(
          tick=tick,
          env=env,
          motion=mid,
          frame=frame,
          error=maximum,
          returning=bool(data["returning"][tick, env]),
          index=int(delta.argmax()),
        )
      for name, section in slices.items():
        term_errors[name] = max(term_errors[name], float(delta[section].max()))
      reference_error = max(
        reference_error,
        float(
          np.max(
            np.abs(
              np.r_[reference.q[frame], reference.dq[frame]]
              - data["reference_command"][tick, env]
            )
          )
        ),
      )
      action_error = max(
        action_error,
        float(np.max(np.abs(deployment.actor(actual) - data["action"][tick, env]))),
      )
      nominal_obs = actual.copy()
      nominal_obs[slices["joint_pos"]] += actor.default - nominal_default
      nominal_observation_error = max(
        nominal_observation_error, float(np.max(np.abs(nominal_obs - expected)))
      )
      nominal_action_error = max(
        nominal_action_error,
        float(
          np.max(np.abs(deployment.actor(nominal_obs) - data["action"][tick, env]))
        ),
      )
  return dict(
    conditions="actual Isaac state and sampled goals; same previous raw action; instantaneous root velocity (diagnostic only)",
    samples=len(errors),
    returning_samples=int(data["returning"].sum()),
    motions=np.unique(data["motion_index"]).tolist(),
    max_observation_error=max(errors),
    observation_error_by_term=term_errors,
    max_reference_command_error=reference_error,
    max_actor_action_error=action_error,
    max_sampled_default_joint_shift=default_shift,
    nominal_default_max_observation_error=nominal_observation_error,
    nominal_default_max_actor_action_error=nominal_action_error,
    worst=worst,
    observation_tolerance=1e-4,
    action_tolerance=1e-3,
    passed=max(errors) < 1e-4 and reference_error < 1e-5 and action_error < 1e-3,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--isaac-contract", type=Path, required=True)
  parser.add_argument("--config", type=Path, default=CONFIG_DIR / "task.yaml")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  config = yaml.safe_load(args.config.read_text())
  config.pop("task")
  result = compare(load_deployment(**config), args.isaac_contract)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
