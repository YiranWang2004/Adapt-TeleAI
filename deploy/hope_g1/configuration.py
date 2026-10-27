"""Read HOPE's saved training contract without importing Isaac Sim."""

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from deploy.g1_pingpong.configuration import Deployment
from deploy.g1_pingpong.core import Motion
from deploy.g1_pingpong.pd_gains import deployment_gains
from deploy.g1_pingpong.policy import training_config
from deploy.hope_g1.policy import HopePolicy

TASK = "HOPEPingPongMultiMotionG1"
OBSERVATIONS = {
  "command": 58,
  "motion_anchor_pos_b": 3,
  "motion_anchor_ori_b": 6,
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "joint_pos": 29,
  "joint_vel": 29,
  "actions": 29,
  "projected_gravity": 3,
  "base_target_pos_b": 2,
  "racket_target_pos_b": 3,
  "racket_target_vel_w": 3,
  "racket_target_normal_w": 3,
  "time_to_strike": 1,
}

_OBS_MODULE = "isaaclab.envs.mdp.observations:"
_MOTION_MODULE = "whole_body_tracking.tasks.tracking.mdp.observations:"
_HOPE_MODULE = "whole_body_tracking.tasks.tracking.mdp.hope_observations:"
OBSERVATION_FUNCTIONS = {
  **{n: _OBS_MODULE + n for n in ("base_lin_vel", "base_ang_vel", "projected_gravity")},
  "command": _OBS_MODULE + "generated_commands",
  "joint_pos": _OBS_MODULE + "joint_pos_rel",
  "joint_vel": _OBS_MODULE + "joint_vel_rel",
  "actions": _OBS_MODULE + "last_action",
  **{n: _MOTION_MODULE + n for n in ("motion_anchor_pos_b", "motion_anchor_ori_b")},
  **{
    n: _HOPE_MODULE + n
    for n in (
      "base_target_pos_b",
      "racket_target_pos_b",
      "racket_target_vel_w",
      "racket_target_normal_w",
      "time_to_strike",
    )
  },
}


def resolve_patterns(value, names, *, default=None):
  """Resolve Isaac regex dictionaries by joint name, rejecting ambiguity."""
  if not isinstance(value, dict):
    result = np.full(len(names), value, dtype=float)
  else:
    result = []
    for name in names:
      matches = [v for expr, v in value.items() if re.fullmatch(expr, name)]
      if len(matches) > 1 or (not matches and default is None):
        raise ValueError(f"Ambiguous/missing joint setting: {name}")
      result.append(matches[0] if matches else default)
    result = np.asarray(result, dtype=float)
  if result.shape != (len(names),) or not np.isfinite(result).all():
    raise ValueError("Invalid per-joint values")
  return result


@dataclass
class HopeDeployment(Deployment):
  def report(self):
    return dict(
      task=TASK,
      controller="deploy.hope_g1.controller.HopeController",
      checkpoint=str(self.checkpoint),
      iteration=self.settings["iteration"],
      actor_input=175,
      actor_output=29,
      control_dt=self.settings["control_dt"],
      num_motions=len(self.motions),
      motions=[dict(index=i, file=p.name) for i, p in enumerate(self.motions)],
      normalization="(obs - mean) / (std + 0.01)",
      self_collisions=self.settings["hope"]["self_collisions"],
      collision_source=self.settings["hope"]["collision_source"],
      world_frame=self.settings["world_frame"],
      simulation_world_translation=self.settings["simulation_world_translation"],
    )


def load_deployment(
  checkpoint,
  *,
  motion_dir=None,
  env_config=None,
  agent_config=None,
  ball_config=None,
  world_config=None,
):
  checkpoint = Path(checkpoint).resolve()
  run = checkpoint.parent
  saved = training_config(env_config or run / "params/env.yaml")
  agent = training_config(agent_config or run / "params/agent.yaml")
  if agent.get("experiment_name") != "g1_hope_multi_motion":
    raise ValueError("Expected the g1_hope_multi_motion training task")
  group = saved["observations"]["policy"]
  terms = {k: v for k, v in group.items() if isinstance(v, dict) and "func" in v}
  if list(terms) != list(OBSERVATIONS):
    raise ValueError("HOPE actor observation order differs from the 175-D contract")
  if not group["concatenate_terms"] or group.get("history_length"):
    raise ValueError("Unsupported HOPE observation history/layout")
  for name, term in terms.items():
    if term["func"] != OBSERVATION_FUNCTIONS[name]:
      raise ValueError(f"HOPE observation function differs: {name}")
    if any(term.get(k) is not None for k in ("scale", "clip", "modifiers")) or term.get(
      "history_length", 0
    ):
      raise ValueError(f"Unsupported HOPE observation transformation: {name}")
  motion = saved["commands"]["motion"]
  racket = saved["commands"]["racket_target"]
  names = list(motion["joint_names"])
  if len(names) != 29 or len(set(names)) != 29:
    raise ValueError("Expected all 29 G1 joints in saved Isaac articulation order")
  action = saved["actions"]["joint_pos"]
  if (
    not action["use_default_offset"]
    or action.get("clip") is not None
    or action["joint_names"] != [".*"]
    or action.get("preserve_order", False)
    or any(action.get("delay_steps_range", [0, 0]))
    or any(action.get("actuator_lag_time_constant_range", [0, 0]))
  ):
    raise ValueError("Unsupported HOPE joint action contract")
  if (
    racket["base_target_pos_dim"] != 2
    or racket["ref_velocity_source"] != "normal_fixed"
  ):
    raise ValueError("Expected XY base target and normal_fixed racket velocity")
  if (
    motion["anchor_body_name"] != "torso_link"
    or motion.get("motion_manifest") is not None
    or not motion["wrap_to_start_without_reset"]
    or motion.get("early_stop_tracking_enabled", False)
  ):
    raise ValueError("Unsupported HOPE anchor, strike manifest or wrap contract")
  paths = [Path(p) for p in motion["motion_files"]]
  if motion_dir:
    paths = [Path(motion_dir) / p.name for p in paths]
  if not paths or len({p.name for p in paths}) != len(paths):
    raise ValueError("Empty/duplicate HOPE motion library")
  paths = [p.resolve() for p in paths]
  # Build this task's robot, floor and court without instantiating another task.
  from deploy.hope_g1.scene import table_cfg
  from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
    get_g1_pingpong_robot_cfg,
  )
  from mjlab.scene import Scene, SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  defaults = yaml.safe_load(
    (Path(__file__).parent / "configs/rally_mujoco.yaml").read_text()
  )
  ball_config = ball_config if ball_config is not None else defaults["ball"]
  world_config = world_config if world_config is not None else defaults["world"]
  from deploy.hope_g1.world import (
    mujoco_ball_config,
    simulation_translation,
    validate_table_world,
  )

  validate_table_world(ball_config)
  translation = simulation_translation(world_config)
  simulation_ball = mujoco_ball_config(ball_config, translation)

  robot = get_g1_pingpong_robot_cfg()
  # This task trained with its fast URDF and self-collisions enabled. Do not
  # silently inherit the other deploy task's foot spheres/disabled self contact.
  from functools import partial

  from deploy.hope_g1.assets import training_collision_spec

  urdf = run / "robot/g1_29dof_pingpong_fast.urdf"
  if not urdf.is_file():
    raise FileNotFoundError(
      f"Migrate the training collision URDF into the task assets: {urdf}"
    )
  self_collisions = bool(
    saved["scene"]["robot"]["spawn"]["articulation_props"]["enabled_self_collisions"]
  )
  robot.spec_fn = partial(training_collision_spec, robot.spec_fn, urdf, self_collisions)
  robot.collisions = tuple(
    replace(
      c,
      geom_names_expr=("hope_training_shape_.*",),
      contype=1 if self_collisions else 0,
      conaffinity=1,
    )
    for c in robot.collisions
  )
  init = saved["scene"]["robot"]["init_state"]
  robot.init_state.joint_pos = init["joint_pos"]
  robot.init_state.joint_vel = init["joint_vel"]
  robot.init_state.pos = tuple(init["pos"])
  model = Scene(
    SceneCfg(
      entities={"robot": robot, "table": table_cfg(simulation_ball)},
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    device="cpu",
  ).compile()
  collision_ids = [
    i
    for i in range(model.ngeom)
    if model.geom(i).name.startswith("robot/hope_training_shape_")
  ]
  if (
    not collision_ids
    or not np.all(model.geom_conaffinity[collision_ids] == 1)
    or not np.all(model.geom_contype[collision_ids] == (1 if self_collisions else 0))
  ):
    raise ValueError(
      "Training collision masks were not propagated into the compiled scene"
    )
  joints = [model.joint("robot/" + n).id for n in names]
  default = resolve_patterns(init["joint_pos"], names, default=0.0)
  scales = resolve_patterns(action["scale"], names)
  kp, kd, torque, armature = [np.zeros(29) for _ in range(4)]
  covered = np.zeros(29, dtype=int)
  for actuator in saved["scene"]["robot"]["actuators"].values():
    indices = [
      i
      for i, n in enumerate(names)
      if any(re.fullmatch(p, n) for p in actuator["joint_names_expr"])
    ]
    subset = [names[i] for i in indices]
    for dest, key in (
      (kp, "stiffness"),
      (kd, "damping"),
      (torque, "effort_limit_sim"),
      (armature, "armature"),
    ):
      dest[indices] = resolve_patterns(actuator[key], subset)
    covered[indices] += 1
  if not np.all(covered == 1) or np.any(scales <= 0):
    raise ValueError("Invalid actuator coverage or action scales")
  # The shared runtime uses literal execution gains; require agreement up front.
  execution = deployment_gains(names)
  if not np.allclose(kp, execution[0]) or not np.allclose(kd, execution[1]):
    raise ValueError("HOPE training PD differs from the shared execution PD table")
  model.dof_armature[model.jnt_dofadr[joints]] = armature
  for j, limit in zip(joints, torque, strict=True):
    a = np.flatnonzero(model.actuator_trnid[:, 0] == j)
    if len(a) != 1:
      raise ValueError("Expected one position servo per G1 joint")
    model.actuator_forcerange[a[0]] = [-limit, limit]
  dt = float(saved["sim"]["dt"])
  control_dt = dt * saved["decimation"]
  if not np.isclose(control_dt, 0.02):
    raise ValueError("HOPE task must run at 50Hz")
  hit_path = Path(racket["ref_velocity_hit_type_file"])
  if motion_dir:
    hit_path = Path(motion_dir) / hit_path.name
  hit_map = json.loads(hit_path.read_text())
  hit_types = [
    hit_map.get(p.stem, hit_map.get(p.name, racket["ref_velocity_default_hit_type"]))
    for p in paths
  ]
  for path in paths:
    clip = Motion(path, names, motion["anchor_body_name"])
    if not np.isclose(clip.fps, 1 / control_dt):
      raise ValueError(f"Motion fps differs from policy rate: {path}")
  policy = HopePolicy(checkpoint, agent)
  settings = dict(
    ball=dict(ball_config),
    world_frame="hope_table",
    simulation_world_translation=translation.tolist(),
    simulation_ball=simulation_ball,
    task=TASK,
    stage="hope_multi_motion",
    iteration=policy.iteration,
    control_dt=control_dt,
    physics_dt=dt,
    decimation=saved["decimation"],
    sim_options=dict(integrator="implicitfast", timestep=dt, iterations=20),
    anchor=motion["anchor_body_name"],
    motions=[p.name for p in paths],
    robot=dict(
      joint_names=names,
      default_q=default.tolist(),
      default_dq=resolve_patterns(init["joint_vel"], names).tolist(),
      kp=kp.tolist(),
      kd=kd.tolist(),
      torque_limits=torque.tolist(),
      joint_limits=model.jnt_range[joints].tolist(),
      action_scale=scales.tolist(),
    ),
    hope=dict(
      collision_source=str(urdf),
      self_collisions=self_collisions,
      motion=motion,
      racket=racket,
      hit_types=hit_types,
      return_seconds=motion["return_to_start_steps"] * control_dt,
      max_start_frame=motion["random_start_frame_max"],
    ),
    actor_terms=OBSERVATIONS,
  )
  return HopeDeployment(
    settings=settings,
    model=model,
    motions=paths,
    tracker=None,
    actor=policy,
    checkpoint=checkpoint,
    tracker_path=None,
  )
