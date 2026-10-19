"""HOPE single-actor receiving controller; no Stage 1/2 policy dependencies."""

from dataclasses import dataclass

import numpy as np

from deploy.g1_pingpong.core import quaternion_multiply, rotation
from deploy.g1_pingpong.lifecycle import Lifecycle
from deploy.g1_pingpong.motions import MotionLibrary, RebasedMotion
from deploy.g1_pingpong.perception import ReceivePlanner


@dataclass
class RacketGoal:
  position: np.ndarray
  velocity: np.ndarray
  normal: np.ndarray
  base_xy: np.ndarray


class HopeActorController:
  """Construct the saved 175-D actor observation in Isaac articulation order."""

  def __init__(self, cfg, policy):
    self.cfg, self.policy = cfg, policy
    self.default = np.asarray(cfg["robot"]["default_q"])
    self.default_dq = np.asarray(cfg["robot"]["default_dq"])
    self.scale = np.asarray(cfg["robot"]["action_scale"])
    self.reset()

  def reset(self):
    self.previous_joint = np.zeros(29, dtype=np.float32)
    self.previous_position = self.previous_time = None
    self.motion = None
    self.t = 0.0
    self.debug = {}

  def observation(self, state, reference, frame, goal, time_to_strike, now):
    frame = min(max(int(frame), 0), len(reference.q) - 1)
    q, pos, quat = reference.sample(frame / reference.fps)
    anchor_inv = rotation(state.anchor_quat).T
    root_inv = rotation(state.root_quat).T
    root_rot = rotation(state.root_quat)
    yaw = np.arctan2(root_rot[1, 0], root_rot[0, 0])
    heading_inv = rotation([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]).T
    # The actor was trained on world-frame targets in the NPZ/training world.
    # A live world rebase rotates the robot, reference and physical targets;
    # map these two absolute-vector terms back into that training world.
    policy_world_inv = reference.rotation.T
    # Use measured odometry, also in simulation; no plant-velocity oracle.
    velocity = np.zeros(3)
    if self.previous_time is not None:
      elapsed = now - self.previous_time
      if elapsed <= 0:
        raise ValueError("HOPE observation time must increase")
      velocity = (state.root_pos - self.previous_position) / elapsed
    terms = [
      q,
      reference.dq[frame],
      anchor_inv @ (pos - state.anchor_pos),
      (anchor_inv @ rotation(quat))[:, :2].reshape(-1),
      root_inv @ velocity,
      state.gyro,
      state.q - self.default,
      state.dq - self.default_dq,
      self.previous_joint,
      root_inv @ [0.0, 0.0, -1.0],
      (heading_inv @ np.r_[goal.base_xy - state.root_pos[:2], 0.0])[:2],
      heading_inv @ (goal.position - state.root_pos),
      policy_world_inv @ goal.velocity,
      policy_world_inv @ goal.normal,
      [time_to_strike],
    ]
    observation = np.concatenate(terms).astype(np.float32)
    if observation.shape != (175,) or not np.isfinite(observation).all():
      raise ValueError("Invalid HOPE observation")
    return observation, velocity

  def step(self, state, reference, frame, goal, time_to_strike, now):
    obs, velocity = self.observation(state, reference, frame, goal, time_to_strike, now)
    action = self.policy(obs)
    target = self.default + self.scale * action
    if target.shape != (29,) or not np.isfinite(target).all():
      raise ValueError("Invalid HOPE target")
    self.previous_joint = action.copy()
    self.previous_position, self.previous_time = state.root_pos.copy(), now
    self.motion, self.t = reference, frame / reference.fps
    self.debug = dict(
      actor_obs=obs,
      joint_action=action,
      target_q=target,
      reference_time=self.t,
      reference_frame=frame,
      time_to_strike=time_to_strike,
      root_velocity_world=velocity,
      goal_position=goal.position,
      goal_velocity=goal.velocity,
      goal_normal=goal.normal,
      goal_velocity_policy_frame=obs[168:171].copy(),
      goal_normal_policy_frame=obs[171:174].copy(),
      goal_base_xy=goal.base_xy,
    )
    return target


class HopeStanding:
  """Readiness diagnostics consumed by the shared AMP handoff."""

  def __init__(self, actor, runtime, home):
    self.controller, self.runtime, self.home = actor, runtime, home
    self.readiness_mode = runtime.get("readiness_mode", "timed")
    self.previous = None
    self.debug = {}

  def observe(self, state):
    limits = self.runtime["standing"]
    metrics = dict(
      pose_error=float(np.max(np.abs(state.q - self.home.q[0]))),
      joint_velocity=float(np.max(np.abs(state.dq))),
      gyro=float(np.max(np.abs(state.gyro))),
      tilt_degrees=float(
        np.degrees(np.arccos(np.clip(rotation(state.root_quat)[2, 2], -1, 1)))
      ),
    )
    settled = all(
      v <= limits[f"max_{k}"] for k, v in metrics.items() if f"max_{k}" in limits
    )
    self.debug = {**metrics, "stable": settled, "readiness_mode": self.readiness_mode}
    return settled, metrics


class HopeController:
  """Fixed-home receiving lifecycle around a single HOPE actor.

  Goals latch at selection. Idle/return use frame zero and the same actor;
  the motion clock advances only during EXECUTE. No tracker is substituted.
  """

  def __init__(
    self,
    deployment,
    runtime,
    options,
    *,
    home_index=0,
    allow=None,
    selection="fixed",
    receiving=True,
    auto_start=False,
    auto_arm=False,
  ):
    self.cfg, self.runtime, self.options = deployment.settings, runtime, options
    if self.cfg["task"] != "HOPEPingPongMultiMotionG1":
      raise ValueError("HopeController requires the independent HOPE task")
    if selection not in ("fixed", "auto"):
      raise ValueError("HOPE motion selection must be fixed or auto")
    if (
      not np.isfinite(options["max_intercept_distance"])
      or options["max_intercept_distance"] <= 0
      or not 0 <= options["max_start_frame"] <= self.cfg["hope"]["max_start_frame"]
    ):
      raise ValueError("Invalid HOPE interception distance/start-frame window")
    self.selection, self.receiving = selection, receiving
    self.auto_start, self.auto_arm = auto_start, auto_arm
    self.library = MotionLibrary(
      deployment, [home_index] if selection == "fixed" else allow
    )
    self.library.hit_types = self.cfg["hope"]["hit_types"]
    if not 0 <= home_index < len(self.library.motions):
      raise ValueError("Invalid HOPE home index")
    self.home_index = self.motion_index = home_index
    # The saved task derives strike frames from strike_phase, not NPZ annotations.
    for clip in self.library.motions:
      clip.annotated_strike = (
        round((len(clip.q) - 1) * self.cfg["hope"]["racket"]["strike_phase"]) / clip.fps
      )
    self.home = self.library.motions[home_index]
    self.lifecycle = Lifecycle(self.home.q[0], runtime["transition_seconds"])
    self.controller = HopeActorController(self.cfg, deployment.actor)
    self.standing = HopeStanding(self.controller, runtime, self.home)
    self.completed = self.recovered = 0
    self.events, self.debug = [], {}
    self.previous_target = self.planner = None
    self.physical_limit_observer = None
    self.home_frozen = False
    self.references = None
    self.frame = 0
    self.goal = None
    racket = self.cfg["hope"]["racket"]
    self.target_perturb_scale = options.get(
      "target_perturb_scale", racket["ref_perturb_curriculum_start"]
    )
    if (
      not np.isfinite(self.target_perturb_scale)
      or not 0 < self.target_perturb_scale <= 1
    ):
      raise ValueError("HOPE target perturbation scale must be within (0, 1]")
    self.target_offset_limit = (
      np.asarray(racket["ref_perturb_pos"]) * self.target_perturb_scale
    )
    self.recovery_stable_since = None
    self.command_debug = {}

  @property
  def candidate_indices(self):
    return [self.home_index] if self.selection == "fixed" else self.library.allow

  @property
  def phase(self):
    return {
      "ARMED": "IDLE",
      "EXECUTE": "SWING",
      "RECOVER": "RETURN",
      "READY": "IDLE",
    }.get(self.lifecycle.state, self.lifecycle.state)

  def describe(self):
    return (
      f"[hope_g1] task={self.cfg['task']}; home={self.home_index} "
      f"({self.library.paths[self.home_index].name}); selection={self.selection}; "
      f"candidates={self.candidate_indices}; actor=175->29"
    )

  def candidates(self, state):
    references = self.references or self.aligned_library(state)
    return [(i, references[i]) for i in self.candidate_indices]

  def aligned_library(self, state):
    """Keep all clips in one fixed training-world frame, as in play.

    Isaac adds the same environment origin to every clip. Recentring each
    clip on the moving robot at selection changes its anchor/target contract.
    """
    home = RebasedMotion(
      self.library.motions[self.home_index], state.anchor_pos, state.anchor_quat
    )
    references = []
    for clip in self.library.motions:
      _, p, q = clip.sample(0)
      references.append(
        RebasedMotion(
          clip,
          home.rotation @ p + home.translation,
          quaternion_multiply(home.quaternion, q),
        )
      )
    return references

  def reference_goal(self, reference, index, *, home=False):
    cfg = self.cfg["hope"]["racket"]
    f = round(reference.strike_time(self.cfg["ball"]) * reference.fps)
    source = reference.source
    body = source.body_names.index(cfg["wrist_body_name"])
    wrist_q = source.body_quat[f, body]
    mount_q = quaternion_multiply(wrist_q, cfg["mount_quat"])
    normal = (
      reference.rotation
      @ rotation(mount_q)[:, cfg["mount_normal_axis"]]
      * cfg["mount_normal_sign"]
    )
    normal /= np.linalg.norm(normal) + 1e-6
    sign = (
      -1 if self.library.hit_types[index] in cfg["ref_velocity_backhand_values"] else 1
    )
    position = reference.strike_position(self.cfg["ball"])
    base = reference.body_position("pelvis", 0 if home else f / reference.fps)[:2]
    return RacketGoal(position, normal * sign * cfg["ref_velocity_speed"], normal, base)

  def select(self, state, estimate, now):
    choices = []
    for index, reference in self.candidates(state):
      planner = ReceivePlanner(
        reference,
        self.cfg["ball"],
        spatial_tolerance=[np.inf] * 3,
        timing_tolerance=self.runtime["planner"]["timing_tolerance"],
        max_start_frame=self.options["max_start_frame"],
      )
      if not planner.try_start(estimate, now):
        self.debug["decision"] = {**planner.diagnostic, "motion_index": index}
        continue
      distance = float(
        np.linalg.norm(planner.plan[0] - reference.strike_position(self.cfg["ball"]))
      )
      offset = planner.plan[0] - reference.strike_position(self.cfg["ball"])
      if np.any(np.abs(offset) > self.target_offset_limit):
        self.debug["decision"] = dict(
          reason="goal_outside_play_perturbation",
          motion_index=index,
          offset=offset.tolist(),
          allowed_offset=self.target_offset_limit.tolist(),
        )
        continue
      if distance <= self.options["max_intercept_distance"]:
        choices.append((distance, index, reference, planner))
    if not choices:
      return None
    return min(choices, key=lambda choice: (choice[0], choice[1]))

  def start_policy(self, target, now, *, reset_history=False, blend_reference=True):
    if self.lifecycle.state not in ("STABLE", "POSE_HOLD"):
      raise RuntimeError("HOPE policy entry requires STABLE or POSE_HOLD")
    self.controller.reset()
    # Match the actual training reset: last_action starts at zero.
    self.lifecycle.state = "ARMED"
    self.previous_target = np.asarray(target).copy()
    self.home_frozen = False
    self.references = None
    self.frame = 0
    self.events.append(dict(time=now, event="hope_policy_enabled"))

  def reset_manual_policy(self, now):
    self.controller.reset()
    self.home = self.library.motions[self.home_index]
    self.home_frozen = False
    self.references = None
    self.goal = self.planner = None
    self.motion_index = self.home_index
    self.lifecycle.state = "WAIT"
    self.events.append(dict(time=now, event="returned_to_stable"))

  def step(self, state, estimate, now):
    if self.lifecycle.state == "HALT":
      return None
    if not self.home_frozen:
      self.references = self.aligned_library(state)
      self.home = self.references[self.home_index]
      self.standing.home = self.home
      self.home_goal = self.reference_goal(self.home, self.home_index, home=True)
      self.home_frozen = True
      self.events.append(
        dict(
          time=now,
          event="home_frozen",
          translation=self.home.translation.tolist(),
          quaternion_wxyz=self.home.quaternion.tolist(),
        )
      )
    self.standing.observe(state)
    tilt = self.standing.debug["tilt_degrees"]
    if tilt > self.runtime["max_tilt_degrees"]:
      error = ValueError("HOPE robot tilt exceeds configured limit")
      if self.physical_limit_observer is None:
        raise error
      self.physical_limit_observer("hope_tilt", error)
    self.debug = {"decision": {"reason": "waiting_for_ball"}}
    if self.lifecycle.state == "RECOVER":
      settled, _ = self.standing.observe(state)
      self.recovery_stable_since = (
        (now if self.recovery_stable_since is None else self.recovery_stable_since)
        if settled
        else None
      )
      elapsed = now - self.lifecycle.transition_start
      ready = elapsed >= self.cfg["hope"]["return_seconds"]
      if self.standing.readiness_mode == "measured":
        ready &= (
          self.recovery_stable_since is not None
          and now - self.recovery_stable_since
          >= self.runtime["standing"]["dwell_seconds"]
        )
      if ready:
        self.recovered += 1
        self.motion_index = self.home_index
        self.planner = None
        self.lifecycle.ready()
        self.lifecycle.arm(now, True)
        self.events.append(dict(time=now, event="recovered", home=self.home_index))
      elif elapsed > self.runtime["standing"]["timeout_seconds"]:
        raise TimeoutError("HOPE recovery did not stabilize")
    if self.lifecycle.state == "ARMED" and self.receiving and estimate is not None:
      choice = self.select(state, estimate, now)
      if choice is not None:
        distance, self.motion_index, self.reference, self.planner = choice
        self.frame = int(self.planner.diagnostic["start_frame"])
        self.goal = self.reference_goal(self.reference, self.motion_index)
        # The planner supplies desired contact position; velocity/face normal
        # retain the trained reference goal convention and stroke sign.
        displacement = self.planner.plan[0] - self.goal.position
        # RacketTargetCommand couples XY base placement to contact position:
        # target_base_xy = target_racket_xy - reference_reach_offset_xy.
        self.goal.base_xy = self.goal.base_xy + displacement[:2]
        self.goal.position = self.planner.plan[0].copy()
        self.lifecycle.execute(now)
        self.events.append(
          dict(
            time=now,
            event="motion_selected",
            motion_index=self.motion_index,
            motion=self.library.paths[self.motion_index].name,
            distance=distance,
            start_frame=self.frame,
            deadline=self.planner.plan[1],
          )
        )
    executing = self.lifecycle.state == "EXECUTE"
    reference = self.reference if executing else self.home
    frame = self.frame if executing else 0
    # Deployment uses a fixed home reference/goal in return and idle instead
    # of training's random resampling of the next clip at wrap.
    goal = self.goal if executing else self.home_goal
    strike_frame = round(reference.strike_time(self.cfg["ball"]) * reference.fps)
    target = self.controller.step(
      state,
      reference,
      frame,
      goal,
      (strike_frame - frame) * self.cfg["control_dt"],
      now,
    )
    self.command_policy = "hope_actor"
    self.command_debug = self.controller.debug
    self.debug.update(self.command_debug)
    self.previous_target = target.copy()
    if executing:
      self.frame += 1
      if self.frame >= len(reference.q):
        self.completed += 1
        self.lifecycle.recover(state.q, now)
        self.recovery_stable_since = None
        self.events.append(
          dict(time=now, event="swing_finished", motion_index=self.motion_index)
        )
    return target
