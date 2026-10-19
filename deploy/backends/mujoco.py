"""Independent CPU MuJoCo backend and training-compatible analytical ball."""

import mujoco
import numpy as np
import torch

from deploy.g1_pingpong.ball_physics import (
  ballistic_step,
  swept_net_collision,
  swept_racket_collision,
  swept_table_collision,
)
from deploy.g1_pingpong.core import (
  BallState,
  RobotState,
  rotation,
)


def configure_model(model, options):
  enums = {
    "integrator": {
      "euler": mujoco.mjtIntegrator.mjINT_EULER,
      "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    },
    "cone": {
      "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
      "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    },
    "jacobian": {
      "auto": mujoco.mjtJacobian.mjJAC_AUTO,
      "dense": mujoco.mjtJacobian.mjJAC_DENSE,
      "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    },
    "solver": {
      "newton": mujoco.mjtSolver.mjSOL_NEWTON,
      "cg": mujoco.mjtSolver.mjSOL_CG,
      "pgs": mujoco.mjtSolver.mjSOL_PGS,
    },
  }
  for key, value in options.items():
    if key in enums:
      setattr(model.opt, key, enums[key][value])
    elif key in ("disableflags", "enableflags"):
      enum = mujoco.mjtDisableBit if key == "disableflags" else mujoco.mjtEnableBit
      prefix = "mjDSBL_" if key == "disableflags" else "mjENBL_"
      bits = 0
      for name in value:
        bits |= int(getattr(enum, prefix + name.upper()))
      setattr(model.opt, key, bits)
    else:
      setattr(model.opt, key, value)


class MujocoBackend:
  def __init__(self, scene_path, manifest):
    self.cfg = manifest
    self.model = (
      scene_path
      if isinstance(scene_path, mujoco.MjModel)
      else mujoco.MjModel.from_xml_path(str(scene_path))
    )
    configure_model(self.model, manifest.get("sim_options", {}))
    self.model.opt.timestep = manifest["physics_dt"]
    self.data = mujoco.MjData(self.model)
    self.joints = [
      self.model.joint("robot/" + n).id for n in manifest["robot"]["joint_names"]
    ]
    self.qpos = self.model.jnt_qposadr[self.joints]
    self.qvel = self.model.jnt_dofadr[self.joints]
    self.actuators = [
      int(np.flatnonzero(self.model.actuator_trnid[:, 0] == j)[0]) for j in self.joints
    ]
    if "kp" in manifest["robot"] and "kd" in manifest["robot"]:
      self.set_gains(manifest["robot"]["kp"], manifest["robot"]["kd"])
    self.root = self.model.body("robot/pelvis").id
    self.anchor = self.model.body("robot/" + manifest["anchor"]).id
    self.racket = self.model.site("robot/right_racket_surface").id
    self.feet = [
      self.model.body("robot/" + name).id
      for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    self.free_joint = self.model.body_jntadr[self.root]
    self.free_qpos = int(self.model.jnt_qposadr[self.free_joint])
    self.free_qvel = int(self.model.jnt_dofadr[self.free_joint])
    self.reset()

  def reset(self, motion=None, *, stationary=False):
    mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
    if motion is not None:
      self.data.qpos[self.qpos] = motion.q[0]
      self.data.qvel[self.qvel] = motion.dq[0]
      i = self.free_qpos
      self.data.qpos[i : i + 3] = motion.body_pos[0, motion.root]
      self.data.qpos[i + 3 : i + 7] = motion.body_quat[0, motion.root]
      v = self.free_qvel
      self.data.qvel[v : v + 3] = motion.body_lin_vel[0, motion.root]
      self.data.qvel[v + 3 : v + 6] = (
        rotation(motion.body_quat[0, motion.root]).T
        @ motion.body_ang_vel[0, motion.root]
      )
    if stationary:
      # Live waiting starts from a held pose, not an already moving training reset.
      self.data.qvel[:] = 0
    self.data.ctrl[self.actuators] = self.data.qpos[self.qpos]
    mujoco.mj_forward(self.model, self.data)

  def state(self):
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
      self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.root, velocity, 0
    )
    return RobotState(
      self.data.qpos[self.qpos].copy(),
      self.data.qvel[self.qvel].copy(),
      self.data.xpos[self.root].copy(),
      self.data.xquat[self.root].copy(),
      # mj_objectVelocity(local=1) uses the body's principal-inertia frame,
      # which differs markedly from the G1 pelvis link/IMU axes.
      rotation(self.data.xquat[self.root]).T @ velocity[:3],
      self.data.xpos[self.anchor].copy(),
      self.data.xquat[self.anchor].copy(),
      self.data.site_xpos[self.racket].copy(),
      self.data.site_xmat[self.racket].reshape(3, 3)[:, 1].copy(),
    )

  def forward_kinematics(self):
    """Update measured poses/velocities without solving fictitious contacts."""
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)
    mujoco.mj_comVel(self.model, self.data)

  def set_gains(self, kp, kd):
    self.model.actuator_gainprm[self.actuators, 0] = kp
    self.model.actuator_biasprm[self.actuators, 1] = -np.asarray(kp)
    self.model.actuator_biasprm[self.actuators, 2] = -np.asarray(kd)

  def set_target(self, target):
    self.data.ctrl[self.actuators] = target

  def substep(self):
    mujoco.mj_step(self.model, self.data)
    mujoco.mj_forward(self.model, self.data)

  def damp(self, kd=2.0):
    """Apply damping in simulation after a control halt, without resetting pose."""
    self.model.actuator_gainprm[self.actuators, 0] = 0
    self.model.actuator_biasprm[self.actuators, 1] = 0
    self.model.actuator_biasprm[self.actuators, 2] = -kd


class SimBall:
  def __init__(self, cfg, motion, rng, *, stop_on_finished=True):
    self.cfg = cfg
    self.stop_on_finished = stop_on_finished
    self.elapsed = 0.0
    self.intercept = (
      motion.strike_position(cfg) + rng.uniform(-1, 1, 3) * cfg["intercept_jitter"]
    )
    self.arrival = max(
      motion.strike_time(cfg)
      + rng.uniform(-cfg["arrival_time_jitter"], cfg["arrival_time_jitter"]),
      cfg["minimum_intercept_time"],
    )
    flight = min(rng.uniform(*cfg["flight_time_range"]), self.arrival)
    self.launch = self.arrival - flight
    self.velocity = np.array(
      [
        -rng.uniform(*cfg["incoming_speed_range"]),
        rng.uniform(*cfg["incoming_lateral_speed_range"]),
        cfg["incoming_vertical_speed"] + cfg["gravity"] * flight,
      ]
    )
    self.position = self.intercept - self.velocity * flight
    self.position[2] += 0.5 * cfg["gravity"] * flight**2
    if cfg["drag"]:
      # Match the training launcher's four shooting corrections and 24 steps.
      for _ in range(4):
        p = torch.tensor(self.position, dtype=torch.float32).reshape(1, 3)
        v = torch.tensor(self.velocity, dtype=torch.float32).reshape(1, 3)
        for _ in range(24):
          p, v = ballistic_step(p, v, flight / 24, cfg["gravity"], cfg["drag"])
        self.velocity += (self.intercept - p.numpy()[0]) / flight
    self.target = np.array(
      [*cfg["target_xy"], cfg["table_height"] + cfg["ball_radius"]]
    )
    self.target[:2] += rng.uniform(-1, 1, 2) * cfg["target_jitter"]
    self.hit = self.landed = self.missed = self.net_clear = self.net_hit = False
    self.landing_error = None
    self.previous_racket = None

  @classmethod
  def receiving_feed(cls, cfg, motion, rng):
    """Observable incoming rally: approach, table bounce, then strike plane.

    Training's delayed launcher assumes the policy is already running before
    release. A position-triggered deployment instead needs the pre-bounce
    approach, as in HOPE's flight/bounce planner. This feed uses the same table
    collision coefficients and incoming strike velocity, without policy truth.
    """
    if cfg["drag"] != 0:
      raise ValueError("Full rally feed currently requires drag=0")
    height = cfg["table_height"] + cfg["ball_radius"]
    gravity = cfg["gravity"]
    vz = cfg["incoming_vertical_speed"]
    net_x = (cfg["table_near_x"] + cfg["table_far_x"]) / 2
    for _ in range(64):
      ball = cls(cfg, motion, rng, stop_on_finished=False)
      discriminant = vz**2 + 2 * gravity * (ball.intercept[2] - height)
      if discriminant <= 0:
        continue
      # The longer backwards root is the ascending state just after the bounce.
      post_flight = (-vz + np.sqrt(discriminant)) / gravity
      post_velocity = np.array(
        [ball.velocity[0], ball.velocity[1], vz + gravity * post_flight]
      )
      bounce = ball.intercept.copy()
      bounce[:2] -= post_velocity[:2] * post_flight
      bounce[2] = height
      if not cfg["table_near_x"] <= bounce[0] < net_x:
        # A fixed downward strike velocity can place the bounce beyond the
        # net for high G1 strokes. Sample a receiving-side bounce and solve
        # its vertical flight instead of declaring these motions unfeedable.
        bounce_x = rng.uniform(
          cfg["table_near_x"] + cfg["ball_radius"], net_x - cfg["ball_radius"]
        )
        post_flight = (ball.intercept[0] - bounce_x) / post_velocity[0]
        if post_flight <= 0:
          continue
        post_velocity[2] = (
          ball.intercept[2] - height
        ) / post_flight + 0.5 * gravity * post_flight
        if post_velocity[2] <= 0:
          continue
        bounce[:2] = ball.intercept[:2] - post_velocity[:2] * post_flight
      if (
        cfg["table_near_x"] <= bounce[0] < net_x
        and abs(bounce[1]) <= cfg["table_width"] / 2
      ):
        break
    else:
      raise ValueError("No on-table rally feed in the configured incoming range")
    pre_velocity = post_velocity.copy()
    pre_velocity[:2] /= 0.96  # swept_table_collision tangential damping.
    pre_velocity[2] /= -cfg["table_restitution"]
    pre_flight = max(0.2, motion.strike_time(cfg) + 0.16 - post_flight)
    ball.position = bounce - pre_velocity * pre_flight
    ball.position[2] -= 0.5 * gravity * pre_flight**2
    ball.velocity = pre_velocity + np.array([0, 0, gravity * pre_flight])
    ball.launch = 0.0
    ball.arrival = pre_flight + post_flight
    return ball

  def observation(self):
    return BallState(
      self.position.copy(),
      self.velocity.copy(),
      self.intercept,
      self.arrival - self.elapsed,
      self.target,
      self.hit,
    )

  def substep(self, state, dt):
    previous_elapsed = self.elapsed
    self.elapsed += dt
    if self.previous_racket is None:
      self.previous_racket = state.racket_pos.copy()
    if self.elapsed < self.launch or (
      self.stop_on_finished and (self.landed or self.missed)
    ):
      self.previous_racket = state.racket_pos.copy()
      return

    def tensor(a):
      return torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(1, 3)

    p0, v0 = tensor(self.position), tensor(self.velocity)
    live_dt = min(self.elapsed - max(previous_elapsed, self.launch), dt)
    p1, v1 = ballistic_step(p0, v0, live_dt, self.cfg["gravity"], self.cfg["drag"])
    if not self.hit:
      p1, v1, contact = swept_racket_collision(
        p0,
        p1,
        v1,
        tensor(self.previous_racket),
        tensor(state.racket_pos),
        tensor(state.racket_normal),
        dt,
        ball_radius=self.cfg["ball_radius"],
        racket_radius=self.cfg["racket_radius"],
        restitution=self.cfg["racket_restitution"],
      )
      self.hit |= bool(contact.item())
    net_x = (self.cfg["table_near_x"] + self.cfg["table_far_x"]) / 2
    p1, v1, net_hit, net_clear = swept_net_collision(
      p0,
      p1,
      v1,
      dt,
      net_x=net_x,
      width=self.cfg["table_width"],
      table_height=self.cfg["table_height"],
      net_height=self.cfg["net_height"],
      ball_radius=self.cfg["ball_radius"],
    )
    self.net_hit |= bool(net_hit.item())
    self.net_clear |= bool(net_clear.item()) and self.hit
    p1, v1, table_hit, landing = swept_table_collision(
      p0,
      p1,
      v1,
      dt,
      near_x=self.cfg["table_near_x"],
      far_x=self.cfg["table_far_x"],
      width=self.cfg["table_width"],
      height=self.cfg["table_height"],
      ball_radius=self.cfg["ball_radius"],
      restitution=self.cfg["table_restitution"],
    )
    if bool(table_hit.item()) and self.hit and not self.landed:
      self.landed = True
      self.success = (
        self.net_clear and not self.net_hit and float(landing[0, 0]) > net_x
      )
      self.landing_error = float(
        np.linalg.norm(landing.numpy()[0, :2] - self.target[:2])
      )
    self.position, self.velocity = p1.numpy()[0], v1.numpy()[0]
    self.missed |= (
      self.net_hit
      or self.position[2] < self.cfg["ball_radius"]
      or self.position[0] < -1
      or self.position[0] > self.cfg["table_far_x"] + 2
      or abs(self.position[1]) > 2.5
      or self.elapsed > self.cfg["max_ball_time"]
    )
    self.previous_racket = state.racket_pos.copy()
