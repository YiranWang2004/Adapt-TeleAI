"""Command-conditioned incoming ball, then unrestricted physical response.

Uses the same portable dynamics/contact kernels as the previous MuJoCo ball.
No motion, selector, actor, outgoing target or success-rate inputs are accepted.
"""

import numpy as np
import torch

from deploy.g1_pingpong.ball_physics import (
  ballistic_step,
  swept_net_collision,
  swept_racket_collision,
  swept_table_collision,
)
from deploy.hope_g1.a3_parameters import ball_parameters

DEFAULTS = dict(
  **ball_parameters(),
  floor_height=-0.76,
  floor_restitution=0.4,
  incoming_speed_range=[2.0, 3.5],
  incoming_downward_speed_range=[1.0, 4.5],
  attempts=64,
)


def tensor(value):
  return torch.as_tensor(np.asarray(value), dtype=torch.float32).reshape(1, 3)


class PhysicalBall:
  def __init__(self, position, velocity, config=None):
    self.cfg = {**DEFAULTS, **(config or {})}
    self.position = np.asarray(position, dtype=float).copy()
    self.velocity = np.asarray(velocity, dtype=float).copy()
    self.elapsed = 0.0
    self.previous_racket = None
    self.contact = None
    self.net_crossing = None
    self.first_landing = None
    self.net_hit = False
    self.floor_hit = False
    self.bounces = 0
    self.trajectory = []
    self.feed = {}
    self._next_record = 0.0
    self.kernel = None
    if self.cfg.get("fast_physics", False):
      from deploy.benchmarks.racket_command.fast_ball import compiled_step, parameters

      self.kernel = compiled_step(parameters(self.cfg))

  def step(self, dt, racket_position=None, racket_normal=None):
    c = self.cfg
    shift = np.array([0.0, c["table_center_y"], 0.0])
    p0, v0 = tensor(self.position - shift), tensor(self.velocity)
    fast = None
    if self.kernel:
      center = (
        np.zeros(3) if racket_position is None else np.asarray(racket_position) - shift
      )
      previous = (
        center if self.previous_racket is None else self.previous_racket - shift
      )
      fast = self.kernel(
        p0,
        v0,
        tensor(previous),
        tensor(center),
        tensor([1.0, 0.0, 0.0] if racket_normal is None else racket_normal),
        torch.tensor(dt, dtype=torch.float32),
        torch.tensor([racket_position is not None and self.contact is None]),
      )
      p1, v1 = fast[:2]
    else:
      p1, v1 = ballistic_step(p0, v0, dt, c["gravity"], c["drag"])
    if racket_position is not None:
      previous = (
        racket_position if self.previous_racket is None else self.previous_racket
      )
      if self.contact is None:
        p1, v1, hit = (
          fast[:3]
          if fast is not None
          else swept_racket_collision(
            p0,
            p1,
            v1,
            tensor(previous - shift),
            tensor(racket_position - shift),
            tensor(racket_normal),
            dt,
            ball_radius=c["ball_radius"],
            racket_radius=c["racket_radius"],
            restitution=c["racket_restitution"],
          )
        )
        if bool(hit.item()):
          self.contact = dict(
            time=self.elapsed + dt,
            racket_position=np.asarray(racket_position).tolist(),
            racket_ball_distance_m=float(
              np.linalg.norm(racket_position - (p1.numpy()[0] + shift))
            ),
            position=(p1.numpy()[0] + shift).tolist(),
            incoming_velocity=v0.numpy()[0].tolist(),
            outgoing_velocity=v1.numpy()[0].tolist(),
            racket_velocity=((racket_position - previous) / dt).tolist(),
            racket_normal=np.asarray(racket_normal).tolist(),
          )
      self.previous_racket = np.asarray(racket_position).copy()
    net = (c["table_near_x"] + c["table_far_x"]) / 2
    p1, v1, net_hit, net_clear = (
      fast[3:7]
      if fast is not None
      else swept_net_collision(
        p0,
        p1,
        v1,
        dt,
        net_x=net,
        width=c["table_width"] + 2 * c.get("net_overhang", 0),
        table_height=c["table_height"],
        net_height=c["net_height"],
        ball_radius=c["ball_radius"],
      )
    )
    self.net_hit |= bool(net_hit.item())
    if self.contact and bool(net_clear.item()) and self.net_crossing is None:
      fraction = (net - float(p0[0, 0])) / max(float(p1[0, 0] - p0[0, 0]), 1e-12)
      point = (p0 + fraction * (p1 - p0)).numpy()[0] + shift
      self.net_crossing = dict(
        time=self.elapsed + dt * fraction, position=point.tolist()
      )
    plane = c["table_height"] + c["ball_radius"]
    if (
      self.contact
      and self.first_landing is None
      and float(p0[0, 2]) >= plane >= float(p1[0, 2])
      and float(v1[0, 2]) < 0
    ):
      fraction = (float(p0[0, 2]) - plane) / max(float(p0[0, 2] - p1[0, 2]), 1e-12)
      point = (p0 + fraction * (p1 - p0)).numpy()[0] + shift
      legal = (
        net < point[0] <= c["table_far_x"]
        and abs(point[1] - c["table_center_y"]) <= c["table_width"] / 2
      )
      self.first_landing = dict(
        time=self.elapsed + dt * fraction,
        position=point.tolist(),
        opponent_table=bool(legal),
      )
    if fast is not None:
      p1, v1, table_hit, floor_hit = fast[7:]
    else:
      p1, v1, table_hit, _ = swept_table_collision(
        p0,
        p1,
        v1,
        dt,
        near_x=c["table_near_x"],
        far_x=c["table_far_x"],
        width=c["table_width"],
        height=c["table_height"],
        ball_radius=c["ball_radius"],
        restitution=c["table_restitution"],
        tangential_damping=c["table_tangential_damping"],
      )
      p1, v1, floor_hit, _ = swept_table_collision(
        p0,
        p1,
        v1,
        dt,
        near_x=-1e6,
        far_x=1e6,
        width=2e6,
        height=c["floor_height"],
        ball_radius=c["ball_radius"],
        restitution=c["floor_restitution"],
        tangential_damping=0.8,
      )
    self.bounces += int(table_hit.item())
    self.floor_hit |= bool(floor_hit.item()) and self.first_landing is None
    self.position = p1.numpy()[0] + shift
    self.velocity = v1.numpy()[0]
    self.elapsed += dt
    if self.elapsed >= self._next_record:
      self.trajectory.append(
        dict(
          time=self.elapsed,
          position=self.position.tolist(),
          velocity=self.velocity.tolist(),
        )
      )
      self._next_record = self.elapsed + 0.01

  def report(self):
    legal = bool(
      self.contact
      and self.net_crossing
      and self.first_landing
      and self.first_landing["opponent_table"]
      and not self.net_hit
      and not self.floor_hit
      and self.net_crossing["time"] < self.first_landing["time"]
    )
    return dict(
      available=True,
      world_frame="hope_table",
      feed=self.feed,
      hit=self.contact is not None,
      contact=self.contact,
      net_hit=self.net_hit,
      floor_before_return_landing=self.floor_hit,
      net_crossing=self.net_crossing,
      first_landing=self.first_landing,
      legal_return=legal,
      dynamics=self.cfg,
      trajectory=self.trajectory,
    )


def make_incoming(trial, seed, config=None):
  """Back-propagate from the independent command, never a motion reference.

  Rejection concerns incoming legality only. Targets are never replaced, and
  failed construction remains visible in results. Time zero can be mid-flight.
  """
  c = {**DEFAULTS, **(config or {})}
  entropy = (
    [seed, trial.point_id, trial.direction_id, 991]
    if trial.point_id is not None and trial.direction_id is not None
    else [seed, trial.trial_id, 991]
  )
  rng = np.random.default_rng(np.random.SeedSequence(entropy))
  net = (c["table_near_x"] + c["table_far_x"]) / 2
  normal = np.asarray(trial.normal)
  if normal[0] < 0:
    normal = -normal
  target = np.asarray(trial.position) + c["ball_radius"] * normal
  plane = c["table_height"] + c["ball_radius"]

  def inside(p):
    return (
      c["table_near_x"] <= p[0] <= c["table_far_x"]
      and abs(p[1] - c["table_center_y"]) <= c["table_width"] / 2
    )

  for _ in range(c["attempts"]):
    speed = rng.uniform(*c["incoming_speed_range"])
    net_y = c["table_center_y"] + rng.uniform(-0.4, 0.4) * c["table_width"]
    vy = (target[1] - net_y) * speed / (net - target[0])
    p = target.copy()
    v = np.array([-speed, vy, -rng.uniform(*c["incoming_downward_speed_range"])])
    elapsed = 0.0
    bounces = 0
    initial = None
    invalid = False
    launch = None
    for _step in range(2000):
      # Reverse integration with drag can diverge before finding a legal launch.
      # Reject only this incoming construction, never alter the sampled command.
      if not np.isfinite(np.r_[p, v]).all() or np.linalg.norm(v) > 1000:
        invalid = True
        break
      dt = min(0.001, trial.tts - elapsed) if elapsed < trial.tts - 1e-10 else 0.001
      g = np.array([0.0, 0.0, -c["gravity"]])
      acceleration = g - c["drag"] * np.linalg.norm(v) * v
      q = p - v * dt + 0.5 * acceleration * dt * dt
      w = v - acceleration * dt
      if p[2] >= plane > q[2] and v[2] > 0:
        f = (p[2] - plane) / (p[2] - q[2])
        intersection = p + f * (q - p)
        if inside(intersection):
          if not c["table_near_x"] <= intersection[0] < net or bounces:
            invalid = True
            break
          post = v - acceleration * (dt * f)
          pre = post.copy()
          pre[:2] /= c["table_tangential_damping"]
          pre[2] /= -c["table_restitution"]
          remainder = dt * (1 - f)
          p = intersection - pre * remainder + 0.5 * g * remainder**2
          v = pre - g * remainder
          elapsed += dt
          bounces += 1
          if initial is None and elapsed >= trial.tts - 1e-10:
            initial = (p.copy(), v.copy())
          continue
      if inside(q) and q[2] < plane - 1e-6:
        invalid = True
        break
      if p[0] < net <= q[0]:
        point = p + (q - p) * (net - p[0]) / (q[0] - p[0])
        if (
          point[2] <= c["table_height"] + c["net_height"] + c["ball_radius"]
          or abs(point[1] - c["table_center_y"])
          > c["table_width"] / 2 - c["ball_radius"]
        ):
          invalid = True
          break
      p, v = q, w
      elapsed += dt
      if initial is None and elapsed >= trial.tts - 1e-10:
        initial = (p.copy(), v.copy())
      if (
        elapsed >= trial.tts
        and p[0] > net + 0.1
        and bounces == 1
        and p[2] > c["table_height"] + c["net_height"] + c["ball_radius"]
      ):
        launch = (p.copy(), v.copy(), elapsed)
        break
    if invalid or initial is None or launch is None:
      continue
    if initial[0][2] < c["floor_height"] + c["ball_radius"]:
      continue
    probe = PhysicalBall(*initial, c)
    for _ in range(int(np.floor(trial.tts / 0.001))):
      probe.step(0.001)
    remaining = trial.tts - probe.elapsed
    if remaining > 1e-10:
      probe.step(remaining)
    error = float(np.linalg.norm(probe.position - target))
    if error > 0.01 or probe.net_hit:
      continue
    ball = PhysicalBall(*initial, c)
    ball.feed = dict(
      source="independent_racket_command",
      intended_ball_center=target.tolist(),
      deadline=trial.tts,
      initial_position=initial[0].tolist(),
      initial_velocity=initial[1].tolist(),
      incoming_probe_error_m=error,
      prehistory_launch_position=launch[0].tolist(),
      prehistory_launch_velocity=launch[1].tolist(),
      prehistory_launch_time=trial.tts - launch[2],
      prehistory_table_bounces=1,
    )
    return ball
  return None
