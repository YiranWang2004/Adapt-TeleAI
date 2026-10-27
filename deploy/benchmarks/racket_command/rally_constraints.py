"""NumPy-only command validity checks, independent of planner and policy.

These checks define the sampled task distribution. They never inspect robot
execution, select a motion, or correct an executed ball trajectory.
"""

import numpy as np


def acceleration(v, physics):
  a = -physics["drag"] * np.linalg.norm(v, axis=-1, keepdims=True) * v
  a[..., 2] -= physics["gravity"]
  return a


def contact_center(position, normal, physics):
  n = np.asarray(normal, dtype=float)
  if n[0] < 0:
    n = -n
  return np.asarray(position) + physics["ball_radius"] * n


def incoming_history(position, velocity, physics, dt=0.002):
  """Check a receiving-side bounce and a preceding inbound net crossing."""
  p, v = np.array(position, dtype=float), np.array(velocity, dtype=float)
  plane = physics["table_height"] + physics["ball_radius"]
  near, far = physics["table_near_x"], physics["table_far_x"]
  net, cy, width = (near + far) / 2, physics["table_center_y"], physics["table_width"]
  bounces = 0
  bounce = None
  for step in range(1200):
    if not np.isfinite(v).all() or np.linalg.norm(v) > 50:
      return None
    a = acceleration(v, physics)
    q, w = p - v * dt + 0.5 * a * dt**2, v - a * dt
    if p[2] >= plane > q[2] and v[2] > 0:
      f = (p[2] - plane) / (p[2] - q[2])
      b = p + f * (q - p)
      if near <= b[0] <= far and abs(b[1] - cy) <= width / 2:
        if bounces or not near + 0.04 <= b[0] < net - 0.04:
          return None
        pre = v - a * (dt * f)
        pre[:2] /= physics["table_tangential_damping"]
        pre[2] /= -physics["table_restitution"]
        rem = dt * (1 - f)
        q = b - pre * rem + 0.5 * acceleration(pre, physics) * rem**2
        w = pre - acceleration(pre, physics) * rem
        bounces += 1
        bounce = b.tolist()
    if near <= q[0] <= far and abs(q[1] - cy) <= width / 2 and q[2] < plane - 0.005:
      return None
    if p[0] < net <= q[0]:
      point = p + (q - p) * (net - p[0]) / (q[0] - p[0])
      if (
        bounces != 1
        or point[2]
        < physics["table_height"]
        + physics["net_height"]
        + physics["ball_radius"]
        + 0.02
        or abs(point[1] - cy) > width / 2 - 0.03
      ):
        return None
      return dict(
        velocity=list(velocity),
        bounce_position=bounce,
        net_position=point.tolist(),
        history_seconds=(step + 1) * dt,
      )
    p, v = q, w
  return None


def incoming_bank(position, physics, rng, count=8):
  bank = []
  net = (physics["table_near_x"] + physics["table_far_x"]) / 2
  for _ in range(256):
    speed = rng.uniform(*physics["incoming_speed_range"])
    net_y = physics["table_center_y"] + rng.uniform(-0.3, 0.3) * physics["table_width"]
    vy = (position[1] - net_y) * speed / (net - position[0])
    v = np.array([-speed, vy, -rng.uniform(*physics["incoming_downward_speed_range"])])
    if incoming_history(position, v, physics) is not None:
      bank.append(v)
      if len(bank) == count:
        break
  return np.asarray(bank)


def return_outcomes(position, normal, velocities, incoming, physics, options, dt=0.005):
  """Predict unassisted first landings for a batch of desired racket speeds."""
  normal = np.asarray(normal)
  vr = np.asarray(velocities)
  vi = np.broadcast_to(incoming, vr.shape)
  rel = np.sum((vi - vr) * normal, axis=-1)
  v = vi - (1 + physics["racket_restitution"]) * rel[:, None] * normal
  p = np.broadcast_to(contact_center(position, normal, physics), v.shape).copy()
  speed = np.linalg.norm(v, axis=-1)
  live = (
    (v[:, 0] > 0)
    & (speed >= options["outgoing_speed_range"][0])
    & (speed <= options["outgoing_speed_range"][1])
  )
  passed = np.zeros(len(v), dtype=bool)
  landed = np.zeros(len(v), dtype=bool)
  landings = np.full_like(v, np.nan)
  net_heights = np.full(len(v), np.nan)
  near, far = physics["table_near_x"], physics["table_far_x"]
  net, cy = (near + far) / 2, physics["table_center_y"]
  plane = physics["table_height"] + physics["ball_radius"]
  margin = options["landing_margin"]
  for _ in range(round(2 / dt)):
    if not live.any():
      break
    a = acceleration(v, physics)
    q, w = p + v * dt + 0.5 * a * dt**2, v + a * dt
    enters = live & (p[:, 0] < near) & (q[:, 0] >= near)
    f = np.clip((near - p[:, 0]) / np.maximum(q[:, 0] - p[:, 0], 1e-12), 0, 1)
    edge_z = p[:, 2] + f * (q[:, 2] - p[:, 2])
    live &= ~(enters & (edge_z < plane))
    crosses = live & (p[:, 0] < net) & (q[:, 0] >= net)
    f = np.clip((net - p[:, 0]) / np.maximum(q[:, 0] - p[:, 0], 1e-12), 0, 1)
    at_net = p + f[:, None] * (q - p)
    good = (
      at_net[:, 2]
      >= physics["table_height"]
      + physics["net_height"]
      + physics["ball_radius"]
      + options["net_clearance"]
    )
    good &= abs(at_net[:, 1] - cy) <= physics["table_width"] / 2 - margin
    passed |= crosses & good
    net_heights[crosses] = at_net[crosses, 2]
    live &= ~(crosses & ~good)
    lands = live & (p[:, 2] >= plane) & (q[:, 2] < plane) & (w[:, 2] < 0)
    f = np.clip((p[:, 2] - plane) / np.maximum(p[:, 2] - q[:, 2], 1e-12), 0, 1)
    at_land = p + f[:, None] * (q - p)
    inside = (at_land[:, 0] >= net + margin) & (at_land[:, 0] <= far - margin)
    inside &= abs(at_land[:, 1] - cy) <= physics["table_width"] / 2 - margin
    landed |= lands & passed & inside
    landings[lands] = at_land[lands]
    live &= ~lands & (q[:, 0] <= far + 0.5) & (q[:, 2] >= physics["floor_height"])
    p, v = q, w
  return landed, landings, net_heights


def sample_group(position, members, bank, rng, config, direction_fn, direction_id):
  options, physics = config["rally_constraints"], config["physical_ball"]
  sweep = config.get("sweep")
  count = sweep["speed_levels"] if sweep else 1
  for _ in range(options["max_direction_attempts"]):
    region = int(rng.integers(len(members))) if members else 0
    n, direction, bounds = direction_fn(members, region, config, rng)
    speeds = np.linspace(*bounds, options["speed_probe_levels"])
    incoming = bank[int(rng.integers(len(bank)))]
    ok, _, _ = return_outcomes(
      position, n, speeds[:, None] * direction, incoming, physics, options
    )
    candidates = np.flatnonzero(ok)
    if len(candidates) < count:
      continue
    indices = candidates[
      np.round(np.linspace(0, len(candidates) - 1, count)).astype(int)
    ]
    if not sweep:
      indices = rng.choice(candidates, size=1)
    selected = speeds[indices]
    if count > 1 and selected[-1] - selected[0] < options.get("minimum_speed_span", 0):
      continue
    # Verify selected endpoints at the physical substep size and exact contact center.
    ok, landings, heights = return_outcomes(
      position, n, selected[:, None] * direction, incoming, physics, options, dt=0.001
    )
    history = incoming_history(contact_center(position, n, physics), incoming, physics)
    if not ok.all() or history is None:
      continue
    witness = dict(
      direction_id=direction_id,
      normal=n.tolist(),
      velocity_direction=direction.tolist(),
      speed_levels=selected.tolist(),
      incoming=history,
      predicted_landings=landings.tolist(),
      predicted_net_heights=heights.tolist(),
    )
    return n, direction, selected, witness
  return None
