"""Random launch states filtered only by incoming ball geometry.

No command targets, actor, selector, motion assets, planner outputs or robot
execution are accepted by this module. Oracle crossings are for evaluation only.
"""

import numpy as np
import torch

from deploy.g1_pingpong.ball_physics import (
  ballistic_step,
  swept_net_collision,
  swept_table_collision,
)


def sample_launches(config, physics):
  rng = np.random.default_rng(config["seed"])
  launch = config["launch"]
  ys = np.linspace(*config["y_range"], config["grid"][0] + 1)
  zs = np.linspace(*config["z_range"], config["grid"][1] + 1)
  buckets = {(y, z): [] for z in range(len(zs) - 1) for y in range(len(ys) - 1)}
  near, far = physics["table_near_x"], physics["table_far_x"]
  net, cy = (near + far) / 2, physics["table_center_y"]
  shift = np.array([0, cy, 0])
  size = launch["batch_size"]
  for batch in range(launch["max_batches"]):
    initial_p = rng.uniform(
      [launch[k][0] for k in ["x", "y", "z"]],
      [launch[k][1] for k in ["x", "y", "z"]],
      (size, 3),
    )
    initial_v = rng.uniform(
      [launch[k][0] for k in ["vx", "vy", "vz"]],
      [launch[k][1] for k in ["vx", "vy", "vz"]],
      (size, 3),
    )
    p = torch.tensor(initial_p - shift, dtype=torch.float32)
    v = torch.tensor(initial_v, dtype=torch.float32)
    live = torch.ones(size, dtype=torch.bool)
    bounces = torch.zeros(size, dtype=torch.int)
    receiving_bounce = torch.zeros(size, dtype=torch.bool)
    net_passed = torch.zeros(size, dtype=torch.bool)
    net_blocked = torch.zeros(size, dtype=torch.bool)
    dt = 0.001
    for step in range(round(launch["horizon_seconds"] / dt)):
      q, w = ballistic_step(p, v, dt, physics["gravity"], physics["drag"])
      crossed = live & (p[:, 0] > net) & (q[:, 0] <= net)
      f = ((p[:, 0] - net) / (p[:, 0] - q[:, 0]).clamp_min(1e-12)).clamp(0, 1)
      at = p + f[:, None] * (q - p)
      net_passed |= (
        crossed
        & (
          at[:, 2]
          > physics["table_height"]
          + physics["net_height"]
          + physics["ball_radius"]
          + 0.02
        )
        & (at[:, 1].abs() < physics["table_width"] / 2 - 0.02)
      )
      q, w, blocked, _ = swept_net_collision(
        p,
        q,
        w,
        dt,
        net_x=net,
        width=physics["table_width"] + 2 * physics.get("net_overhang", 0),
        table_height=physics["table_height"],
        net_height=physics["net_height"],
        ball_radius=physics["ball_radius"],
      )
      net_blocked |= blocked
      q, w, hit, bounce = swept_table_collision(
        p,
        q,
        w,
        dt,
        near_x=near,
        far_x=far,
        width=physics["table_width"],
        height=physics["table_height"],
        ball_radius=physics["ball_radius"],
        restitution=physics["table_restitution"],
        tangential_damping=physics["table_tangential_damping"],
      )
      bounces += (live & hit).int()
      receiving_bounce |= live & hit & (bounce[:, 0] < net)
      crosses = live & (p[:, 0] > config["plane_x"]) & (q[:, 0] <= config["plane_x"])
      eligible = crosses & net_passed & ~net_blocked & (bounces == 1) & receiving_bounce
      for index in torch.nonzero(eligible).flatten().tolist():
        f = float((p[index, 0] - config["plane_x"]) / (p[index, 0] - q[index, 0]))
        crossing_time = (step + f) * dt
        if crossing_time < launch.get("minimum_crossing_seconds", 0.0):
          continue
        point = (p[index] + f * (q[index] - p[index])).numpy() + shift
        if not ys[0] <= point[1] < ys[-1] or not zs[0] <= point[2] < zs[-1]:
          continue
        y, z = (
          np.searchsorted(ys, point[1], side="right") - 1,
          np.searchsorted(zs, point[2], side="right") - 1,
        )
        if len(buckets[(y, z)]) >= config["balls_per_cell"]:
          continue
        buckets[(y, z)].append(
          dict(
            cell_y=int(y),
            cell_z=int(z),
            initial_position=initial_p[index].tolist(),
            initial_velocity=initial_v[index].tolist(),
            oracle_crossing_position=point.tolist(),
            oracle_crossing_time=crossing_time,
            source_batch=batch,
            source_index=index,
          )
        )
      live &= (
        ~crosses
        & ~net_blocked
        & (q[:, 2] > physics.get("floor_height", -0.76) + physics["ball_radius"])
      )
      p, v = q, w
      if not live.any():
        break
    if all(len(items) == config["balls_per_cell"] for items in buckets.values()):
      break
  if not all(len(items) == config["balls_per_cell"] for items in buckets.values()):
    raise ValueError(
      f"Insufficient independent serves: {[len(v) for v in buckets.values()]}"
    )
  serves = [s for group in buckets.values() for s in group]
  rng.shuffle(serves)
  for i, serve in enumerate(serves):
    serve["trial_id"] = i
  return serves
