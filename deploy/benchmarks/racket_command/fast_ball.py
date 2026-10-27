"""Trace the existing portable ball kernels together; no new dynamics model."""

from functools import lru_cache

import torch

from deploy.g1_pingpong.ball_physics import (
  ballistic_step,
  swept_net_collision,
  swept_racket_collision,
  swept_table_collision,
)


@lru_cache(maxsize=8)
def compiled_step(parameters):
  c = dict(parameters)

  def step(p0, v0, center0, center1, normal, dt, enabled):
    p, v = ballistic_step(p0, v0, dt, c["gravity"], c["drag"])
    pr, vr, hit = swept_racket_collision(
      p0,
      p,
      v,
      center0,
      center1,
      normal,
      dt,
      ball_radius=c["ball_radius"],
      racket_radius=c["racket_radius"],
      restitution=c["racket_restitution"],
    )
    hit = hit & enabled
    pr = torch.where(hit[:, None], pr, p)
    vr = torch.where(hit[:, None], vr, v)
    pn, vn, net_hit, net_clear = swept_net_collision(
      p0,
      pr,
      vr,
      dt,
      net_x=(c["table_near_x"] + c["table_far_x"]) / 2,
      width=c["table_width"] + 2 * c["net_overhang"],
      table_height=c["table_height"],
      net_height=c["net_height"],
      ball_radius=c["ball_radius"],
    )
    pt, vt, table_hit, _ = swept_table_collision(
      p0,
      pn,
      vn,
      dt,
      near_x=c["table_near_x"],
      far_x=c["table_far_x"],
      width=c["table_width"],
      height=c["table_height"],
      ball_radius=c["ball_radius"],
      restitution=c["table_restitution"],
      tangential_damping=c["table_tangential_damping"],
    )
    pf, vf, floor_hit, _ = swept_table_collision(
      p0,
      pt,
      vt,
      dt,
      near_x=-1e6,
      far_x=1e6,
      width=2e6,
      height=c["floor_height"],
      ball_radius=c["ball_radius"],
      restitution=c["floor_restitution"],
      tangential_damping=0.8,
    )
    return pr, vr, hit, pn, vn, net_hit, net_clear, pf, vf, table_hit, floor_hit

  vector = torch.zeros((1, 3))
  traced = torch.jit.trace(
    step,
    (
      vector,
      vector,
      vector,
      vector,
      torch.tensor([[1.0, 0.0, 0.0]]),
      torch.tensor(0.001),
      torch.tensor([False]),
    ),
    check_trace=False,
  )
  return torch.compile(traced, fullgraph=True)


def parameters(config):
  keys = (
    "gravity",
    "drag",
    "ball_radius",
    "racket_radius",
    "racket_restitution",
    "table_near_x",
    "table_far_x",
    "table_width",
    "net_overhang",
    "table_height",
    "net_height",
    "table_restitution",
    "table_tangential_damping",
    "floor_height",
    "floor_restitution",
  )
  return tuple((key, float(config.get(key, 0.0))) for key in keys)
