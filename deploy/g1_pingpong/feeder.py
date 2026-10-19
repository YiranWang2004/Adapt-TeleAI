"""HOPE-like randomized legal incoming flights with physical table/net checks."""

import copy
from types import SimpleNamespace

import numpy as np

from deploy.backends.mujoco import SimBall


class RandomBallFeeder:
  def __init__(self, ball_cfg, options, rng):
    self.cfg, self.options, self.rng = ball_cfg, options, rng

  def sample(self, candidates):
    cfg = {
      **self.cfg,
      "incoming_speed_range": self.options["speed_range"],
      "incoming_lateral_speed_range": self.options["lateral_speed_range"],
      "intercept_jitter": self.options["intercept_jitter"],
    }
    net_x = (cfg["table_near_x"] + cfg["table_far_x"]) / 2
    dummy = SimpleNamespace(
      racket_pos=np.full(3, 100.0), racket_normal=np.array([1.0, 0, 0])
    )
    for _ in range(64):
      _, reference = candidates[int(self.rng.integers(len(candidates)))]
      point = reference.strike_position(cfg).copy()
      feed_reference = SimpleNamespace(
        strike_position=lambda cfg, point=point: point,
        strike_time=reference.strike_time,
      )
      try:
        ball = SimBall.receiving_feed(cfg, feed_reference, self.rng)
      except ValueError:
        continue
      # Backwards reconstruction can extend a short post-bounce flight so far
      # that its launch lies below the floor/table. The analytical ball has no
      # table-side or floor collision, so such a candidate could rise through
      # the table and falsely pass the subsequent bounce/net checks.
      if ball.position[2] < cfg["table_height"] + cfg["ball_radius"]:
        continue
      # Validate the generated initial state with the actual simulation ball
      # physics; only legal incoming balls with a P1-side bounce are published.
      probe = copy.deepcopy(ball)
      bounces = []
      while probe.elapsed < probe.arrival:
        vz = probe.velocity[2]
        probe.substep(dummy, 0.005)
        if vz < 0 < probe.velocity[2]:
          bounces.append(probe.position[0])
      if (
        ball.position[0] > net_x
        and len(bounces) == 1
        and cfg["table_near_x"] <= bounces[0] < net_x
        and not probe.net_hit
        and np.linalg.norm(probe.position - probe.intercept) < 0.08
      ):
        return ball
    raise ValueError("No legal incoming feed in the configured motion/speed region")
