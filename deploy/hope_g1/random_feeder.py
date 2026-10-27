"""Continuous deployment's independent random incoming balls.

Only launch geometry and ball physics enter sampling. No benchmark environment,
motion references, policy state or planner commands are used here.
"""

import numpy as np

from deploy.backends.mujoco import SimBall
from deploy.g1_pingpong.launch_sampling import sample_launches


class RandomLaunchFeeder:
  independent_launch = True
  requires_motion_geometry = False

  def __init__(self, ball_cfg, options, rng):
    self.cfg, self.options, self.rng = ball_cfg, options, rng
    self.translation = np.asarray(options["world_translation"], dtype=float)
    self.physics = dict(ball_cfg)
    for name, axis in (
      ("table_near_x", 0),
      ("table_far_x", 0),
      ("table_center_y", 1),
      ("table_height", 2),
    ):
      self.physics[name] += self.translation[axis]
    self.physics["floor_height"] = self.translation[2]
    self.pending = []
    self.refill()

  def refill(self):
    config = {
      key: self.options[key]
      for key in ("plane_x", "y_range", "z_range", "grid", "balls_per_cell", "launch")
    }
    config["seed"] = int(self.rng.integers(0, 2**31))
    # Geometry-stratified random launches; each new bank has a fresh seed.
    # The free-flight crossing is never passed to the controller.
    self.pending = sample_launches(config, self.physics)

  def sample(self, _candidates=None):
    if not self.pending:
      self.refill()
    feed = self.pending.pop()
    ball = SimBall.from_launch(
      self.cfg,
      np.asarray(feed["initial_position"]) - self.translation,
      feed["initial_velocity"],
      lifetime=self.cfg["max_ball_time"],
    )
    ball.feed_metadata = dict(mode="random_launch", world_frame="hope_table", **feed)
    return ball

  def draw_grid(self, scene):
    """Unlabelled partition, in the exact configured sampling coordinates."""
    import mujoco

    ys = np.linspace(*self.options["y_range"], self.options["grid"][0] + 1)
    zs = np.linspace(*self.options["z_range"], self.options["grid"][1] + 1)
    x = self.options["plane_x"]
    lines = [([x, y, zs[0]], [x, y, zs[-1]]) for y in ys]
    lines += [([x, ys[0], z], [x, ys[-1], z]) for z in zs]
    for start, end in lines:
      if scene.ngeom >= scene.maxgeom:
        break
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.ones(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.array([0.7, 0.7, 0.7, 0.65], dtype=np.float32),
      )
      mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.0015,
        np.asarray(start) - self.translation,
        np.asarray(end) - self.translation,
      )
      scene.ngeom += 1
