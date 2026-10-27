"""Incoming feeds based only on reference strike geometry, never return outcomes."""

from types import SimpleNamespace

import numpy as np

from deploy.g1_pingpong.core import rotation
from deploy.g1_pingpong.feeder import RandomBallFeeder


class MotionPlaneFeeder(RandomBallFeeder):
  requires_motion_geometry = True
  requires_over_net = True

  def prepare(self, index, reference, cfg):
    frame = round(reference.strike_time(cfg) * reference.fps)
    source = reference.source if hasattr(reference, "source") else reference
    body = source.body_names.index(cfg["racket_body_name"])
    world_rotation = getattr(reference, "rotation", np.eye(3))
    normal = (
      world_rotation
      @ rotation(source.body_quat[frame, body])[:, cfg["racket_normal_axis"]]
    )
    normal /= np.linalg.norm(normal)
    # The analytical disk has two faces. Choose the face towards the opponent
    # solely to define incoming approach direction, not to change the paddle.
    if normal[0] < 0:
      normal = -normal
    if normal[0] < 0.15:
      raise ValueError("Strike plane faces away from the opponent feed region")
    center = reference.strike_position(cfg).copy()
    tangent_y = np.cross([0.0, 0.0, 1.0], normal)
    tangent_y /= np.linalg.norm(tangent_y)
    tangent_z = np.cross(normal, tangent_y)
    jitter = np.asarray(self.options["intercept_jitter"])
    point = (
      center
      + tangent_y * self.rng.uniform(-jitter[1], jitter[1])
      + tangent_z * self.rng.uniform(-jitter[2], jitter[2])
    )
    speed = self.rng.uniform(*self.options["speed_range"])
    # Match incoming azimuth to the reference plane normal. Vertical velocity
    # still follows the legal one-bounce gravity trajectory, not an outgoing
    # target solver. Neither actor measurements nor rebound quality is used.
    # Keep that azimuth within the finite incoming net corridor. An oblique
    # arrival is allowed; do not send wide strokes around the side of the net.
    net_x = (cfg["table_near_x"] + cfg["table_far_x"]) / 2
    distance = net_x - point[0]
    if distance <= 0:
      raise ValueError("Reference strike point is not on the receiving side")
    half_width = cfg["table_width"] / 2 - cfg["ball_radius"] - 0.03
    ratio = np.clip(
      normal[1] / normal[0],
      (-half_width - point[1]) / distance,
      (half_width - point[1]) / distance,
    )
    vy = -speed * ratio
    candidate_cfg = {
      **cfg,
      "incoming_speed_range": [speed, speed],
      "incoming_lateral_speed_range": [vy, vy],
      "intercept_jitter": [0.0, 0.0, 0.0],
    }
    feed_reference = SimpleNamespace(
      strike_position=lambda cfg: point.copy(), strike_time=reference.strike_time
    )
    return (
      candidate_cfg,
      feed_reference,
      dict(
        mode="motion_plane",
        motion_index=index,
        strike_frame=frame,
        plane_center=center.tolist(),
        plane_normal=normal.tolist(),
        intended_contact=point.tolist(),
        signed_plane_offset_m=float((point - center) @ normal),
        incoming_horizontal_velocity=[-speed, vy],
      ),
    )
