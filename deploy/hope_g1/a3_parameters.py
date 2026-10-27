"""Shared ball/table parameters from the read-only A3 launch configuration."""

from pathlib import Path

import yaml

from deploy.hope_g1.a3_reference.constants import (
  BallPhysics,
  PlannerConfig,
  TableParams,
)

PROFILE = Path(__file__).with_name("configs") / "a3_planner.yaml"


def ball_parameters():
  params = yaml.safe_load(PROFILE.read_text())["hope_planner_optimized"][
    "ros__parameters"
  ]
  physics, table, planner = BallPhysics(), TableParams(), PlannerConfig()
  return dict(
    gravity=float(-physics.g[2]),
    drag=params["drag_k"],
    ball_radius=physics.radius,
    racket_radius=planner.racket_radius,
    racket_restitution=params["restitution_racket"],
    table_restitution=params["restitution_v"],
    table_tangential_damping=params["restitution_h"],
    table_near_x=0.0,
    table_far_x=table.length,
    table_center_y=-table.width / 2,
    table_width=table.width,
    table_height=0.0,  # Tabletop origin; simulator floor transform is separate.
    net_height=table.net_height,
    net_overhang=table.net_overhang,
  )
