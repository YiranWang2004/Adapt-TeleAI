"""Export training command support, independently of subsequent trial outcomes."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from deploy.hope_g1.command_adapter import HopeCommandAdapter


def section(center, rotation, half_extent, plane_x):
  corners = list(itertools.product((-1, 1), repeat=3))
  vertices = [center + rotation @ (np.asarray(c) * half_extent) for c in corners]
  points = []
  for i, a in enumerate(corners):
    for j, b in enumerate(corners[i + 1 :], i + 1):
      if sum(x != y for x, y in zip(a, b, strict=True)) != 1:
        continue
      p, q = vertices[i], vertices[j]
      if abs(p[0] - plane_x) < 1e-10:
        points.append(p[1:])
      if (p[0] - plane_x) * (q[0] - plane_x) < 0:
        points.append((p + (q - p) * (plane_x - p[0]) / (q[0] - p[0]))[1:])
      if abs(q[0] - plane_x) < 1e-10:
        points.append(q[1:])
  if len(points) < 3:
    return []
  points = np.unique(np.round(points, 12), axis=0)
  mean = points.mean(axis=0)
  return points[
    np.argsort(np.arctan2(points[:, 1] - mean[1], points[:, 0] - mean[0]))
  ].tolist()


def export(agent, plane_x, scale):
  engine = agent.baseline
  racket = engine.cfg["hope"]["racket"]
  if racket["ref_velocity_perturb_mode"] != "scale":
    raise ValueError("Exporter requires saved scale-mode velocity randomization")
  regions = []
  excluded = []
  for index, reference in engine.candidates(agent.state()):
    goal = engine.reference_goal(reference, index)
    extent = np.asarray(racket["ref_perturb_pos"]) * scale
    polygon = section(goal.position, reference.rotation, extent, plane_x)
    if not polygon:
      excluded.append(index)
      continue
    direction = goal.velocity / np.linalg.norm(goal.velocity)
    speed = racket["ref_velocity_speed"]
    half = racket["ref_perturb_vel_scale"] * scale
    lead = reference.strike_time(engine.cfg["ball"])
    regions.append(
      dict(
        source_motion=index,
        source_file=engine.library.paths[index].name,
        position_center=goal.position.tolist(),
        rotation=reference.rotation.tolist(),
        position_half_extent=extent.tolist(),
        polygon_yz=polygon,
        normal_center=direction.tolist(),
        normal_half_extent=racket["ref_perturb_normal"] * scale,
        velocity_direction=direction.tolist(),
        speed_range=[speed * (1 - half), speed * (1 + half)],
        tts_range=[
          max(0.15, lead - engine.options["max_start_frame"] / reference.fps),
          min(lead, engine.a3_options["max_command_time_to_strike"]),
        ],
      )
    )
  if not regions:
    raise ValueError("No training position support intersects requested plane")
  points = np.concatenate([r["polygon_yz"] for r in regions])
  source = Path(
    "deploy/hope_g1/artifacts/hope_20261012_18000/audit_sources/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py"
  )
  return dict(
    schema_version=1,
    world_frame="hope_table",
    plane_x=plane_x,
    curriculum_scale=scale,
    description="Union of saved-motion position perturbation boxes intersected with fixed plane; theoretical training support, not demonstrated reachability",
    vector_sampling="Reference velocity scaled independently of normalized additive normal perturbation; joint support conditioned only on position",
    y_range=[float(points[:, 0].min()), float(points[:, 0].max())],
    z_range=[float(points[:, 1].min()), float(points[:, 1].max())],
    speed_range=[
      min(r["speed_range"][0] for r in regions),
      max(r["speed_range"][1] for r in regions),
    ],
    tts_range=[
      max(r["tts_range"][0] for r in regions),
      min(r["tts_range"][1] for r in regions),
    ],
    excluded_motions_no_plane_intersection=excluded,
    regions=regions,
    alignment=dict(
      home_index=engine.home_index,
      translation=engine.home.translation.tolist(),
      rotation=engine.home.rotation.tolist(),
    ),
    training_parameters={
      k: racket[k]
      for k in [
        "ref_perturb_pos",
        "ref_velocity_speed",
        "ref_velocity_perturb_mode",
        "ref_perturb_vel_scale",
        "ref_perturb_normal",
      ]
    },
    source_command_implementation=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    checkpoint=str(agent.deployment.checkpoint),
    checkpoint_sha256=hashlib.sha256(
      agent.deployment.checkpoint.read_bytes()
    ).hexdigest(),
    validation_scope="Exported before evaluation; no command success, actor tracking error or selected-motion outcomes used",
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output", type=Path, default=Path("deploy/hope_g1/configs/command_domain.json")
  )
  parser.add_argument("--plane-x", type=float, default=-0.25)
  parser.add_argument("--scale", type=float, default=1.0)
  parser.add_argument("--home-index", type=int, default=0)
  args = parser.parse_args()
  if not np.isfinite(args.scale) or not 0 < args.scale <= 1:
    parser.error("scale must be in (0,1]")
  import torch

  torch.set_num_threads(1)
  base = Path("deploy/hope_g1/configs")
  agent = HopeCommandAdapter(
    base / "task.yaml",
    base / "rally_mujoco.yaml",
    base / "real.yaml",
    base / "mocap.yaml",
    Path("deploy/recordings/racket_command/domain_export"),
    home_index=args.home_index,
  )
  result = export(agent, args.plane_x, args.scale)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2))
  print(
    json.dumps(
      {
        k: result[k]
        for k in ["y_range", "z_range", "speed_range", "tts_range", "curriculum_scale"]
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
