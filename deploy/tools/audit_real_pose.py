"""Offline ground/root pose audit; never changes logs or calibration."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import mujoco
import numpy as np

from deploy.g1_pingpong.perception import RootPoseCache, sample_received_time
from deploy.hope_g1.assets import training_collision_spec
from deploy.tools.view_real_log import LogScene, Recording, json_rows


def describe(values):
  a = np.asarray(values, dtype=float)
  if not a.size:
    return None
  return dict(
    n=len(a),
    median=float(np.median(a)),
    p05=float(np.quantile(a, 0.05)),
    p95=float(np.quantile(a, 0.95)),
    max_abs=float(np.max(np.abs(a))),
  )


def root_history(recording, offset=None):
  cfg = recording.summary["mocap_runtime_config"]
  cache = RootPoseCache(
    cfg["root_marker_to_pelvis"] if offset is None else offset,
    cfg["root_stale_timeout"],
    cfg.get("root_marker_to_pelvis_quaternion_wxyz", [1, 0, 0, 0]),
    conditioning=cfg.get("root_pose_filter"),
  )
  history = {}
  for item in json_rows(recording.directory / "mocap_packets.jsonl"):
    p = item["packet"]
    if p.get("kind") != "root":
      continue
    if p["frame"] != cfg["world_frame"]:
      continue
    try:
      received = sample_received_time(
        p["stamp"],
        cfg,
        wall_time=item["wall_time"],
        monotonic_time=item["arrival_time"],
      )
    except ValueError:
      continue
    cache.update(p["position"], p["quaternion_wxyz"], p["stamp"], received=received)
    if cache.latest is not None and cache.latest[2] == p["stamp"]:
      history[p["stamp"]] = cache.latest[0].copy()
  return history


def collision_model(recording, urdf):
  from deploy.hope_g1.scene import table_cfg
  from deploy.hope_g1.world import mujoco_ball_config
  from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
    get_g1_pingpong_robot_cfg,
  )
  from mjlab.scene import Scene, SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  robot = get_g1_pingpong_robot_cfg()
  robot.spec_fn = partial(training_collision_spec, robot.spec_fn, urdf, True)
  # No physics is stepped. Only geometry/FK is compared, without actor weights.
  return Scene(
    SceneCfg(
      entities={
        "robot": robot,
        "table": table_cfg(
          mujoco_ball_config(
            recording.summary["rally_runtime_config"]["ball"], recording.offset
          )
        ),
      },
      terrain=TerrainEntityCfg(terrain_type="plane"),
    ),
    device="cpu",
  ).compile()


def foot_height(model, data, side, *, collision=False):
  body = model.body(f"robot/{side}_ankle_roll_link").id
  points = []
  for gid in np.flatnonzero(model.geom_bodyid == body):
    name = model.geom(gid).name
    if collision:
      if not name.startswith("robot/hope_training_shape_"):
        continue
    elif name != f"robot/{side}_ankle_roll_link_0_visual":
      continue
    position = data.geom_xpos[gid]
    rotation = data.geom_xmat[gid].reshape(3, 3)
    kind = model.geom_type[gid]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
      mesh = model.geom_dataid[gid]
      start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
      points.append(
        float(
          np.min((model.mesh_vert[start : start + count] @ rotation.T + position)[:, 2])
        )
      )
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
      points.append(float(position[2] - np.abs(rotation[2]) @ model.geom_size[gid]))
    else:
      raise ValueError(f"Unsupported foot shape {name}")
  if not points:
    raise ValueError(f"No foot geometry for {side}")
  return min(points)


def audit(directory, urdf, nominal_offset=None):
  recording = Recording(directory)
  scene = LogScene(recording)
  model = collision_model(recording, urdf)
  data = mujoco.MjData(model)
  addresses = [
    model.jnt_qposadr[model.joint("robot/" + name).id] for name in recording.names
  ]
  root = model.jnt_qposadr[model.body_jntadr[model.body("robot/pelvis").id]]
  history = root_history(recording)
  hypothetical = (
    root_history(recording, nominal_offset) if nominal_offset is not None else {}
  )
  receiver_errors, fk_errors, viewer_q_errors, viewer_root_errors = [], [], [], []
  statics, missed_sources = [], 0
  for i, row in enumerate(json_rows(recording.directory / "trace.jsonl")):
    source = row.get("root_source_stamp")
    if (
      source is not None
      and row.get("root_pos") is not None
      and not row.get("root_pose_frozen")
    ):
      if source in history:
        receiver_errors.append(float(np.linalg.norm(history[source] - row["root_pos"])))
      else:
        missed_sources += 1
    if i % 5 or row.get("root_pos") is None or row.get("root_quat") is None:
      continue
    scene.update(row["time"] - recording.origin)
    viewer_q_errors.append(
      float(np.max(np.abs(scene.data.qpos[scene.addresses] - row["q"])))
    )
    viewer_root_errors.append(
      float(
        np.linalg.norm(
          scene.data.qpos[scene.root : scene.root + 3]
          + recording.offset
          - row["root_pos"]
        )
      )
    )
    if row.get("racket_pos") is not None:
      pos = (
        scene.data.site_xpos[scene.model.site("robot/right_racket_surface").id]
        + recording.offset
      )
      fk_errors.append(float(np.linalg.norm(pos - row["racket_pos"])))
    if (
      row.get("state") != "ARMED"
      or row.get("phase") != "IDLE"
      or np.max(np.abs(row["dq"])) > 0.3
      or np.linalg.norm(row["gyro"]) > 0.15
      or row.get("root_age", 1) > 0.1
    ):
      continue
    data.qpos[addresses] = row["q"]
    data.qpos[root : root + 7] = scene.data.qpos[scene.root : scene.root + 7]
    mujoco.mj_forward(model, data)
    visual = [foot_height(scene.model, scene.data, side) for side in ("left", "right")]
    contact = [
      foot_height(model, data, side, collision=True) for side in ("left", "right")
    ]
    entry = dict(
      time=row["time"] - recording.origin,
      visual_sole_z=visual,
      training_contact_z=contact,
      root_table_z=row["root_pos"][2],
      training_root_identity_error_m=float(
        np.linalg.norm(
          data.xpos[model.body("robot/pelvis").id]
          - scene.data.xpos[scene.model.body("robot/pelvis").id]
        )
      ),
    )
    if source in hypothetical:
      delta = hypothetical[source] - row["root_pos"]
      entry["hypothetical_visual_sole_z"] = (np.asarray(visual) + delta[2]).tolist()
      entry["hypothetical_root_delta"] = delta.tolist()
    statics.append(entry)

  def column(key, index=None):
    values = [v[key] if index is None else v[key][index] for v in statics if key in v]
    return describe(values)

  return dict(
    recording=str(recording.directory),
    evidence_scope="Stored normalized UDP root packets onward; raw VRPN/GUI rigid-body definition not recorded",
    receiver_position_error_m=describe(receiver_errors),
    unmatched_trace_sources=missed_sources,
    viewer_joint_error_rad=describe(viewer_q_errors),
    viewer_root_error_m=describe(viewer_root_errors),
    viewer_racket_vs_logged_fk_error_m=describe(fk_errors),
    static_selection="10Hz subsample; ARMED/IDLE; max |dq|<=0.3; |gyro|<=0.15; root_age<=0.1",
    static_frames=len(statics),
    visual_left_z_m=column("visual_sole_z", 0),
    visual_right_z_m=column("visual_sole_z", 1),
    contact_left_z_m=column("training_contact_z", 0),
    contact_right_z_m=column("training_contact_z", 1),
    root_identity_error_m=column("training_root_identity_error_m"),
    hypothetical_offset_m=nominal_offset,
    hypothetical_left_z_m=column("hypothetical_visual_sole_z", 0),
    hypothetical_right_z_m=column("hypothetical_visual_sole_z", 1),
    hypothesis_note="Replace the downstream offset in an offline receiver reprocess only; not a recommended calibration or edit to logs",
    static_samples=statics,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--recording", type=Path, required=True, action="append")
  parser.add_argument(
    "--training-urdf",
    type=Path,
    default=Path(
      "deploy/hope_g1/artifacts/hope_20261012_18000/robot/g1_29dof_pingpong_fast.urdf"
    ),
  )
  parser.add_argument("--nominal-offset", type=float, nargs=3)
  parser.add_argument("--marker-csv", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = dict(
    read_only=True,
    runs=[audit(p, args.training_urdf, args.nominal_offset) for p in args.recording],
  )
  if args.marker_csv:
    rows = list(csv.DictReader(args.marker_csv.open()))
    centroid = (
      np.mean([[float(r[f"sphere_{a}_mm"]) for a in "xyz"] for r in rows], axis=0)
      / 1000
    )
    result["marker_reference"] = dict(
      source=str(args.marker_csv),
      count=len(rows),
      sphere_centroid_in_pelvis_m=centroid.tolist(),
      negative_centroid_m=(-centroid).tolist(),
      assumption="Negative centroid is marker-to-pelvis translation only if rigid-body origin is this centroid and its axes equal pelvis axes",
    )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2))
  print(
    json.dumps(
      {
        **result,
        "runs": [
          {k: v for k, v in r.items() if k != "static_samples"} for r in result["runs"]
        ],
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
