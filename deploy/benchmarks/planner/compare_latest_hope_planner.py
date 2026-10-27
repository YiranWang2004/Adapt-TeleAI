"""Replay real ball packets through the latest read-only HOPE planner snapshot.

The comparison samples the new planner at the original deployment's
``motion_selected`` epochs and compares its racket command with the measured
ball crossing of the virtual hit plane. No robot, ROS node, or MuJoCo policy is
started by this diagnostic.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECORDINGS = [
    ROOT / "deploy/recordings/hope_g1/real_20261022_121714",
    ROOT / "deploy/recordings/hope_g1/real_20261022_121834",
]
DEFAULT_SOURCE = ROOT / "deploy/hope_g1/artifacts/hope_planner_optimized_2_A30808"
DEFAULT_OUTPUT = ROOT / "deploy/recordings/planner/latest_hope_planner_vs_real_20261022"


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def load_ball_packets(path: Path) -> list[tuple[float, np.ndarray]]:
  rows = []
  with path.open() as stream:
    for line in stream:
      record = json.loads(line)
      packet = record["packet"]
      if packet.get("kind") == "ball":
        position = np.asarray(packet["position"], dtype=float)
        if position.shape == (3,) and np.isfinite(position).all():
          rows.append((float(packet["stamp"]), position))
  rows.sort(key=lambda row: row[0])
  return rows


def make_planner(source: Path, *, hit_plane_x: float):
  sys.path.insert(0, str(source))
  from hope_planner_optimized_2.constants import (  # noqa: PLC0415
      BallPhysics,
      PlannerConfig,
      TableParams,
  )
  from hope_planner_optimized_2.planner import HOPEPlanner  # noqa: PLC0415

  config_path = source / "config/hope_planner.yaml"
  params = yaml.safe_load(config_path.read_text())["hope_planner_optimized_2"][
      "ros__parameters"
  ]
  config = PlannerConfig(
      x_hit=float(hit_plane_x),
      target_land=np.array(
          [params["target_land_x"], params["target_land_y"], 0.0], dtype=float
      ),
      delta_t_flight=float(params["delta_t_flight"]),
      C_r=float(params["restitution_racket"]),
      dt_integrate=float(params["dt_integrate"]),
      stage3_dt_integrate=float(params["stage3_dt_integrate"]),
      outgoing_velocity_tolerance_m=float(params["outgoing_velocity_tolerance_m"]),
      time_to_strike_invalid_x=float(params["time_to_strike_invalid_x"]),
      publish_intercept_x_min=float(hit_plane_x),
      publish_intercept_x_max=float(hit_plane_x),
      publish_intercept_y_min=float(params["publish_intercept_y_min"]),
      publish_intercept_y_max=float(params["publish_intercept_y_max"]),
      publish_intercept_z_min=float(params["publish_intercept_z_min"]),
      publish_intercept_z_max=float(params["publish_intercept_z_max"]),
  )
  physics = BallPhysics(
      k=float(params["drag_k"]),
      C_h=float(params["restitution_h"]),
      C_v=float(params["restitution_v"]),
  )
  return HOPEPlanner(physics, config, TableParams()), params


def first_measured_crossing(rows, after_stamp: float, plane_x: float):
  for (t0, p0), (t1, p1) in zip(rows[:-1], rows[1:], strict=False):
    dt = t1 - t0
    if (
        t0 >= after_stamp
        and t1 - after_stamp <= 1.5
        and 0.0 < dt <= 0.05
        and p0[0] > plane_x >= p1[0]
        and -1.7 <= p1[1] <= 0.2
        and 0.0 <= p1[2] <= 1.2
    ):
      fraction = (p0[0] - plane_x) / (p0[0] - p1[0])
      return {
          "stamp": float(t0 + fraction * dt),
          "position": (p0 + fraction * (p1 - p0)).tolist(),
          "bracket_dt_s": float(dt),
      }
  return None


def replay_recording(recording: Path, source: Path, config_params: dict):
  summary = json.loads((recording / "summary.json").read_text())
  rows = load_ball_packets(recording / "mocap_packets.jsonl")
  if not rows:
    return {"recording": str(recording), "error": "no ball packets"}, []
  events = [
      event for event in summary.get("events", [])
      if event.get("event") == "motion_selected"
  ]
  planner, _ = make_planner(source, hit_plane_x=-0.25)
  base_stamp = rows[0][0]
  plane_x = -0.25
  results = []
  packet_index = 0

  for event in events:
    old = event["raw_command"]
    event_stamp = float(old["estimate_stamp"])
    command = None
    last_diag = {}
    last_stamp = None
    while packet_index < len(rows) and rows[packet_index][0] <= event_stamp + 1e-7:
      stamp, position = rows[packet_index]
      command = planner.update(stamp - base_stamp, position)
      last_diag = planner.last_update_diagnostics
      last_stamp = stamp
      packet_index += 1

    crossing = first_measured_crossing(rows, event_stamp, plane_x)
    item = {
        "recording": recording.name,
        "motion_index": event.get("motion_index"),
        "trigger_stamp": event_stamp,
        "trigger_relative_s": event_stamp - base_stamp,
        "planner_input_stamp": last_stamp,
        "planner_input_lag_s": None if last_stamp is None else event_stamp - last_stamp,
        "planner_command_published": command is not None,
        "planner_decision": last_diag.get("command_reason"),
        "planner_buffer_reset_reason": last_diag.get("buffer_reset_reason"),
        "planner_buffer_points": last_diag.get("buffer_points"),
        "planner_fit_rmse_m": planner.estimator.fit_rmse_m,
        "measured_crossing": crossing,
        "old_deploy_command": {
            "position": old.get("position"),
            "velocity": old.get("velocity"),
            "normal": old.get("striking_normal"),
            "tts_s": old.get("remaining"),
        },
    }
    if command is not None:
      position_table = command.p_intercept.copy()
      tts = float(command.t_strike - (event_stamp - base_stamp))
      item["latest_planner_command"] = {
          "position": position_table.tolist(),
          "velocity": command.v_racket.tolist(),
          "normal": command.n_racket.tolist(),
          "tts_s": tts,
          "strike_time_relative_s": float(command.t_strike),
          "valid": bool(command.valid),
          "clears_net": bool(command.clears_net),
          "bypasses_net_posts": bool(command.bypasses_net_posts),
          "predicted_bounces": int(command.num_bounces),
          "stage1_fit_rmse_m": float(command.stage1_fit_rmse_m),
          "estimated_incoming_velocity": planner.latest_velocity.tolist(),
      }
      if crossing is not None:
        crossing_tts = crossing["stamp"] - event_stamp
        position_error = float(
            np.linalg.norm(position_table - np.asarray(crossing["position"]))
        )
        tts_error = tts - crossing_tts
        item["comparison"] = {
            "actual_tts_s": float(crossing_tts),
            "position_error_m": position_error,
            "tts_error_s": float(tts_error),
            "old_position_error_m": float(
                np.linalg.norm(
                    np.asarray(old["position"]) - np.asarray(crossing["position"])
                )
            ),
            "old_tts_error_s": float(old["remaining"] - crossing_tts),
        }
    results.append(item)
  return {
      "recording": str(recording),
      "recording_sha256": {
          name: sha256(recording / name)
          for name in ("summary.json", "mocap_packets.jsonl")
      },
      "motion_selected_count": len(events),
      "results": results,
  }, results


def summarize(results):
  compared = [r for r in results if "comparison" in r]
  def metric(key):
    values = [r["comparison"][key] for r in compared]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "mean_abs": float(np.mean(np.abs(values))) if values else None,
        "max_abs": float(np.max(np.abs(values))) if values else None,
    }
  return {
      "events": len(results),
      "new_planner_commands": sum(r["planner_command_published"] for r in results),
      "measured_plane_crossings": sum(r["measured_crossing"] is not None for r in results),
      "paired_comparisons": len(compared),
      "latest_position_error_m": metric("position_error_m"),
      "old_position_error_m": metric("old_position_error_m"),
      "latest_tts_error_s": metric("tts_error_s"),
      "old_tts_error_s": metric("old_tts_error_s"),
  }


def plot_results(rows, path: Path):
  fig, axes = plt.subplots(max(1, len(rows)), 2, figsize=(10, 3.5 * max(1, len(rows))))
  axes = np.atleast_2d(axes)
  for i, item in enumerate(rows):
    ax = axes[i, 0]
    actual = item.get("measured_crossing")
    old = item.get("old_deploy_command", {})
    latest = item.get("latest_planner_command")
    if actual:
      ax.scatter(actual["position"][1], actual["position"][2], marker="*", s=120, label="Measured crossing")
    if old.get("position"):
      ax.scatter(old["position"][1], old["position"][2], marker="x", s=65, label="Previous deploy")
    if latest:
      ax.scatter(latest["position"][1], latest["position"][2], marker="o", s=55, label="A30808 planner")
    ax.set_xlabel("Y (hope_table, m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"{item['recording']} / motion {item['motion_index']}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[i, 1]
    if actual:
      actual_tts = actual["stamp"] - item["trigger_stamp"]
      bars = [("Measured", actual_tts)]
      if old.get("tts_s") is not None:
        bars.append(("Previous", old["tts_s"]))
      if latest:
        bars.append(("A30808", latest["tts_s"]))
      ax.bar([b[0] for b in bars], [b[1] for b in bars], color=["#3b82f6", "#f59e0b", "#16a34a"][:len(bars)])
      ax.set_ylabel("TTS (s)")
      ax.set_ylim(bottom=0)
    else:
      ax.text(0.5, 0.5, "No measured plane crossing", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, axis="y", alpha=0.3)
  fig.suptitle("A30808 planner vs measured real trajectory", fontsize=13)
  fig.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--recording", type=Path, action="append", default=None)
  parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  recordings = args.recording or DEFAULT_RECORDINGS
  _, params = make_planner(args.source, hit_plane_x=-0.25)
  source_yaml_params = yaml.safe_load(
      (args.source / "config/hope_planner.yaml").read_text()
  )["hope_planner_optimized_2"]["ros__parameters"]
  effective_params = dict(params)
  effective_params["x_hit"] = -0.25
  effective_params["publish_intercept_x_min"] = -0.25
  effective_params["publish_intercept_x_max"] = -0.25
  per_recording = []
  flat = []
  for recording in recordings:
    result, rows = replay_recording(recording, args.source, params)
    per_recording.append(result)
    flat.extend(rows)
  files = [
      args.source / "config/hope_planner.yaml",
      *sorted((args.source / "hope_planner_optimized_2").glob("*.py")),
  ]
  report = {
      "planner_snapshot": str(args.source),
      "planner_source_sha256": {str(p.relative_to(args.source)): sha256(p) for p in files if p.is_file()},
      "planner_config": effective_params,
      "source_yaml_config": source_yaml_params,
      "coordinate_mapping": {
          "source_frame": "hope_table",
          "planner_frame": "HOPE canonical",
          "position_to_planner": "[x, y, z] (identity; table origins and axes agree)",
          "command_to_source": "[x, y, z] (identity)",
          "plane_source_x": -0.25,
          "plane_planner_x": -0.25,
          "source_yaml_x_hit": float(source_yaml_params["x_hit"]),
          "adaptation": "Override only x_hit and its inclusive publish gate to -0.25 m for the G1 racket plane; keep measured ball positions, table geometry, and target coordinates unchanged.",
          "basis": "G1 hope_table and HOPE canonical both use table X=[0,2.74], Y=[-1.525,0], tabletop Z=0; only the configured racket plane differs.",
      },
      "method": "Replay measured ball packets causally through exact A30808 HOPEPlanner at original motion_selected trigger epochs; compare predicted position/TTS with the next measured crossing of hope_table x=-0.25. No future samples are passed to planner.",
      "per_recording": per_recording,
      "summary": summarize(flat),
  }
  args.output.mkdir(parents=True, exist_ok=True)
  (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
  plot_results(flat, args.output / "comparison.png")
  print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
