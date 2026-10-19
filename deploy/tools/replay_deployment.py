"""Replay real snapshots through the shared loop; never open SDK or ROS IO."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from deploy.backends.runtime import SimulationIO
from deploy.backends.unitree import run_real
from deploy.g1_pingpong.configuration import load_deployment
from deploy.g1_pingpong.perception import BallEstimate


class RecordedIO:
  def __init__(self, rows):
    self.rows, self.index = rows, 0
    self.closed = self.halted = False
    self.checked = 0
    self.errors = {}
    self.mismatches = []

  def monotonic(self):
    return (
      self.rows[self.index]["time"]
      if self.index < len(self.rows)
      else self.rows[-1]["time"] + 0.02
    )

  def time(self):
    return 10000 + self.monotonic()

  def sleep(self, seconds):
    self.index += 1

  def bind(self, engine):
    self.engine = engine

  def source(self):
    return self.rows[min(self.index, len(self.rows) - 1)]

  def feedback(self):
    r = self.source()
    return (
      np.array(r["q"]),
      np.array(r["dq"]),
      np.array(r.get("imu_gyro", r["gyro"])),
      r["buttons"],
      r["mode_machine"],
      np.array(r["imu_quat"]),
      r["time"] - r["dds_state_age"],
    )

  def root(self, now):
    r = self.source()
    if r.get("root_pos") is None:
      return None
    return (
      np.array(r["root_pos"]),
      np.array(r.get("root_quat_raw_mocap", r["root_quat"])),
      r.get("root_source_stamp", now),
      now - (r.get("root_age") or 0),
    )

  def ball(self, now):
    r = self.source()
    if r.get("ball_position") is None:
      return None
    return BallEstimate(
      np.array(r["ball_position"]),
      np.array(r["ball_velocity"]),
      r.get("ball_source_stamp", now),
      now - (r.get("ball_age") or 0),
    )

  def send(self, target, kp=None, kd=None):
    return self.monotonic()

  def close(self):
    self.closed = True

  def record(self, row):
    expected = self.source()
    self.checked += 1
    for key in ["state", "policy", "command_policy", "reference_time"]:
      if key in expected and row.get(key) != expected[key]:
        self.mismatches.append(
          dict(frame=self.index, key=key, actual=row.get(key), expected=expected[key])
        )
    for key in [
      "target_q",
      "kp",
      "kd",
      "tracker_obs",
      "joint_action",
      "policy_target_before_handoff_q",
    ]:
      if expected.get(key) is None:
        continue
      if row.get(key) is None:
        self.mismatches.append(dict(frame=self.index, key=key, actual=None))
        continue
      error = float(np.max(abs(np.asarray(row[key]) - np.asarray(expected[key]))))
      self.errors[key] = max(self.errors.get(key, 0), error)
      if error > 1e-4:
        self.mismatches.append(dict(frame=self.index, key=key, max_abs_error=error))


def replay(
  deployment,
  recording,
  output,
  mocap_config="deploy/configs/mocap_g1_live.yaml",
  simulation_config="deploy/configs/mujoco.yaml",
):
  recording, output = Path(recording), Path(output)
  rows = [
    json.loads(line) for line in (recording / "trace.jsonl").read_text().splitlines()
  ]
  summary = json.loads((recording / "summary.json").read_text())
  io = RecordedIO(rows)
  provider = SimpleNamespace(
    root=SimpleNamespace(get=io.root),
    ball=SimpleNamespace(get=io.ball),
    close=lambda: None,
  )
  adapter = SimulationIO(
    io,
    lambda *a: io,
    lambda cfg: provider,
    io.bind,
    io.record,
    dict(
      replay=True, source_recording=str(recording), actual_robot_commands_sent=False
    ),
  )
  run_real(
    deployment,
    summary["real_runtime_config"],
    mocap_config,
    enable_motors=True,
    motion_index=summary["home_motion"],
    motion_select=summary["motion_select"],
    motion_allow=summary["motion_allow"],
    duration=rows[-1]["time"] - rows[0]["time"] + 0.01,
    output=output,
    simulation_config=simulation_config,
    simulation_io=adapter,
  )
  result = dict(
    passed=io.checked == len(rows) and not io.mismatches,
    frames=len(rows),
    checked=io.checked,
    max_absolute_errors=io.errors,
    mismatches=io.mismatches[:30],
    mismatch_count=len(io.mismatches),
    actual_robot_commands_sent=False,
    actual_network_connections=False,
  )
  (output / "replay_comparison.json").write_text(json.dumps(result, indent=2))
  print(json.dumps(result), flush=True)
  return result


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--recording", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--config", type=Path, default=Path("deploy/configs/g1_tele_gmr.yaml"))
  a = p.parse_args()
  torch.set_num_threads(1)
  c = yaml.safe_load(a.config.read_text())
  d = load_deployment(**{k: v for k, v in c.items() if v is not None})
  r = replay(d, a.recording, a.output)
  if not r["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
