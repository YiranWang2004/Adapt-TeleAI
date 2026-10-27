"""Recorded-state MuJoCo viewer. No controller, ROS, DDS or physics rollout."""

import argparse
import bisect
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from deploy.backends.playback import HELP, PlaybackControl


def json_rows(path):
  """Permit a final incomplete write from a recording still being copied."""
  with path.open() as stream:
    for number, line in enumerate(stream, 1):
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        if stream.read().strip():
          raise ValueError(f"Malformed JSON in {path}:{number}") from None
        print(f"Ignoring incomplete final row: {path}:{number}")


def vector(value, size):
  try:
    result = np.asarray(value, dtype=float)
    return result if result.shape == (size,) and np.isfinite(result).all() else None
  except (TypeError, ValueError):
    return None


def received_command(row):
  """Home goals and unaccepted planner predictions are not received commands."""
  command = row.get("raw_racket_command")
  if (row.get("state") == "EXECUTE" or row.get("phase") == "SWING") and isinstance(
    command, dict
  ):
    return command
  return None


class Recording:
  def __init__(
    self,
    directory,
    *,
    rally_config=None,
    assume_hope_table_coordinates=False,
  ):
    self.directory = Path(directory).resolve()
    self.summary = json.loads((self.directory / "summary.json").read_text())
    self.source_world = self.summary.get("control_world_frame")
    if self.source_world != "hope_table":
      if self.source_world != "world" or not assume_hope_table_coordinates:
        raise ValueError(
          "This log is not labeled hope_table. For a verified identity-frame "
          "legacy log, pass --assume-hope-table-coordinates explicitly."
        )
    fallback = yaml.safe_load(Path(rally_config).read_text()) if rally_config else {}
    self.ball_config = self.summary.get("rally_runtime_config", {}).get("ball")
    if self.ball_config is None:
      self.ball_config = fallback.get("ball")
    if self.ball_config is None:
      raise ValueError(
        "No table/ball geometry in summary; provide --rally-config with a ball section"
      )
    self.offset = vector(self.summary.get("simulation_world_translation"), 3)
    if self.offset is None:
      self.offset = vector(
        fallback.get("world", {}).get("mujoco_to_world_translation"), 3
      )
    if self.offset is None:
      raise ValueError(
        "No MuJoCo/world translation in summary; provide --rally-config with "
        "world.mujoco_to_world_translation"
      )
    self.names = self.summary["robot_command_contract"]["joint_names"]
    if len(set(self.names)) != 29:
      raise ValueError("Expected a recorded contract with 29 unique G1 joints")
    self.rows, self.times, self.events = [], [], []
    self.skipped = 0
    keys = (
      "time",
      "q",
      "root_pos",
      "root_quat",
      "state",
      "phase",
      "motion_index",
      "goal_position",
      "goal_normal",
      "goal_velocity",
      "raw_racket_command",
      "time_to_strike",
      "racket_pos",
      "ball_position",
      "ball_age",
      "root_age",
    )
    for row in json_rows(self.directory / "trace.jsonl"):
      stamp = row.get("time")
      if (
        not isinstance(stamp, (int, float))
        or not math.isfinite(stamp)
        or any(
          vector(row.get(k), n) is None
          for k, n in (("q", 29), ("root_pos", 3), ("root_quat", 4))
        )
        or np.linalg.norm(row["root_quat"]) < 1e-8
      ):
        self.skipped += 1
        continue
      if self.times and stamp < self.times[-1]:
        raise ValueError("Trace time moves backwards; select one recording folder")
      if not self.rows:
        self.origin = stamp
      t = stamp - self.origin
      saved = {key: row.get(key) for key in keys}
      if not self.rows or (saved["state"], saved["phase"], saved["motion_index"]) != (
        self.rows[-1]["state"],
        self.rows[-1]["phase"],
        self.rows[-1]["motion_index"],
      ):
        self.events.append(t)
      self.times.append(t)
      self.rows.append(saved)
    if not self.rows:
      raise ValueError("No usable measured robot frames in trace.jsonl")
    self.duration = self.times[-1]
    self.ball_times, self.ball_positions = [], []
    packets = self.directory / "mocap_packets.jsonl"
    if packets.is_file():
      for row in json_rows(packets):
        packet = row.get("packet", {})
        p = vector(packet.get("position"), 3)
        arrival = row.get("arrival_time")
        if packet.get("kind") != "ball" or p is None or arrival is None:
          continue
        if packet.get("frame") != self.source_world:
          raise ValueError("Ball packet frame differs from recorded robot world")
        self.ball_times.append(float(arrival) - self.origin)
        self.ball_positions.append(p)
    if self.ball_times:
      order = np.argsort(self.ball_times, kind="stable")
      self.ball_times = np.asarray(self.ball_times)[order].tolist()
      self.ball_positions = np.asarray(self.ball_positions)[order]

  def index(self, seconds):
    return max(0, bisect.bisect_right(self.times, seconds) - 1)

  def ball(self, seconds, max_age=0.1):
    i = bisect.bisect_right(self.ball_times, seconds) - 1
    if i < 0 or seconds - self.ball_times[i] > max_age:
      return None
    return self.ball_positions[i]

  def describe(self):
    return dict(
      recording=str(self.directory),
      seconds=self.duration,
      measured_robot_frames=len(self.rows),
      skipped_frames=self.skipped,
      raw_ball_samples=len(self.ball_times),
      phase_events=self.events,
      source_world=self.source_world,
      display_world="hope_table",
      simulation_world_translation=self.offset.tolist(),
      ball_clock="local packet arrival_time; causal sample-and-hold",
      replay="measured q/root pose; no policy or physics integration",
    )


class Timeline:
  def __init__(self, recording, start=0.0, rate=1.0, paused=False):
    self.recording = recording
    self.position = float(np.clip(start, 0, recording.duration))
    self.speed = PlaybackControl(rate)
    self.paused = paused
    self.lock = threading.RLock()

  def advance(self, elapsed):
    with self.lock:
      if not self.paused:
        self.position = min(
          self.recording.duration, self.position + elapsed * self.speed.rate
        )
        if self.position == self.recording.duration:
          self.paused = True
      return self.position

  def key(self, key):
    with self.lock:
      if self.speed.key(key):
        return
      if key == 32:
        self.paused = not self.paused
      elif key in (ord("R"), ord("r")):
        self.position = 0.0
        self.paused = False
      elif key in (263, 262):  # left/right: seek one second
        self.position += -1 if key == 263 else 1
      elif key in (268, 269):  # home/end
        self.position = 0 if key == 268 else self.recording.duration
      elif key in (ord(","), ord(".")):
        i = self.recording.index(self.position)
        i = int(
          np.clip(i + (-1 if key == ord(",") else 1), 0, len(self.recording.times) - 1)
        )
        self.position = self.recording.times[i]
        self.paused = True
      elif key in (ord("N"), ord("P")):
        events = self.recording.events
        i = (
          bisect.bisect_right(events, self.position + 1e-6)
          if key == ord("N")
          else bisect.bisect_left(events, self.position - 1e-6) - 1
        )
        self.position = events[int(np.clip(i, 0, len(events) - 1))]
      self.position = float(np.clip(self.position, 0, self.recording.duration))


class LogScene:
  def __init__(self, recording):
    import mujoco

    from deploy.hope_g1.scene import table_cfg
    from deploy.hope_g1.world import mujoco_ball_config
    from mjlab.asset_zoo.robots.unitree_g1_pingpong.g1_constants import (
      get_g1_pingpong_robot_cfg,
    )
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    ball = recording.ball_config
    self.recording = recording
    self.model = Scene(
      SceneCfg(
        entities={
          "robot": get_g1_pingpong_robot_cfg(),
          "table": table_cfg(mujoco_ball_config(ball, recording.offset)),
        },
        terrain=TerrainEntityCfg(terrain_type="plane"),
      ),
      device="cpu",
    ).compile()
    self.data = mujoco.MjData(self.model)
    self.addresses = [
      self.model.jnt_qposadr[self.model.joint("robot/" + n).id] for n in recording.names
    ]
    free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
      raise ValueError("Expected one robot root free joint")
    self.root = self.model.jnt_qposadr[free[0]]
    self.camera = mujoco.MjvCamera()
    self.camera.lookat[:] = [1.2, 0, 0.9]
    self.camera.distance, self.camera.azimuth, self.camera.elevation = 4.5, 135, -22

  def update(self, seconds):
    import mujoco

    row = self.recording.rows[self.recording.index(seconds)]
    self.data.qpos[self.addresses] = row["q"]
    self.data.qpos[self.root : self.root + 3] = (
      np.asarray(row["root_pos"]) - self.recording.offset
    )
    quat = np.asarray(row["root_quat"])
    self.data.qpos[self.root + 3 : self.root + 7] = quat / np.linalg.norm(quat)
    self.data.time = seconds
    mujoco.mj_forward(self.model, self.data)
    return row

  def overlays(self, scene, seconds, row, *, clear=True):
    import mujoco

    if clear:
      scene.ngeom = 0

    def sphere(p, radius, color):
      if p is None or scene.ngeom >= scene.maxgeom:
        return
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(p) - self.recording.offset,
        np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
      )
      scene.ngeom += 1

    def arrow(p, direction, color, scale):
      v = vector(direction, 3)
      if v is None or scene.ngeom >= scene.maxgeom:
        return
      geom = scene.geoms[scene.ngeom]
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.ones(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
      )
      a = np.asarray(p) - self.recording.offset
      mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, a, a + v * scale)
      scene.ngeom += 1

    sphere(self.recording.ball(seconds), 0.02, [1, 0.65, 0.05, 1])
    # Trail includes only observations already delivered; never bridges missing data.
    start = bisect.bisect_left(self.recording.ball_times, seconds - 0.4)
    end = bisect.bisect_right(self.recording.ball_times, seconds)
    for i in range(start, end, 3):
      sphere(self.recording.ball_positions[i], 0.006, [1, 0.65, 0.05, 0.35])
    goal = (
      vector(row.get("goal_position"), 3) if received_command(row) is not None else None
    )
    if goal is not None:
      sphere(goal, 0.025, [0.2, 1, 0.3, 0.65])
      arrow(goal, row.get("goal_normal"), [0.2, 1, 0.3, 1], 0.2)
      arrow(goal, row.get("goal_velocity"), [0.2, 0.7, 1, 1], 0.06)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--recording",
    "--log-dir",
    required=True,
    type=Path,
    help="Folder containing summary.json, trace.jsonl and optional mocap_packets.jsonl",
  )
  parser.add_argument("--rate", type=float, default=1.0)
  parser.add_argument(
    "--start", type=float, default=0.0, help="Seconds from first usable robot frame"
  )
  parser.add_argument("--paused", action="store_true")
  parser.add_argument(
    "--rally-config",
    type=Path,
    help="Fallback court geometry/translation for legacy logs without these metadata",
  )
  parser.add_argument(
    "--assume-hope-table-coordinates",
    action="store_true",
    help=(
      "Explicitly treat a world-labeled log as HOPE tabletop coordinates "
      "without changing XYZ; only use when its source transform is identity"
    ),
  )
  parser.add_argument(
    "--inspect", action="store_true", help="Inspect recording without opening a window"
  )
  parser.add_argument(
    "--video", type=Path, help="Export an MP4 instead of opening a window"
  )
  parser.add_argument("--end", type=float, help="Video end time in recording seconds")
  parser.add_argument("--fps", type=int, default=30)
  args = parser.parse_args()
  if not math.isfinite(args.start) or args.start < 0 or args.fps <= 0:
    parser.error("start must be finite/nonnegative and fps positive")
  recording = Recording(
    args.recording,
    rally_config=args.rally_config,
    assume_hope_table_coordinates=args.assume_hope_table_coordinates,
  )
  timeline = Timeline(recording, args.start, args.rate, args.paused)
  print(json.dumps(recording.describe(), indent=2))
  if args.inspect:
    return
  import mujoco

  scene = LogScene(recording)
  if args.video:
    import imageio.v2 as imageio

    end = recording.duration if args.end is None else min(args.end, recording.duration)
    if not math.isfinite(end) or end <= args.start:
      parser.error("Video end must be greater than start")
    args.video.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(scene.model, height=720, width=1280) as renderer:
      with imageio.get_writer(str(args.video), fps=args.fps) as writer:
        for t in np.arange(args.start, end, args.rate / args.fps):
          row = scene.update(t)
          renderer.update_scene(scene.data, camera=scene.camera)
          scene.overlays(renderer.scene, t, row, clear=False)
          writer.append_data(renderer.render())
    print(f"Saved {args.video}")
    return
  import mujoco.viewer

  print(
    f"{HELP}; R restart; SPACE pause/resume; arrows seek 1s; ,/. step; N/P phase; HOME/END"
  )
  with mujoco.viewer.launch_passive(
    scene.model, scene.data, key_callback=timeline.key
  ) as viewer:
    viewer.cam.lookat[:] = scene.camera.lookat
    viewer.cam.distance = scene.camera.distance
    viewer.cam.azimuth = scene.camera.azimuth
    viewer.cam.elevation = scene.camera.elevation
    previous = time.monotonic()
    while viewer.is_running():
      now = time.monotonic()
      t = timeline.advance(now - previous)
      previous = now
      with viewer.lock():
        row = scene.update(t)
        scene.overlays(viewer.user_scn, t, row)
      ball = recording.ball(t)
      ball_text = (
        "missing/stale" if ball is None else np.array2string(ball, precision=3)
      )
      command_tts = (
        row.get("time_to_strike") if received_command(row) is not None else "--"
      )
      viewer.set_texts(
        (
          None,
          None,
          "RECORDED REAL STATE\nTime / speed\nState / phase / motion\nBall (table world)\nCommand TTS (s)\nRobot frame age\nLegend\nControls",
          f"{recording.directory.name}\n{t:.2f}/{recording.duration:.2f}s  {timeline.speed.rate:g}x {'PAUSED' if timeline.paused else ''}\n"
          f"{row['state']} / {row['phase']} / {row['motion_index']}\n{ball_text}\n{command_tts}\n{t - recording.times[recording.index(t)]:.3f}s\n"
          "orange: measured ball; green: command; blue: velocity\nR restart; SPACE pause/resume; [ ] speed; arrows seek 1s; , . step; N/P phase",
        )
      )
      viewer.sync()
      time.sleep(1 / 120)


if __name__ == "__main__":
  main()
