"""Replay recorded real ball positions through the A30808 planner and render GIFs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

from deploy.hope_g1.hope_planner_adapter import HopePlannerAdapter


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECORDINGS = [
  ROOT / "deploy/recordings/hope_g1/real_20261022_121714",
  ROOT / "deploy/recordings/hope_g1/real_20261022_121834",
]
DEFAULT_OUTPUT = ROOT / "deploy/recordings/planner/a30808_real_trajectory_replay"
PLANE_X = -0.25


def read_packets(recording):
  rows = []
  with (recording / "mocap_packets.jsonl").open() as stream:
    for line in stream:
      packet = json.loads(line)["packet"]
      if packet.get("kind") != "ball":
        continue
      position = np.asarray(packet["position"], dtype=float)
      stamp = float(packet["stamp"])
      if position.shape == (3,) and np.isfinite(position).all() and np.isfinite(stamp):
        rows.append((stamp, position))
  rows.sort(key=lambda item: item[0])
  if not rows:
    raise ValueError(f"No ball packets in {recording}")
  return rows


def measured_crossings(rows):
  crossings = []
  for (t0, p0), (t1, p1) in zip(rows[:-1], rows[1:]):
    dt = t1 - t0
    if 0 < dt <= 0.05 and p0[0] > PLANE_X >= p1[0]:
      ratio = (p0[0] - PLANE_X) / (p0[0] - p1[0])
      point = p0 + ratio * (p1 - p0)
      if -1.7 <= point[1] <= 0.2 and 0.0 <= point[2] <= 1.2:
        crossings.append((t0 + ratio * dt, point))
  return crossings


def replay(recording, rows, event_stamps=()):
  adapter = HopePlannerAdapter(hit_plane_x=PLANE_X)
  planner = adapter.planner
  frames, command_history = [], []
  last_frame_stamp = -np.inf
  next_event = 0
  event_stamps = sorted(event_stamps)
  latest_command = None
  latest_prediction = None
  for index, (stamp, position) in enumerate(rows):
    event_due = next_event < len(event_stamps) and stamp >= event_stamps[next_event]
    capture = stamp - last_frame_stamp >= 0.10 or index == len(rows) - 1 or event_due
    command = planner.update(stamp, position, collect_prediction_samples=capture)
    if command is not None:
      latest_command = {
        "stamp": stamp,
        "position": command.p_intercept.copy(),
        "velocity": command.v_racket.copy(),
        "normal": command.n_racket.copy(),
        "t_strike": float(command.t_strike),
        "valid": bool(command.valid),
      }
      command_history.append(latest_command)
    if not capture:
      continue
    last_frame_stamp = stamp
    while next_event < len(event_stamps) and event_stamps[next_event] <= stamp:
      next_event += 1
    if latest_command is not None and latest_command["t_strike"] <= stamp:
      latest_command = None
    prediction = planner.latest_prediction_samples
    latest_prediction = prediction if prediction is not None and len(prediction) else None
    start = max(0, index - 150)  # roughly the preceding 1.5 s at mocap rate
    frames.append({
      "index": index,
      "stamp": stamp,
      "position": position.copy(),
      "history": np.stack([row[1] for row in rows[start:index + 1]]),
      "command": latest_command,
      "prediction": latest_prediction,
      "fit_position": planner.latest_estimate,
      "fit_velocity": planner.latest_velocity,
    })
  summary = {
    "recording": recording.name,
    "ball_samples": len(rows),
    "captured_frames": len(frames),
    "planner_commands": len(command_history),
    "measured_plane_crossings": len(measured_crossings(rows)),
    "hit_plane_x_m": PLANE_X,
    "first_stamp": rows[0][0],
    "last_stamp": rows[-1][0],
  }
  return frames, command_history, summary


def render_event_summary(recording, frames, old_commands, crossings, path):
  if not old_commands:
    return
  fig, axes = plt.subplots(len(old_commands), 2, figsize=(13, 4.2 * len(old_commands)))
  axes = np.asarray(axes).reshape(len(old_commands), 2)
  for row, old in enumerate(old_commands):
    frame = min(frames, key=lambda item: abs(item["stamp"] - old["stamp"]))
    history = frame["history"]
    command = frame["command"]
    crossing = next((item for item in crossings if item[0] >= old["stamp"]), None)
    xz, xy = axes[row]
    xz.plot([0, 2.74], [0, 0], color="#735b3e", lw=5, alpha=0.4)
    xz.plot([1.37, 1.37], [0, 0.1525], color="#333333", lw=3)
    xz.axvline(PLANE_X, color="#d62728", ls="--", label="G1 hit plane")
    xz.plot(history[:, 0], history[:, 2], color="#777777", lw=1.4, label="Measured trajectory")
    xy.add_patch(Rectangle((0, -1.525), 2.74, 1.525, fill=False, color="#735b3e", lw=2))
    xy.axvline(PLANE_X, color="#d62728", ls="--", label="G1 hit plane")
    xy.plot(history[:, 0], history[:, 1], color="#777777", lw=1.4, label="Measured trajectory")
    if frame["prediction"] is not None:
      prediction = frame["prediction"]
      xz.plot(prediction[:, 1], prediction[:, 3], color="#137c8b", lw=2, label="A30808 predicted flight")
      xy.plot(prediction[:, 1], prediction[:, 2], color="#137c8b", lw=2, label="A30808 predicted flight")
    xz.scatter(old["position"][0], old["position"][2], marker="x", s=90, color="#f08a24", label="Previous deploy")
    xy.scatter(old["position"][0], old["position"][1], marker="x", s=90, color="#f08a24", label="Previous deploy")
    command_text = "No A30808 command"
    if command is not None:
      xz.scatter(command["position"][0], command["position"][2], marker="*", s=160, color="#13a36f", label="A30808 strike point")
      xy.scatter(command["position"][0], command["position"][1], marker="*", s=160, color="#13a36f", label="A30808 strike point")
      tts = command["t_strike"] - frame["stamp"]
      command_text = f"A30808 TTS={tts:.3f}s"
    actual_text = "No later measured plane crossing"
    if crossing is not None:
      xz.scatter(crossing[1][0], crossing[1][2], marker="X", s=100, color="#d62728", label="Measured crossing")
      xy.scatter(crossing[1][0], crossing[1][1], marker="X", s=100, color="#d62728", label="Measured crossing")
      actual_text = f"Measured @ +{crossing[0] - old['stamp']:.3f}s, Z={crossing[1][2]:.3f}m"
    for ax in (xz, xy):
      ax.set_xlim(-0.55, 2.85)
      ax.grid(True, alpha=0.25)
      ax.legend(fontsize=8, loc="best")
    xz.set_ylim(-0.08, 1.6); xz.set_xlabel("X (m)"); xz.set_ylabel("Z (m)")
    xy.set_ylim(-1.7, 0.15); xy.set_xlabel("X (m)"); xy.set_ylabel("Y (m)")
    t_rel = frame["stamp"] - frames[0]["stamp"]
    xz.set_title(
      f"Motion {old['motion_index']} | t={t_rel:.2f}s\n{command_text}; {actual_text}",
      fontsize=9,
    )
    xy.set_title("X-Y")
  fig.suptitle(recording.name, fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def render_gif(recording, rows, frames, old_commands, output):
  crossings = measured_crossings(rows)
  fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
  ax_xz, ax_xy = axes
  ax_xz.set_title("Side view: X-Z")
  ax_xz.set_xlabel("X in hope_table (m)")
  ax_xz.set_ylabel("Z (m)")
  ax_xz.set_xlim(-0.55, 2.85)
  ax_xz.set_ylim(-0.08, 1.6)
  ax_xz.plot([0, 2.74], [0, 0], color="#735b3e", lw=6, alpha=0.45, label="Table surface")
  ax_xz.plot([1.37, 1.37], [0, 0.1525], color="#333333", lw=4, label="Net")
  ax_xz.axvline(PLANE_X, color="#d62728", ls="--", lw=1.5, label="G1 hit plane")
  ax_xy.set_title("Top view: X-Y")
  ax_xy.set_xlabel("X in hope_table (m)")
  ax_xy.set_ylabel("Y (m)")
  ax_xy.set_xlim(-0.55, 2.85)
  ax_xy.set_ylim(-1.7, 0.15)
  ax_xy.add_patch(Rectangle((0, -1.525), 2.74, 1.525, fill=False, color="#735b3e", lw=2))
  ax_xy.axvline(1.37, color="#333333", lw=1.5, label="Net")
  ax_xy.axvline(PLANE_X, color="#d62728", ls="--", lw=1.5, label="G1 hit plane")

  measured_xz, = ax_xz.plot([], [], color="#777777", lw=1.3, alpha=0.7, label="Measured ball")
  predicted_xz, = ax_xz.plot([], [], color="#137c8b", lw=2, label="A30808 predicted flight")
  ball_xz = ax_xz.scatter([], [], s=50, color="#2878b5", zorder=5)
  intercept_xz = ax_xz.scatter([], [], marker="*", s=190, color="#13a36f", zorder=6, label="Planner strike point")
  crossing_xz = ax_xz.scatter([], [], marker="X", s=100, color="#d62728", zorder=7, label="Measured plane crossing")
  oldcross_xz, = ax_xz.plot([], [], "x", color="#f08a24", ms=10, mew=2, label="Previous deploy command")
  measured_xy, = ax_xy.plot([], [], color="#777777", lw=1.3, alpha=0.7, label="Measured ball")
  predicted_xy, = ax_xy.plot([], [], color="#137c8b", lw=2, label="A30808 predicted flight")
  ball_xy = ax_xy.scatter([], [], s=50, color="#2878b5", zorder=5)
  intercept_xy = ax_xy.scatter([], [], marker="*", s=190, color="#13a36f", zorder=6, label="Planner strike point")
  crossing_xy = ax_xy.scatter([], [], marker="X", s=100, color="#d62728", zorder=7, label="Measured plane crossing")
  oldcross_xy, = ax_xy.plot([], [], "x", color="#f08a24", ms=10, mew=2, label="Previous deploy command")
  normal_arrow = None
  velocity_arrow = None
  crossing_shown = []

  for ax in axes:
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

  def set_scatter(scatter, point):
    if point is None:
      scatter.set_offsets(np.empty((0, 2)))
    else:
      scatter.set_offsets(np.asarray(point, dtype=float).reshape(1, 2))

  def draw(frame_index):
    nonlocal normal_arrow, velocity_arrow
    frame = frames[frame_index]
    history = frame["history"]
    measured_xz.set_data(history[:, 0], history[:, 2])
    measured_xy.set_data(history[:, 0], history[:, 1])
    set_scatter(ball_xz, frame["position"][[0, 2]])
    set_scatter(ball_xy, frame["position"][[0, 1]])

    pred = frame["prediction"]
    if pred is not None:
      predicted_xz.set_data(pred[:, 1], pred[:, 3])
      predicted_xy.set_data(pred[:, 1], pred[:, 2])
    else:
      predicted_xz.set_data([], [])
      predicted_xy.set_data([], [])

    cmd = frame["command"]
    if cmd is not None:
      set_scatter(intercept_xz, cmd["position"][[0, 2]])
      set_scatter(intercept_xy, cmd["position"][[0, 1]])
      if normal_arrow is not None:
        normal_arrow.remove()
      if velocity_arrow is not None:
        velocity_arrow.remove()
      # Scaled direction glyphs; the labels retain the physical m/s values.
      normal_arrow = ax_xy.quiver(
        cmd["position"][0], cmd["position"][1],
        cmd["normal"][0] * 0.08, cmd["normal"][1] * 0.08,
        color="#7551a8", angles="xy", scale_units="xy", scale=1,
        width=0.007, zorder=8,
      )
      velocity_arrow = ax_xy.quiver(
        cmd["position"][0], cmd["position"][1],
        cmd["velocity"][0] * 0.025, cmd["velocity"][1] * 0.025,
        color="#f08a24", angles="xy", scale_units="xy", scale=1,
        width=0.006, zorder=8,
      )
      tts = cmd["t_strike"] - frame["stamp"]
      cmd_text = (
        f"A30808 command: TTS={tts:.3f}s, "
        f"racket v=({cmd['velocity'][0]:.2f}, {cmd['velocity'][1]:.2f}, {cmd['velocity'][2]:.2f}) m/s"
      )
    else:
      set_scatter(intercept_xz, None)
      set_scatter(intercept_xy, None)
      if normal_arrow is not None:
        normal_arrow.remove(); normal_arrow = None
      if velocity_arrow is not None:
        velocity_arrow.remove(); velocity_arrow = None
      cmd_text = "A30808 command: none"

    available = [crossing for crossing in crossings if crossing[0] <= frame["stamp"]]
    latest_crossing = available[-1][1] if available else None
    if latest_crossing is None:
      set_scatter(crossing_xz, None)
      set_scatter(crossing_xy, None)
    else:
      set_scatter(crossing_xz, latest_crossing[[0, 2]])
      set_scatter(crossing_xy, latest_crossing[[0, 1]])

    old = [c for c in old_commands if c["stamp"] <= frame["stamp"]]
    if old:
      previous = old[-1]["position"]
      oldcross_xz.set_data([previous[0]], [previous[2]])
      oldcross_xy.set_data([previous[0]], [previous[1]])
    else:
      oldcross_xz.set_data([], [])
      oldcross_xy.set_data([], [])
    title_stamp = frame["stamp"] - rows[0][0]
    fig.suptitle(
      f"{recording.name} | t={title_stamp:.2f}s | ball=({frame['position'][0]:.2f}, "
      f"{frame['position'][1]:.2f}, {frame['position'][2]:.2f}) m\n{cmd_text}",
      fontsize=11,
    )
    return (
      measured_xz, predicted_xz, ball_xz, intercept_xz, crossing_xz, oldcross_xz,
      measured_xy, predicted_xy, ball_xy, intercept_xy, crossing_xy, oldcross_xy,
    )

  animation = FuncAnimation(fig, draw, frames=len(frames), interval=100, blit=False)
  path = output / f"{recording.name}_A30808.gif"
  animation.save(path, writer=PillowWriter(fps=10))
  draw(len(frames) - 1)
  fig.savefig(output / f"{recording.name}_final.png", dpi=150, bbox_inches="tight")
  plt.close(fig)
  return path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--recording", type=Path, action="append")
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  recordings = args.recording or DEFAULT_RECORDINGS
  args.output.mkdir(parents=True, exist_ok=True)
  summaries = []
  for recording in recordings:
    rows = read_packets(recording)
    with (recording / "summary.json").open() as stream:
      old_events = json.load(stream).get("events", [])
    old_commands = []
    for event in old_events:
      if event.get("event") != "motion_selected":
        continue
      command = event.get("raw_command") or {}
      if command.get("position") is not None and command.get("estimate_stamp") is not None:
        old_commands.append({
          "stamp": float(command["estimate_stamp"]),
          "position": np.asarray(command["position"], dtype=float),
          "motion_index": event.get("motion_index"),
        })
    old_commands.sort(key=lambda item: item["stamp"])
    frames, commands, summary = replay(
      recording, rows, [item["stamp"] for item in old_commands]
    )
    gif_path = render_gif(recording, rows, frames, old_commands, args.output)
    event_plot = args.output / f"{recording.name}_planner_events.png"
    render_event_summary(
      recording, frames, old_commands, measured_crossings(rows), event_plot
    )
    summary["gif"] = str(gif_path)
    summary["event_plot"] = str(event_plot)
    summaries.append(summary)
    print(json.dumps(summary, indent=2))
  (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
  main()
