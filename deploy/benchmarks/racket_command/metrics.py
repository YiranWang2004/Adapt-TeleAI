"""Evaluate at the requested deadline; never reward another crossing time."""

import numpy as np


def evaluate(trial, samples, accepted, recovered, failure, tolerances):
  times = np.asarray([s["time"] for s in samples])
  deadline = trial.tts
  covered = bool(len(times) >= 2 and times[0] <= deadline <= times[-1])
  result = dict(
    trial_id=trial.trial_id,
    cell_y=trial.cell_y,
    cell_z=trial.cell_z,
    accepted=accepted,
    recovered=recovered,
    failure=failure,
    deadline_observed=covered,
    success=False,
  )
  result.update(
    {
      k: getattr(trial, k)
      for k in ("point_id", "direction_id", "speed_id", "tts_id")
      if getattr(trial, k, None) is not None
    }
  )
  if not covered:
    return result

  def at(key):
    values = np.asarray([s[key] for s in samples])
    return np.array([np.interp(deadline, times, values[:, i]) for i in range(3)])

  position, normal, velocity = at("position"), at("normal"), at("velocity")
  if all("ball_position" in s for s in samples):
    ball_position = at("ball_position")
    delta = position - ball_position
    result.update(
      ball_position_at_deadline=ball_position.tolist(),
      racket_ball_delta_at_deadline_m=delta.tolist(),
      racket_ball_distance_at_deadline_m=float(np.linalg.norm(delta)),
    )
  normal /= max(np.linalg.norm(normal), 1e-12)
  # A circular disk has equivalent faces; command_goal applies hit-type sign.
  dot = abs(float(normal @ np.asarray(trial.normal)))
  position_error = float(np.linalg.norm(position - trial.position))
  normal_error = float(np.degrees(np.arccos(np.clip(dot, 0, 1))))
  velocity_error = float(np.linalg.norm(velocity - trial.velocity))
  near = np.abs(times - deadline) <= 0.25
  distances = np.linalg.norm(
    np.asarray([s["position"] for s in samples]) - trial.position, axis=1
  )
  ids = np.flatnonzero(near)
  closest = ids[np.argmin(distances[ids])] if len(ids) else np.argmin(distances)
  timing_error = float(times[closest] - deadline)
  pos_ok = position_error <= tolerances["position_m"]
  normal_ok = normal_error <= tolerances["normal_degrees"]
  velocity_ok = velocity_error <= tolerances["velocity_mps"]
  timing_ok = abs(timing_error) <= tolerances["timing_seconds"]
  result.update(
    position_error_m=position_error,
    normal_error_degrees=normal_error,
    velocity_error_mps=velocity_error,
    timing_error_seconds=timing_error,
    actual_position=position.tolist(),
    actual_normal=normal.tolist(),
    actual_velocity=velocity.tolist(),
    position_pass=pos_ok,
    normal_pass=normal_ok,
    velocity_pass=velocity_ok,
    timing_pass=timing_ok,
    target_pass=bool(accepted and pos_ok and normal_ok and velocity_ok),
    success=bool(
      accepted
      and pos_ok
      and normal_ok
      and velocity_ok
      and recovered
      and failure is None
    ),
  )
  return result
