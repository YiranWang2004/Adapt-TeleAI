"""Per-TTS spatial coverage, with explicit denominators and missing errors."""

import csv
import json

import numpy as np

from deploy.benchmarks.racket_command.domain import grid_edges

ERRORS = (
  "position_error_m",
  "normal_error_degrees",
  "velocity_error_mps",
  "racket_ball_distance_at_deadline_m",
  "racket_ball_distance_at_contact_m",
)
RATES = (
  "accepted",
  "target_pass",
  "success",
  "ball_hit",
  "legal_return",
  "command_in_training_support",
)


def summarize(rows, scheduled, enabled_rates):
  cell = dict(
    scheduled=scheduled,
    evaluated=len(rows),
    deadline_observed=sum(bool(r.get("deadline_observed")) for r in rows),
    physical_feed_available=sum(bool(r.get("physical_feed_available")) for r in rows),
    rejected_start_window_count=sum(
      (r.get("decision") or {}).get("reason") == "arrival_outside_start_window"
      for r in rows
    ),
    rejected_position_count=sum(
      (r.get("decision") or {}).get("reason") == "a3_intercept_unreachable"
      for r in rows
    ),
    execution_failure_count=sum(bool(r.get("failure")) for r in rows),
  )
  for key in RATES:
    enabled = key in enabled_rates
    count = sum(bool(r.get(key)) for r in rows)
    n = len(rows)
    rate = count / n if n and enabled else None
    cell[key + "_count"] = count if enabled else None
    cell[key + "_rate"] = rate
    low = high = None
    if rate is not None:
      z = 1.959963984540054
      denominator = 1 + z * z / n
      center = (rate + z * z / (2 * n)) / denominator
      half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
      low, high = max(0.0, float(center - half)), min(1.0, float(center + half))
    cell[key + "_ci95_low"] = low
    cell[key + "_ci95_high"] = high
  for key in ERRORS:
    values = [r[key] for r in rows if r.get(key) is not None]
    cell[key + "_n"] = len(values)
    for stat, function in (
      ("mean", np.mean),
      ("median", np.median),
      ("p95", lambda v: np.percentile(v, 95)),
      ("std", np.std),
    ):
      cell[key + "_" + stat] = float(function(values)) if values else None
    cell[key + "_sem"] = (
      float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    )
  return cell


def write_tts_grids(output, config, suite, results):
  folder = output / "tts_grids"
  folder.mkdir(exist_ok=True)
  trials = suite["trials"]
  measured = {r["trial_id"]: r for r in results}
  times = sorted({t["tts"] for t in trials})
  ys, zs = grid_edges(config)
  enabled_rates = {"accepted", "target_pass", "success"} | {
    k for k in RATES if any(k in r for r in results)
  }
  grouped = {}
  for t in trials:
    grouped.setdefault((t["tts"], t["cell_y"], t["cell_z"]), []).append(t)
  cells = []
  for tts in times:
    for z in range(len(zs) - 1):
      for y in range(len(ys) - 1):
        scheduled = grouped.get((tts, y, z), [])
        rows = [measured[t["trial_id"]] for t in scheduled if t["trial_id"] in measured]
        cell = dict(
          tts=tts,
          cell_y=y,
          cell_z=z,
          y_min=float(ys[y]),
          y_max=float(ys[y + 1]),
          z_min=float(zs[z]),
          z_max=float(zs[z + 1]),
          scheduled=len(scheduled),
          evaluated=len(rows),
          deadline_observed=sum(bool(r.get("deadline_observed")) for r in rows),
        )
        cell.update(summarize(rows, len(scheduled), enabled_rates))
        cells.append(cell)
  (folder / "summary.json").write_text(json.dumps(cells, indent=2, allow_nan=False))
  if cells:
    with (folder / "summary.csv").open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
      writer.writeheader()
      writer.writerows(cells)
  if not results:
    return cells
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fields = [
    ("target_pass_rate", "Deadline target pass (%)", 100),
    ("success_rate", "Success incl. recovery (%)", 100),
    ("position_error_m_mean", "Racket-target at deadline (cm)", 100),
    ("normal_error_degrees_mean", "Normal error (deg)", 1),
    ("velocity_error_mps_mean", "Velocity error (m/s)", 1),
    ("accepted_rate", "Acceptance (%)", 100),
  ]
  if any("ball_hit" in r for r in results):
    fields.extend(
      [
        ("ball_hit_rate", "Physical contact (%)", 100),
        ("legal_return_rate", "Legal return (%)", 100),
        (
          "racket_ball_distance_at_deadline_m_mean",
          "Racket-ball at deadline (cm)",
          100,
        ),
        ("racket_ball_distance_at_contact_m_mean", "Racket-ball at contact (cm)", 100),
      ]
    )
  for index, tts in enumerate(times):
    fig, axes = plt.subplots(
      (len(fields) + 2) // 3,
      3,
      figsize=(16, 4 * ((len(fields) + 2) // 3)),
      constrained_layout=True,
      squeeze=False,
    )
    subset = [c for c in cells if c["tts"] == tts]
    for ax, (key, title, scale) in zip(axes.flat, fields, strict=False):
      data = np.full((len(zs) - 1, len(ys) - 1), np.nan)
      for c in subset:
        if c[key] is not None:
          data[c["cell_z"], c["cell_y"]] = c[key] * scale
      rate = key.endswith("_rate")
      # Same scale for all TTS panels of the same metric.
      maximum = (
        100
        if rate
        else max([c[key] * scale for c in cells if c[key] is not None] + [1e-6])
      )
      cmap = plt.get_cmap("RdYlGn" if rate else "magma_r").with_extremes(bad="#dddddd")
      im = ax.pcolormesh(
        ys, zs, data, cmap=cmap, vmin=0, vmax=maximum, edgecolors="white", linewidth=0.5
      )
      ax.set_aspect("equal", adjustable="box")
      for c in subset:
        value = c[key]
        label = "--" if value is None else f"{value * scale:.1f}"
        count = c["evaluated"] if rate else c[key.removesuffix("_mean") + "_n"]
        ax.text(
          (c["y_min"] + c["y_max"]) / 2,
          (c["z_min"] + c["z_max"]) / 2,
          f"{label}\nn={count}",
          ha="center",
          va="center",
          fontsize=6,
          bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
        )
      ax.set(xlabel="Y (m)", ylabel="Z (m)", title=title)
      fig.colorbar(im, ax=ax)
    for ax in list(axes.flat)[len(fields) :]:
      ax.set_visible(False)
    fig.suptitle(
      f"TTS = {tts:g} s | all evaluated targets in rate denominators | grey = no data"
    )
    fig.savefig(folder / f"tts_{index:02d}_{tts:g}s.png", dpi=160)
    plt.close(fig)
  return cells
