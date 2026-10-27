"""Stage-two metrics keep planning, tracking and physical returns separate."""

import html
import json

import numpy as np


def write_report(output, suite, rows, metadata, status):
  n = len(rows)

  def count(key):
    return sum(bool(r.get(key)) for r in rows)

  summary = dict(
    evaluated=n,
    scheduled=len(suite["serves"]),
    planned_and_accepted=count("accepted"),
    contacts=count("ball_hit"),
    legal_returns=count("legal_return"),
    recovered=count("recovered"),
    execution_failures=sum(r["failure"] is not None for r in rows),
    command_target_passes=sum(r["tracking"].get("target_pass", False) for r in rows),
    planner_return_target=metadata.get("planner_return_target"),
  )
  summary.update(
    contact_rate=count("ball_hit") / n if n else None,
    legal_return_rate=count("legal_return") / n if n else None,
  )
  commands = [r for r in rows if r["accepted"]]
  if commands:
    summary.update(
      planner_position_error_median_m=float(
        np.median([r["planner_position_error_m"] for r in commands])
      ),
      planner_abs_time_error_median_s=float(
        np.median([abs(r["planner_time_error_s"]) for r in commands])
      ),
      planned_net_clear=sum(r["planned_net_clear"] for r in commands),
    )
  report = dict(
    task="planner_receive",
    status=status,
    suite_sha256=suite["sha256"],
    config=suite["config"],
    physics=suite["physics"],
    metadata=metadata,
    summary=summary,
    results=rows,
  )
  (output / "report.json").write_text(json.dumps(report, indent=2))
  if not rows:
    return report
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  config = suite["config"]
  fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
  fields = [
    ("accepted", "Planner + selector acceptance", True),
    ("ball_hit", "Physical contact", True),
    ("legal_return", "Legal return", True),
    ("planner_position_error_m", "Planner position error (cm)", False),
    ("planner_time_error_s", "Absolute planner time error (ms)", False),
    (
      "tracking_position_error",
      "Racket position error at command deadline (cm)",
      False,
    ),
  ]
  ny, nz = config["grid"]
  for ax, (key, title, rate) in zip(axes.flat, fields, strict=True):
    grid = np.full((nz, ny), np.nan)
    for y in range(ny):
      for z in range(nz):
        group = [r for r in rows if (r["cell_y"], r["cell_z"]) == (y, z)]
        values = (
          [
            r["tracking"]["position_error_m"] * 100
            for r in group
            if "position_error_m" in r["tracking"]
          ]
          if key == "tracking_position_error"
          else [
            abs(r[key])
            * (
              100
              if key == "planner_position_error_m"
              else 1000
              if key == "planner_time_error_s"
              else 100
            )
            for r in group
            if key in r
          ]
        )
        if values:
          grid[z, y] = np.mean(values)
    im = ax.imshow(
      grid,
      origin="lower",
      aspect="auto",
      extent=[*config["y_range"], *config["z_range"]],
      cmap="RdYlGn" if rate else "magma_r",
      vmin=0,
      vmax=100 if rate else None,
    )
    ax.set(
      title=title + (" (%)" if rate else ""),
      xlabel="Incoming crossing Y (m)",
      ylabel="Incoming crossing Z (m)",
    )
    fig.colorbar(im, ax=ax)
  fig.suptitle(
    f"Independent central random serves: {n}/{len(suite['serves'])}; contacts {summary['contacts']}; legal returns {summary['legal_returns']}"
  )
  fig.savefig(output / "capability.png", dpi=150)
  plt.close(fig)
  details = []
  for r in rows:
    command = r["command"]
    details.append(
      f"<tr><td>{r['trial_id']}</td><td>{r['cell_y']},{r['cell_z']}</td><td>{r['motion_index']}</td><td>{r['accepted']}</td><td>{r['ball_hit']}</td><td>{r['legal_return']}</td><td>{command['tts_at_issue']:.3f}</td><td>{r['planner_position_error_m'] * 100:.2f}</td><td>{r['planner_time_error_s'] * 1000:+.1f}</td><td>{r['tracking'].get('position_error_m', float('nan')) * 100:.2f}</td></tr>"
      if command
      else f"<tr><td>{r['trial_id']}</td><td>{r['cell_y']},{r['cell_z']}</td><td colspan='8'>No accepted command</td></tr>"
    )
  page = "<!doctype html><meta charset='utf-8'><title>第二阶段：中心区域随机接球</title><style>body{font:16px system-ui;margin:24px;background:#f6f7fa;color:#192536}img{max-width:100%}table{border-collapse:collapse;background:white}th,td{padding:8px;border:1px solid #ccc}pre{white-space:pre-wrap}</style><h1>独立随机来球 → 估计 → A3 planner → 选择器 / policy</h1><p>来球只按球轨迹几何筛选，不读取motion、racket command或策略成绩。目标与时刻来自运行时位置观测和planner。热力图按真实来球在击球平面的交点分组。</p>"
  page += (
    "<pre>"
    + html.escape(json.dumps(summary, indent=2))
    + "</pre><img src='capability.png'><table><tr><th>球号</th><th>格子</th><th>motion</th><th>接受</th><th>碰拍</th><th>有效回球</th><th>下发TTS/s</th><th>规划位置误差/cm</th><th>规划时间误差/ms</th><th>实际拍心误差/cm</th></tr>"
    + "".join(details)
    + "</table><p><a href='report.json'>完整数据</a> · <a href='suite.json'>冻结发球序列</a></p>"
  )
  frames = (
    sorted((output / "frames").glob("*.png")) if (output / "frames").exists() else []
  )
  for frame in frames:
    page += (
      f"<p>{html.escape(frame.name)}</p><img src='frames/{html.escape(frame.name)}'>"
    )
  (output / "report.html").write_text(page)
  return report
