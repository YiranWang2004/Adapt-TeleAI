"""Live MuJoCo command geometry and standalone capability reports."""

import json

import mujoco
import numpy as np

from deploy.benchmarks.racket_command.domain import grid_edges


def draw_scene(
  scene, config, trial, samples, translation, results, *, show_ball=True, ball=None
):
  scene.ngeom = 0
  shift = np.asarray(translation)

  def geom(kind, size, position, color, matrix=None):
    if scene.ngeom >= scene.maxgeom:
      return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
      g,
      kind,
      np.asarray(size, dtype=float),
      np.asarray(position) - shift,
      np.eye(3).ravel() if matrix is None else matrix.ravel(),
      np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1
    return g

  def line(a, b, color, width=0.003, arrow=False):
    g = geom(mujoco.mjtGeom.mjGEOM_SPHERE, [width] * 3, a, color)
    if g is not None:
      mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(a) - shift,
        np.asarray(b) - shift,
      )

  ys, zs = grid_edges(config)
  x = config["plane_x"]

  for y in ys:
    line([x, y, zs[0]], [x, y, zs[-1]], [0.7, 0.7, 0.7, 0.65], 0.0015)
  for z in zs:
    line([x, ys[0], z], [x, ys[-1], z], [0.7, 0.7, 0.7, 0.65], 0.0015)
  for iz in range(len(zs) - 1):
    for iy in range(len(ys) - 1):
      active = "active_cells" not in config or [iy, iz] in config["active_cells"]
      current = trial is not None and (iy, iz) == (trial.cell_y, trial.cell_z)
      if current:
        corners = [
          [x, ys[iy], zs[iz]],
          [x, ys[iy + 1], zs[iz]],
          [x, ys[iy + 1], zs[iz + 1]],
          [x, ys[iy], zs[iz + 1]],
        ]
        for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
          line(a, b, [1, 0.8, 0.1, 1], 0.004)
      cell = [r for r in results if r["cell_y"] == iy and r["cell_z"] == iz]
      if not cell:
        if not active:
          geom(
            mujoco.mjtGeom.mjGEOM_BOX,
            [0.0005, (ys[iy + 1] - ys[iy]) / 2, (zs[iz + 1] - zs[iz]) / 2],
            [x + 0.002, (ys[iy + 1] + ys[iy]) / 2, (zs[iz + 1] + zs[iz]) / 2],
            [0.3, 0.3, 0.3, 0.25],
          )
        continue
      key = (
        "target_pass"
        if config.get("evaluation_mode") == "policy_tracking"
        else "success"
      )
      rate = sum(bool(r.get(key)) for r in cell) / len(cell)
      geom(
        mujoco.mjtGeom.mjGEOM_BOX,
        [0.0005, (ys[iy + 1] - ys[iy]) / 2, (zs[iz + 1] - zs[iz]) / 2],
        [x + 0.002, (ys[iy + 1] + ys[iy]) / 2, (zs[iz + 1] + zs[iz]) / 2],
        [1 - rate, rate, 0.15, 0.20],
      )
  if trial is not None:
    p, n, v = map(np.asarray, (trial.position, trial.normal, trial.velocity))
    tangent = np.cross([0, 0, 1], n)
    if np.linalg.norm(tangent) < 1e-9:
      tangent = np.cross([0, 1, 0], n)
    tangent /= np.linalg.norm(tangent)
    matrix = np.column_stack([tangent, np.cross(n, tangent), n])
    radius = config.get("physical_ball", {}).get("racket_radius", 0.075)
    geom(
      mujoco.mjtGeom.mjGEOM_CYLINDER, [radius, 0.004, 0], p, [1, 0.7, 0.05, 0.5], matrix
    )
    line(p, p + n * 0.22, [1, 0.75, 0.05, 1], 0.008, True)
    line(p, p + v * 0.12, [0.9, 0.1, 0.8, 1], 0.006, True)
  trail = samples[-70:]
  for a, b in zip(trail, trail[1:], strict=False):
    line(a["position"], b["position"], [0, 1, 1, 1], 0.002)
  if samples:
    geom(
      mujoco.mjtGeom.mjGEOM_SPHERE, [0.012] * 3, samples[-1]["position"], [0, 1, 1, 1]
    )
  if trial is not None and samples and samples[-1]["time"] >= trial.tts:
    times = [s["time"] for s in samples]
    measured = np.asarray([s["position"] for s in samples])
    deadline_position = np.array(
      [np.interp(trial.tts, times, measured[:, i]) for i in range(3)]
    )
    geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.018] * 3, deadline_position, [1, 0.2, 0.1, 1])
    line(p, deadline_position, [1, 0.2, 0.1, 1], 0.003)
  if show_ball and ball is not None:
    history = ball.trajectory[-65:]
    for a, b in zip(history, history[1:], strict=False):
      line(a["position"], b["position"], [1, 0.4, 0.05, 0.65], 0.002)
    sphere = geom(
      mujoco.mjtGeom.mjGEOM_SPHERE,
      [ball.cfg["ball_radius"]] * 3,
      ball.position,
      [1, 0.4, 0.05, 1],
    )
    if sphere is not None:
      sphere.emission = 0.4


def write_reports(output, config, suite, results, metadata, status="complete"):
  from deploy.benchmarks.racket_command.grid_report import write_tts_grids

  tracking_only = metadata.get("evaluation_mode") == "policy_tracking"
  report = dict(
    schema_version=1,
    status=status,
    task="random_racket_command",
    scope=(
      "Stage 1: sampled-command tracking with existing motion selector and actor; no ball or ball planner"
      if tracking_only
      else "Independent command tracking; physical ball contact/return reported separately"
    ),
    config=config,
    suite_sha256=suite["sha256"],
    metadata=metadata,
    scheduled=len(suite["trials"]),
    evaluated=len(results),
    results=results if not config.get("compact_report") else [],
  )
  sampling = config.get("rally_sampling_audit")
  if sampling is not None:
    report["sampling_summary"] = dict(
      physical_constraints_enabled=True,
      requested_points=len(sampling),
      scheduled_points=len({t["point_id"] for t in suite["trials"]}),
      requested_commands=len(sampling)
      * config["samples_per_cell"]
      * len(config.get("tts_values", [None])),
      scheduled_commands=len(suite["trials"]),
      incomplete_points=[
        dict(
          point_id=p["point_id"],
          position=p["position"],
          status=p["status"],
          valid_direction_groups=len(p["groups"]),
        )
        for p in sampling
        if p["status"] != "complete"
      ],
      scope="Pre-policy physical validity only; exhausted random search is not proof of infeasibility or a policy failure",
    )
  n = len(results)
  accepted = sum(r["accepted"] for r in results)
  successes = sum(r["success"] for r in results)
  report["summary"] = dict(
    accepted=accepted,
    successes=successes,
    acceptance_rate=accepted / n if n else None,
    success_rate_all=successes / n if n else None,
    success_rate_accepted=successes / accepted if accepted else None,
    recovered=sum(r["recovered"] for r in results),
    target_passes=sum(bool(r.get("target_pass")) for r in results),
    target_pass_rate=sum(bool(r.get("target_pass")) for r in results) / n
    if n
    else None,
    position_passes=sum(
      bool(r.get("position_pass")) and r["accepted"] for r in results
    ),
    normal_passes=sum(bool(r.get("normal_pass")) and r["accepted"] for r in results),
    velocity_passes=sum(
      bool(r.get("velocity_pass")) and r["accepted"] for r in results
    ),
    selected_motion_in_exported_support=sum(
      r.get("selected_motion_exported_support") is True for r in results
    ),
    selected_motion_outside_exported_support=sum(
      r.get("selected_motion_exported_support") is False for r in results
    ),
  )
  balls = [r["ball"] for r in results if r.get("ball", {}).get("available")]
  report["ball_summary"] = dict(
    enabled=not tracking_only,
    available=len(balls),
    unavailable=n - len(balls) if not tracking_only else 0,
    contacts=sum(b["hit"] for b in balls),
    legal_returns=sum(b["legal_return"] for b in balls),
    contact_rate_available=sum(b["hit"] for b in balls) / len(balls) if balls else None,
    legal_return_rate_available=sum(b["legal_return"] for b in balls) / len(balls)
    if balls
    else None,
    legal_return_rate_all=sum(b["legal_return"] for b in balls) / n
    if n and not tracking_only
    else None,
  )
  if config.get("tts_values"):
    report["tts_grids"] = write_tts_grids(output, config, suite, results)
  (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
  if config.get("compact_report"):
    from deploy.benchmarks.racket_command.grid_dashboard import write_dashboard

    write_dashboard(output, config, suite, results, report)
    return report
  if not results:
    return report
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fields = [
    ("accepted", "Selector acceptance", 100),
    ("target_pass", "Deadline target pass", 100)
    if tracking_only
    else ("success", "All-command success", 100),
    ("position_error_m", "Deadline position error (cm)", 100),
    ("normal_error_degrees", "Deadline disk normal error (deg)", 1),
    ("velocity_error_mps", "Deadline velocity error (m/s)", 1),
    ("recovered", "Returned to ready", 100),
  ]
  fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
  ny, nz = config["grid"]
  for ax, (key, title, scale) in zip(axes.flat, fields, strict=True):
    values = np.full((nz, ny), np.nan)
    for z in range(nz):
      for y in range(ny):
        items = [
          r[key] for r in results if r["cell_y"] == y and r["cell_z"] == z and key in r
        ]
        if items:
          values[z, y] = np.mean(items) * scale
    rate = key in ("accepted", "success", "target_pass", "recovered")
    im = ax.imshow(
      values,
      origin="lower",
      aspect="equal",
      extent=[*config["y_range"], *config["z_range"]],
      cmap="RdYlGn" if rate else "magma_r",
      vmin=0,
      vmax=100 if rate else None,
    )
    ax.set(xlabel="Y (m)", ylabel="Z (m)", title=title + (" (%)" if rate else ""))
    fig.colorbar(im, ax=ax)
  fig.suptitle(
    f"Independent commands: {n}/{len(suite['trials'])} evaluated; "
    f"{successes} successful (all evaluated commands in denominator)"
  )
  fig.savefig(output / "capability.png", dpi=160)
  plt.close(fig)
  if any("physical_feed_available" in r for r in results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, key, title in zip(
      axes,
      ["physical_feed_available", "ball_hit", "legal_return"],
      ["Physical feed availability", "Actual paddle contact", "Legal return"],
      strict=True,
    ):
      values = np.full((nz, ny), np.nan)
      for z in range(nz):
        for y in range(ny):
          items = [
            bool(r.get(key)) for r in results if r["cell_y"] == y and r["cell_z"] == z
          ]
          if items:
            values[z, y] = 100 * np.mean(items)
      im = ax.imshow(
        values,
        origin="lower",
        aspect="equal",
        extent=[*config["y_range"], *config["z_range"]],
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
      )
      ax.set(xlabel="Y (m)", ylabel="Z (m)", title=title + " (%)")
      fig.colorbar(im, ax=ax)
    fig.suptitle(
      "Physical ball outcomes: all evaluated commands in denominator, no postcontact steering"
    )
    fig.savefig(output / "physical_ball.png", dpi=160)
    plt.close(fig)
  write_difficulty_plot(output, suite, results)
  write_point_reports(output, suite, results)
  write_html(output, report, suite)
  return report


def write_point_reports(output, suite, results):
  import matplotlib.pyplot as plt

  trials = {t["trial_id"]: t for t in suite["trials"]}
  points = sorted(
    {
      trials[r["trial_id"]].get("point_id")
      for r in results
      if trials[r["trial_id"]].get("point_id") is not None
    }
  )
  summaries = []
  for point_id in points:
    rows = [r for r in results if trials[r["trial_id"]].get("point_id") == point_id]
    target = trials[rows[0]["trial_id"]]["position"]
    scheduled = [t for t in trials.values() if t.get("point_id") == point_id]
    directions = sorted({t["direction_id"] for t in scheduled})
    summary = dict(
      point_id=point_id,
      position=target,
      evaluated=len(rows),
      scheduled=len(scheduled),
      accepted=sum(r["accepted"] for r in rows),
      successes=sum(r["success"] for r in rows),
      target_passes=sum(bool(r.get("target_pass")) for r in rows),
    )
    physical = any("ball_hit" in r for r in rows)
    if physical:
      summary.update(
        contacts=sum(bool(r.get("ball_hit")) for r in rows),
        legal_returns=sum(bool(r.get("legal_return")) for r in rows),
      )
    summary["directions"] = []
    for direction_id in directions:
      commands = [t for t in scheduled if t["direction_id"] == direction_id]
      vector = np.asarray(commands[0]["velocity"])
      summary["directions"].append(
        dict(
          direction_id=direction_id,
          normal=commands[0]["normal"],
          velocity_direction=(vector / np.linalg.norm(vector)).tolist(),
          speed_levels=sorted(
            {round(float(np.linalg.norm(t["velocity"])), 8) for t in commands}
          ),
          tts_levels=sorted({t["tts"] for t in commands}),
        )
      )
    summaries.append(summary)
    if not all(trials[r["trial_id"]].get("speed_id") is not None for r in rows):
      continue
    metrics = (
      [
        ("success", "Joint tracking success", True),
        ("velocity_error_mps", "Velocity error (m/s)", False),
        ("ball_hit", "Physical contact", True),
      ]
      if physical
      else [
        ("target_pass", "Deadline target pass", True),
        ("position_error_m", "Position error (m)", False),
        ("normal_error_degrees", "Normal error (deg)", False),
        ("velocity_error_mps", "Velocity error (m/s)", False),
      ]
    )
    fig, axes = plt.subplots(
      len(metrics),
      len(directions),
      figsize=(3.5 * len(directions), 3 * len(metrics)),
      squeeze=False,
      constrained_layout=True,
    )
    for col, d in enumerate(directions):
      group = [r for r in rows if trials[r["trial_id"]]["direction_id"] == d]
      commands = [t for t in scheduled if t["direction_id"] == d]
      speeds = sorted(
        {round(float(np.linalg.norm(t["velocity"])), 8) for t in commands}
      )
      times = sorted({round(float(t["tts"]), 8) for t in commands})
      for row, (metric, title, rate) in enumerate(metrics):
        values = np.full((len(speeds), len(times)), np.nan)
        for iy, s in enumerate(speeds):
          for ix, t in enumerate(times):
            members = [
              r
              for r in group
              if np.isclose(np.linalg.norm(trials[r["trial_id"]]["velocity"]), s)
              and np.isclose(trials[r["trial_id"]]["tts"], t)
              and metric in r
            ]
            if members:
              values[iy, ix] = np.mean([r[metric] for r in members]) * (
                100 if rate else 1
              )
        ax = axes[row, col]
        im = ax.imshow(
          values,
          origin="lower",
          aspect="auto",
          cmap="RdYlGn" if rate else "magma_r",
          vmin=0,
          vmax=100 if rate else max(1e-9, max(r.get(metric, 0) for r in rows)),
        )
        ax.set_xticks(range(len(times)), [f"{v:.3f}" for v in times])
        ax.set_yticks(range(len(speeds)), [f"{v:.2f}" for v in speeds])
        ax.set(
          xlabel="TTS (s)",
          ylabel="Requested speed (m/s)",
          title=f"Direction {d}: {title}",
        )
        fig.colorbar(im, ax=ax)
    fig.suptitle(
      f"Fixed point {point_id}: p={np.round(target, 3).tolist()}; different columns have fixed normal/velocity directions"
    )
    folder = output / "points"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"point_{point_id:04d}.png", dpi=140)
    if len(points) == 1:
      fig.savefig(output / "point_sweeps.png", dpi=140)
    plt.close(fig)
  (output / "point_summary.json").write_text(json.dumps(summaries, indent=2))


def write_difficulty_plot(output, suite, results):
  import matplotlib.pyplot as plt

  trials = {t["trial_id"]: t for t in suite["trials"]}
  fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
  definitions = [
    ("TTS (s)", lambda t: t["tts"]),
    ("Requested speed (m/s)", lambda t: np.linalg.norm(t["velocity"])),
    (
      "Normal deviation from +X (deg)",
      lambda t: np.degrees(np.arccos(np.clip(t["normal"][0], -1, 1))),
    ),
  ]
  for ax, (label, get_value) in zip(axes.flat, definitions, strict=False):
    values = np.asarray([get_value(trials[r["trial_id"]]) for r in results])
    edges = np.linspace(values.min() - 1e-9, values.max() + 1e-9, 7)
    centers = (edges[:-1] + edges[1:]) / 2
    for metric, title in (("accepted", "Accepted"), ("success", "All-command success")):
      rates = []
      for low, high in zip(edges, edges[1:], strict=False):
        members = [r for v, r in zip(values, results, strict=True) if low <= v < high]
        rates.append(100 * np.mean([r[metric] for r in members]) if members else np.nan)
      ax.plot(centers, rates, "o-", label=title)
    for center, low, high in zip(centers, edges, edges[1:], strict=False):
      count = int(np.sum((values >= low) & (values < high)))
      ax.text(center, 105, f"n={count}", ha="center", fontsize=8)
    ax.set(xlabel=label, ylabel="Rate (%)", ylim=(-5, 115))
    ax.legend(fontsize=8)
  counts = {}
  for r in results:
    key = "rejected" if not r["accepted"] else str(r.get("motion_index"))
    counts[key] = counts.get(key, 0) + 1
  axes[1, 1].bar(list(counts), list(counts.values()))
  axes[1, 1].set(xlabel="Selected motion / rejected", ylabel="Commands")
  fig.suptitle("Independent command sensitivity; sample counts shown")
  fig.savefig(output / "difficulty.png", dpi=160)
  plt.close(fig)


def write_html(output, report, suite):
  payload = json.dumps(dict(report=report, suite=suite), allow_nan=False).replace(
    "</", "<\\/"
  )
  html = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>随机 racket command 能力评测</title>
<style>body{font:16px system-ui;margin:24px;color:#192536;background:#f5f7fa}
svg{background:white;max-width:960px;width:100%;border-radius:8px}select,input{margin:8px}
pre{white-space:pre-wrap;background:white;padding:16px}#tip{min-height:65px}</style>
<h1>随机 racket command 能力图</h1>
<p id="description">固定 X 平面，横轴 Y，纵轴 Z。灰色表示无样本或训练包络外。拒绝目标保留在分母；指令达标率与物理回球率分开统计。</p>
<pre id="ball_summary"></pre>
TTS_GRID_GALLERY
<p id="summary"></p><label>指标<select id="metric">
<option value="success">完整成功率（含回位）</option><option value="accepted">选择器接受率</option>
<option value="position_error_m">位置误差 (m)</option><option value="normal_error_degrees">拍面误差 (°)</option>
<option value="velocity_error_mps">速度向量误差 (m/s)</option><option value="recovered">回位完成率</option>
<option value="physical_feed_available">物理来球可用率</option><option value="ball_hit">实际碰拍率</option><option value="legal_return">物理合法回球率</option></select></label>
<label>TTS <input id="tmin" type="number" step=".01"> 至 <input id="tmax" type="number" step=".01"></label>
<label>速度 <input id="vmin" type="number" step=".1"> 至 <input id="vmax" type="number" step=".1"></label>
<label>朝向偏转 ≤ <input id="angle" type="number" min="0" max="180" value="180">°</label>
<label>motion <select id="motion"><option value="all">全部（含拒绝）</option></select></label>
<svg id="chart" viewBox="0 0 960 580"></svg><p id="tip">鼠标移到网格上查看样本数和误差。</p>
<section id="point_section"><h2>固定点位：速度 × TTS</h2>
<label>点位 <select id="point"></select></label><label>方向 <select id="direction"></select></label>
<p id="point_info"></p><svg id="point_chart" viewBox="0 0 960 430"></svg></section>
<h2>失败/拒绝原因</h2><pre id="reasons"></pre>
<h2>网格样本详情</h2><pre id="details">点击网格查看目标、所选 motion 和误差。</pre>
<h2>范围与测试说明</h2><pre id="config"></pre>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),r=data.report,c=r.config;
const trials=new Map(data.suite.trials.map(t=>[t.trial_id,t]));
document.getElementById('ball_summary').textContent=JSON.stringify(r.ball_summary,null,2);
if(r.metadata.evaluation_mode==='policy_tracking'){
document.getElementById('description').textContent='第一阶段：固定 X 平面上的 command 跟踪评测。灰色表示无样本或训练支持域外，拒绝目标保留在分母。达标要求在指定时刻同时满足位置、拍面法向和速度误差阈值。';
document.getElementById('ball_summary').textContent='第一阶段：直接采样 command → 现有 motion 选择器 → policy。无来球，无球轨迹 planner；在指定 TTS 时刻检查球拍位置、法向、速度。';
if(r.sampling_summary)document.getElementById('ball_summary').textContent+=' 正常对打约束：每条command均附独立合法来球和预计过网/对方台内首次落点的模型校验。生成 '+r.sampling_summary.scheduled_commands+'/'+r.sampling_summary.requested_commands+' 条；采样搜索未找到足够合法组合的点位单独记录，不计policy失败。';
document.querySelectorAll('#metric option').forEach(o=>{if(['physical_feed_available','ball_hit','legal_return'].includes(o.value))o.remove()});
let targetOption=document.createElement('option');targetOption.value='target_pass';targetOption.textContent='TTS 时刻三项同时达标率';document.getElementById('metric').prepend(targetOption);document.getElementById('metric').value='target_pass';
}
const el=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
const pointIds=[...new Set(data.suite.trials.map(t=>t.point_id).filter(v=>v!==undefined&&v!==null))].sort((a,b)=>a-b);
pointIds.forEach(v=>{let t=data.suite.trials.find(t=>t.point_id===v),o=document.createElement('option');o.value=v;o.textContent=`${v}: [${t.position.map(x=>x.toFixed(3)).join(', ')}]`;el('point').appendChild(o)});
el('point_section').hidden=pointIds.length===0;
function updateDirections(){let old=el('direction').value;el('direction').replaceChildren();
let ids=[...new Set(data.suite.trials.filter(t=>String(t.point_id)===el('point').value).map(t=>t.direction_id))].sort((a,b)=>a-b);
ids.forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;el('direction').appendChild(o)});
if(ids.some(v=>String(v)===old))el('direction').value=old}
updateDirections();
['tmin','tmax'].forEach((id,i)=>el(id).value=c.tts_seconds[i]);
['vmin','vmax'].forEach((id,i)=>el(id).value=c.speed_mps[i]);
[...new Set(r.results.map(v=>v.motion_index).filter(v=>v!==null&&v!==undefined))].sort((a,b)=>a-b).forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;el('motion').appendChild(o)});
el('config').textContent=JSON.stringify({scope:r.scope,status:r.status,suite_sha256:r.suite_sha256,config:c,metadata:r.metadata},null,2);
function tag(name,attrs,text){let e=document.createElementNS(ns,name);for(let [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;el('chart').appendChild(e);return e}
function drawPoint(rows,key,rate,max){if(!pointIds.length)return;el('point_chart').replaceChildren();
function pt(name,attrs,text){let e=document.createElementNS(ns,name);for(let [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;el('point_chart').appendChild(e);return e}
let commands=data.suite.trials.filter(t=>String(t.point_id)===el('point').value&&String(t.direction_id)===el('direction').value);
if(!commands.length)return;let speeds=[...new Set(commands.map(t=>Math.hypot(...t.velocity).toFixed(6)))].map(Number).sort((a,b)=>a-b),times=[...new Set(commands.map(t=>t.tts.toFixed(6)))].map(Number).sort((a,b)=>a-b);
let example=commands[0],speed=Math.hypot(...example.velocity);
el('point_info').textContent=`目标位置 [${example.position.map(v=>v.toFixed(3))}] m；拍面法向 [${example.normal.map(v=>v.toFixed(3))}]；速度方向 [${example.velocity.map(v=>(v/speed).toFixed(3))}]。当前指标 ${key}；速度和 TTS 改变时以上位置/方向保持一致。`;
let group=rows.filter(v=>{let t=trials.get(v.trial_id);return String(t.point_id)===el('point').value&&String(t.direction_id)===el('direction').value});
let w=800/times.length,h=300/speeds.length;
speeds.forEach((s,y)=>times.forEach((t,x)=>{let all=group.filter(v=>{let c=trials.get(v.trial_id);return Math.abs(Math.hypot(...c.velocity)-s)<1e-5&&Math.abs(c.tts-t)<1e-5}),vs=all.filter(v=>v[key]!==undefined),val=vs.length?vs.reduce((a,v)=>a+Number(v[key]),0)/vs.length:null;
let box=pt('rect',{x:100+x*w,y:25+(speeds.length-1-y)*h,width:w-2,height:h-2,fill:val===null?'#d8dee7':`hsl(${120*(rate?val:1-val/max)},65%,55%)`});
box.onclick=()=>el('details').textContent=JSON.stringify(all.map(v=>({command:trials.get(v.trial_id),result:v})),null,2);
pt('text',{x:100+(x+.5)*w,y:25+(speeds.length-y-.5)*h,'text-anchor':'middle','pointer-events':'none'},val===null?'—':(rate?(100*val).toFixed(1)+'%':val.toFixed(3))+` / n=${all.length}`)}));
times.forEach((v,i)=>pt('text',{x:100+(i+.5)*w,y:355,'text-anchor':'middle'},v.toFixed(3)));
speeds.forEach((v,i)=>pt('text',{x:85,y:25+(speeds.length-i-.5)*h,'text-anchor':'end'},v.toFixed(2)));
pt('text',{x:500,y:400,'text-anchor':'middle'},'TTS (s)');pt('text',{x:10,y:18},'速度 (m/s)')}
function draw(){el('chart').replaceChildren();let key=el('metric').value;
let rows=r.results.filter(v=>{let t=trials.get(v.trial_id),s=Math.hypot(...t.velocity),a=Math.acos(Math.min(1,t.normal[0]))*180/Math.PI;
return t.tts>=+el('tmin').value-1e-9&&t.tts<=+el('tmax').value+1e-9&&s>=+el('vmin').value-1e-9&&s<=+el('vmax').value+1e-9&&a<=+el('angle').value&&(el('motion').value==='all'||String(v.motion_index)===el('motion').value)});
let n=rows.length,accepted=rows.filter(v=>v.accepted).length,ok=rows.filter(v=>v.success).length;
el('summary').textContent=`已评测 ${r.evaluated}/${r.scheduled}；当前筛选 ${n} 条，接受 ${accepted}，综合成功 ${ok}，成功率 ${n?(100*ok/n).toFixed(1):'—'}%。`;
let rate=['success','target_pass','accepted','recovered','physical_feed_available','ball_hit','legal_return'].includes(key),max=rate?1:Math.max(.0001,...rows.map(v=>v[key]??0));
let [ny,nz]=c.grid,plotHeight=800*(c.z_range[1]-c.z_range[0])/(c.y_range[1]-c.y_range[0]),w=800/ny,h=plotHeight/nz;el('chart').setAttribute('viewBox','0 0 960 '+(plotHeight+130));el('chart').setAttribute('preserveAspectRatio','xMidYMid meet');
for(let z=0;z<nz;z++)for(let y=0;y<ny;y++){let all=rows.filter(v=>v.cell_y===y&&v.cell_z===z),vs=all.filter(v=>v[key]!==undefined),val=vs.length?vs.reduce((a,v)=>a+Number(v[key]),0)/vs.length:null;
let col=val===null?'#d8dee7':`hsl(${120*(rate?val:1-val/max)},65%,55%)`;
let box=tag('rect',{x:90+y*w,y:40+(nz-1-z)*h,width:w,height:h,fill:col,stroke:'white','stroke-width':2});
box.onclick=()=>el('details').textContent=JSON.stringify(all.map(v=>({command:trials.get(v.trial_id),result:v})),null,2);
box.onmouseenter=()=>el('tip').textContent=`Y=[${(c.y_range[0]+y*(c.y_range[1]-c.y_range[0])/ny).toFixed(3)},${(c.y_range[0]+(y+1)*(c.y_range[1]-c.y_range[0])/ny).toFixed(3)}]，Z 网格 ${z}；${all.length} 条，指标可测 ${vs.length} 条；${key}=${val===null?'—':(rate?(100*val).toFixed(1)+'%':val.toFixed(3))}`;
tag('text',{x:90+(y+.5)*w,y:40+(nz-z-.5)*h,'text-anchor':'middle','font-size':13,'pointer-events':'none'},val===null?'—':(rate?(100*val).toFixed(0)+'%':val.toFixed(2))+` / n=${all.length}`)}
for(let i=0;i<=ny;i++)tag('text',{x:90+i*w,y:40+plotHeight+30,'text-anchor':'middle'},(c.y_range[0]+i*(c.y_range[1]-c.y_range[0])/ny).toFixed(2));
for(let i=0;i<=nz;i++)tag('text',{x:75,y:40+plotHeight-i*h,'text-anchor':'end'},(c.z_range[0]+i*(c.z_range[1]-c.z_range[0])/nz).toFixed(2));
tag('text',{x:490,y:40+plotHeight+70,'text-anchor':'middle'},`Y (m), X=${c.plane_x} m`);tag('text',{x:20,y:25},'Z (m)');
drawPoint(rows,key,rate,max);
let reasons={};rows.filter(v=>!v.success).forEach(v=>{let k=v.failure??(!v.accepted?v.decision?.reason:'tracking_tolerance_or_recovery');reasons[k]=(reasons[k]??0)+1});el('reasons').textContent=JSON.stringify(reasons,null,2)}
document.querySelectorAll('input,select').forEach(e=>e.oninput=draw);el('point').oninput=()=>{updateDirections();draw()};draw();</script></html>"""
  gallery = ""
  if report.get("tts_grids"):
    times = sorted({c["tts"] for c in report["tts_grids"]})
    gallery = '<h2>按 TTS 分组的网格</h2><p><a href="tts_grids/summary.csv">逐格 CSV</a> · <a href="tts_grids/summary.json">逐格 JSON</a></p>'
    for i, tts in enumerate(times):
      path = f"tts_grids/tts_{i:02d}_{tts:g}s.png"
      gallery += f'<details><summary>TTS = {tts:g}s</summary><a href="{path}"><img src="{path}" style="max-width:100%" loading="lazy"></a></details>'
  (output / "report.html").write_text(
    html.replace("TTS_GRID_GALLERY", gallery).replace("PAYLOAD", payload)
  )
