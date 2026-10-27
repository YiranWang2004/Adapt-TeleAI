"""Standalone aggregate HTML and overview figures for large full-grid runs."""

import csv
import json

import numpy as np

from deploy.benchmarks.racket_command.grid_report import RATES, summarize


def write_dashboard(output, config, suite, results, report):
  trials = {t["trial_id"]: t for t in suite["trials"]}
  times = sorted({t["tts"] for t in trials.values()})
  enabled = {"accepted", "target_pass", "success"} | {
    key for key in RATES if any(key in r for r in results)
  }
  totals = []
  for tts in times:
    rows = [r for r in results if trials[r["trial_id"]]["tts"] == tts]
    totals.append(
      dict(
        tts=tts,
        **summarize(
          rows,
          sum(t["tts"] == tts for t in trials.values()),
          enabled,
        ),
      )
    )
  report["tts_summary"] = totals
  report["results_file"] = "results.jsonl"
  report["statistics"] = dict(
    rate_denominator="All evaluated targets, including rejections and unavailable feeds",
    rate_ci="95% Wilson binomial interval; see actual per-cell sample counts",
    error_denominator="Only observed deadline/contact errors; n is shown separately",
    paired_targets=True,
    physical_scope="Command-conditioned incoming ball, original selector and actor; no incoming-ball planner",
  )
  (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
  if totals:
    with (output / "tts_summary.csv").open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(totals[0]))
      writer.writeheader()
      writer.writerows(totals)
  if results:
    _overview(output, config, report["tts_grids"], totals)
  payload = json.dumps(
    dict(
      config={
        k: config[k]
        for k in (
          "grid",
          "plane_x",
          "y_range",
          "z_range",
          "samples_per_cell",
          "tts_values",
          "tolerances",
        )
      },
      status=report["status"],
      evaluated=len(results),
      scheduled=len(trials),
      cells=report["tts_grids"],
      totals=totals,
      metadata=report["metadata"],
      statistics=report["statistics"],
      sha256=suite["sha256"],
    ),
    ensure_ascii=False,
    allow_nan=False,
  ).replace("</", "<\\/")
  ny, nz = config["grid"]
  n = config["samples_per_cell"]
  half = (
    1.959963984540054
    * np.sqrt(0.25 / n + 1.959963984540054**2 / (4 * n * n))
    / (1 + 1.959963984540054**2 / n)
  )
  sampling = f"本轮覆盖 {ny}×{nz}={ny * nz} 格，每格每档执行 {n} 个随机 target。"
  if config.get("region_crop"):
    sampling += "已移除原始网格最下方一层，保留其余格子的原有边界。"
  sampling += "区域内的训练支持域外位置仍保留。各 TTS 复用同一批位置、拍面法向、速度。每次均恢复同一机器人准备状态。物理来球按 command 构造；使用现有选择器与 actor，不经过来球估计或回球 planner。"
  uncertainty = f"逐格成功率采用 95% Wilson 置信区间。n={n} 时最宽约 ±{half * 100:.1f} 个百分点；均值曲线显示 ±1.96 SEM。结果用于本配置下的模拟测试。"
  (output / "report.html").write_text(
    HTML.replace("SAMPLING_SCOPE", sampling)
    .replace("UNCERTAINTY_NOTE", uncertainty)
    .replace("PAYLOAD", payload)
  )


def _overview(output, config, cells, totals):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  times = [t["tts"] for t in totals]
  fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
  for key, label in [
    ("accepted", "Accepted"),
    ("target_pass", "Deadline tracking pass"),
    ("ball_hit", "Physical contact"),
    ("legal_return", "Legal return"),
  ]:
    if not any(t[key + "_rate"] is not None for t in totals):
      continue
    values = np.array([t[key + "_rate"] for t in totals], dtype=float)
    low = np.array([t[key + "_ci95_low"] for t in totals], dtype=float)
    high = np.array([t[key + "_ci95_high"] for t in totals], dtype=float)
    axes[0].plot(times, values * 100, "o-", label=label)
    axes[0].fill_between(times, low * 100, high * 100, alpha=0.12)
  axes[0].set(
    xlabel="TTS (s)",
    ylabel="Rate (%)",
    ylim=(-2, 102),
    title="All targets; shaded 95% Wilson CI",
  )
  axes[0].legend(fontsize=8)
  for key, label in [
    ("position_error_m", "Racket-target"),
    ("racket_ball_distance_at_deadline_m", "Racket-ball at deadline"),
    ("racket_ball_distance_at_contact_m", "Racket-ball at contact"),
  ]:
    values = np.array([t[key + "_mean"] for t in totals], dtype=float) * 100
    sem = np.array([t[key + "_sem"] for t in totals], dtype=float) * 100
    axes[1].errorbar(times, values, yerr=1.96 * sem, marker="o", capsize=2, label=label)
  axes[1].set(
    xlabel="TTS (s)",
    ylabel="Mean distance (cm)",
    title="Observed samples only; +/-1.96 SEM",
  )
  axes[1].legend(fontsize=8)
  axes[2].plot(
    times,
    [t["velocity_error_mps_mean"] for t in totals],
    "o-",
    label="Velocity error (m/s)",
  )
  axes[2].plot(
    times,
    [t["normal_error_degrees_mean"] for t in totals],
    "o-",
    label="Normal error (deg)",
  )
  axes[2].set(xlabel="TTS (s)", title="Deadline command tracking errors")
  axes[2].legend(fontsize=8)
  for ax in axes:
    ax.grid(alpha=0.2)
  fig.savefig(output / "tts_overview.png", dpi=170)
  plt.close(fig)
  for key, label, scale in (
    ("ball_hit_rate", "Physical contact (%)", 100),
    ("target_pass_rate", "Deadline tracking pass (%)", 100),
    (
      "racket_ball_distance_at_deadline_m_mean",
      "Mean racket-ball distance at deadline (cm)",
      100,
    ),
    ("position_error_m_mean", "Mean racket-target distance at deadline (cm)", 100),
  ):
    columns = min(5, len(times))
    rows = (len(times) + columns - 1) // columns
    fig, axes = plt.subplots(
      rows,
      columns,
      figsize=(4.4 * columns, 4 * rows),
      constrained_layout=True,
      squeeze=False,
    )
    maximum = (
      100
      if key.endswith("_rate")
      else max([c[key] * scale for c in cells if c[key] is not None] + [1e-6])
    )
    cmap = plt.get_cmap("RdYlGn" if key.endswith("_rate") else "magma_r").with_extremes(
      bad="#ddd"
    )
    ny, nz = config["grid"]
    for ax, tts in zip(axes.flat, times, strict=False):
      data = np.full((nz, ny), np.nan)
      for c in cells:
        if c["tts"] == tts and c[key] is not None:
          data[c["cell_z"], c["cell_y"]] = c[key] * scale
      im = ax.imshow(
        data,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=maximum,
        extent=[*config["y_range"], *config["z_range"]],
      )
      ax.vlines(
        np.linspace(*config["y_range"], ny + 1),
        *config["z_range"],
        color="white",
        linewidth=0.5,
        alpha=0.65,
      )
      ax.hlines(
        np.linspace(*config["z_range"], nz + 1),
        *config["y_range"],
        color="white",
        linewidth=0.5,
        alpha=0.65,
      )
      ax.set(title=f"TTS={tts:g}s", xlabel="Y (m)", ylabel="Z (m)")
    for ax in list(axes.flat)[len(times) :]:
      ax.set_visible(False)
    fig.colorbar(im, ax=list(axes.flat), shrink=0.85, label=label)
    fig.suptitle(
      f"Full {ny} x {nz} grid | {label} | {config['samples_per_cell']} targets/cell/TTS"
    )
    fig.savefig(output / f"overview_{key}.png", dpi=170)
    plt.close(fig)


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MuJoCo 完整网格 · TTS 评测</title>
<style>
:root{color-scheme:light;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#1e293b;background:#f1f5f9}body{max-width:1440px;margin:28px auto;padding:0 20px}h1{font-size:27px;margin-bottom:8px}h2{font-size:20px}p{line-height:1.7}a{color:#0369a1}section,.card{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin:18px 0}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{flex:1;margin:0;min-width:150px}.big{font-size:27px;font-weight:650}label{display:inline-block;margin:6px 20px 6px 0}select,input,button{font:inherit;padding:6px;border:1px solid #cbd5e1;border-radius:6px}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 8px;border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}th{background:#f8fafc}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.muted{color:#64748b;font-size:13px}.tip{white-space:pre-wrap;line-height:1.7;background:#f8fafc;padding:14px;min-height:80px;border-radius:8px}.links{display:flex;gap:18px;flex-wrap:wrap}img{max-width:100%}pre{white-space:pre-wrap;word-break:break-all;font-size:12px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:850px){.twocol{grid-template-columns:1fr}}
</style>
<h1>MuJoCo 完整网格 · TTS 评测</h1><p id="scope"></p>
<div class="cards" id="cards"></div><p id="window_note" class="tip"></p>
<section><h2>统计口径</h2><p>SAMPLING_SCOPE</p><p>“碰拍率”来自实际移动拍面碰撞；“跟踪达标率”要求在指定 TTS 同时满足位置、法向和速度阈值；“完整成功率”另要求回位且无失败。拒绝、物理来球不可构造均保留在成功率分母。距离仅统计有效观测并单列 n：拍心–球心距离不等于球面到拍面的间隙，deadline 前碰拍后的分离也会计入。实际碰拍时刻距离只对碰拍样本有定义。</p><p class="muted">UNCERTAINTY_NOTE</p><div class="links"><a href="tts_grids/summary.csv">逐格 CSV</a><a href="tts_summary.csv">TTS 汇总 CSV</a><a href="report.json">统计 JSON</a><a href="suite.json">冻结测试序列</a><a href="results.jsonl">逐次结果 JSONL</a></div></section>
<section><h2>全区域 TTS 趋势</h2><div class="twocol"><svg id="rates" viewBox="0 0 660 360"></svg><svg id="errors" viewBox="0 0 660 360"></svg></div><div class="links"><a href="tts_overview.png">下载趋势图 PNG</a><a href="overview_ball_hit_rate.png">各档碰拍率总览</a><a href="overview_target_pass_rate.png">各档跟踪达标率总览</a><a href="overview_racket_ball_distance_at_deadline_m_mean.png">各档拍球距离总览</a></div></section>
<section><h2>逐格对照</h2><label>TTS <select id="tts"></select></label><label>指标 <select id="metric"></select></label><label>误差统计 <select id="stat"><option value="mean">均值</option><option value="median">中位数</option><option value="p95">P95</option><option value="std">标准差</option></select></label><a id="png">该 TTS 完整网格 PNG</a><p id="legend" class="muted"></p><svg id="grid" viewBox="0 0 1100 670"></svg><div id="tip" class="tip">悬停或点击格子，查看样本数、成功率的置信区间和平均误差。</div></section>
<section><h2>各档汇总</h2><div class="scroll"><table id="totals"></table></div></section>
<section><h2>当前 TTS 的全部网格</h2><div class="scroll"><table id="cells"></table></div></section>
<details><summary>模型、测试范围与复现信息</summary><pre id="metadata"></pre></details>
<script id="data" type="application/json">PAYLOAD</script><script>
const d=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
const times=d.totals.map(t=>t.tts),pct=v=>v==null?'—':(v*100).toFixed(1)+'%',num=(v,n=2)=>v==null?'—':Number(v).toFixed(n),cm=v=>v==null?'—':num(v*100);
const metrics=[['ball_hit','实际碰拍率','rate'],['target_pass','TTS 跟踪联合达标率','rate'],['success','完整成功率（含回位）','rate'],['legal_return','合法回球率','rate'],['accepted','选择器接受率','rate'],['command_in_training_support','位置/法向/速度位于训练支持域的比例（不含 TTS）','rate'],['racket_ball_distance_at_deadline_m','TTS 时刻拍心–球心距离 (cm)','error'],['racket_ball_distance_at_contact_m','实际碰拍时刻拍心–球心距离 (cm)','error'],['position_error_m','TTS 时刻拍心–target 误差 (cm)','error'],['normal_error_degrees','法向误差 (°)','error'],['velocity_error_mps','速度向量误差 (m/s)','error']];
el('scope').textContent=`X=${d.config.plane_x}m；Y=[${d.config.y_range.map(v=>v.toFixed(4))}]m；Z=[${d.config.z_range.map(v=>v.toFixed(4))}]m（桌面 Z=0）。状态：${d.status}。`;
el('cards').innerHTML=[['已完成',d.evaluated.toLocaleString()+' / '+d.scheduled.toLocaleString()],['空间网格',d.config.grid.join(' × ')+' = '+d.config.grid.reduce((a,b)=>a*b,1)],['TTS 档数',times.length],['每格每档',d.config.samples_per_cell+' 次']].map(([k,v])=>`<div class="card"><div class="muted">${k}</div><div class="big">${v}</div></div>`).join('');
const blocked=d.totals.filter(t=>t.evaluated>0&&t.rejected_start_window_count===t.evaluated);el('window_note').textContent=blocked.length?'启动窗口边界：'+blocked.map(t=>t.tts+'s（'+t.evaluated+' 次）').join('、')+' 的指令全部被原选择器拒绝。原选择器最大 TTS='+d.metadata.selector_max_tts+'s，max_start_frame='+d.metadata.selector_max_start_frame+'。这些样本仍计入总分母；其距离是拒绝后的诊断，不表示完成了挥拍执行。':'本轮没有整档因启动窗口被拒绝；逐条原因保存在 results.jsonl。';
times.forEach(t=>el('tts').add(new Option(t+' s',t)));metrics.forEach(m=>el('metric').add(new Option(m[1],m[0])));el('tts').value=times.includes(.4)?.4:times[0];
function node(parent,name,attrs,text){let e=document.createElementNS(ns,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;parent.appendChild(e);return e}
function ci(c,k){return `[${pct(c[k+'_ci95_low'])}, ${pct(c[k+'_ci95_high'])}]`}
function drawGrid(){let t=+el('tts').value,m=metrics.find(m=>m[0]===el('metric').value),rate=m[2]==='rate',key=m[0]+'_'+(rate?'rate':el('stat').value),scale=!rate&&m[0].endsWith('_m')?100:1,svg=el('grid');svg.replaceChildren();el('stat').disabled=rate;let max=rate?1:Math.max(1e-8,...d.cells.map(c=>(c[key]||0)*scale));let [ny,nz]=d.config.grid,ySpan=d.config.y_range[1]-d.config.y_range[0],zSpan=d.config.z_range[1]-d.config.z_range[0],pixelsPerMeter=920/ySpan,plotHeight=zSpan*pixelsPerMeter,w=920/ny,h=plotHeight/nz;svg.setAttribute('viewBox','0 0 1100 '+(plotHeight+150));svg.setAttribute('preserveAspectRatio','xMidYMid meet');
el('legend').textContent='Y、Z 使用相同的米/像素比例，格子长宽按真实尺寸显示。'+(rate?'颜色范围固定 0–100%；每格显示成功率与全部已测样本数。':'同一指标在各 TTS 使用相同颜色尺度；每格显示误差及该误差的有效样本数。');
el('png').href=`tts_grids/tts_${String(times.indexOf(t)).padStart(2,'0')}_${t}s.png`;
let selected=d.cells.filter(c=>c.tts===t);selected.forEach(c=>{let val=c[key],x=100+c.cell_y*w,y=45+(nz-1-c.cell_z)*h,v=val==null?null:val*scale,col=v==null?'#d1d5db':`hsl(${rate?120*v:45*(1-v/max)},${rate?58:78}%,${rate?87-50*v:94-58*v/max}%)`;let g=node(svg,'g',{});node(g,'rect',{x,y,width:w,height:h,fill:col,stroke:'white','stroke-width':2});let n=rate?c.evaluated:c[m[0]+'_n'];node(g,'text',{x:x+w/2,y:y+h/2-3,'text-anchor':'middle','font-size':17,fill:rate&&v>.7?'white':'#111827'},v==null?'—':(rate?pct(val):num(v,1)));node(g,'text',{x:x+w/2,y:y+h/2+20,'text-anchor':'middle','font-size':12,fill:rate&&v>.7?'white':'#111827'},`n=${n} / ${c.scheduled}`);
function show(){el('tip').textContent=`格 (${c.cell_y}, ${c.cell_z}) · Y=[${num(c.y_min,4)}, ${num(c.y_max,4)}] · Z=[${num(c.z_min,4)}, ${num(c.z_max,4)}]m · TTS=${t}s\n已测 ${c.evaluated}/${c.scheduled}；deadline 观测 ${c.deadline_observed}；物理来球可构造 ${c.physical_feed_available}；位置/法向/速度支持域 ${pct(c.command_in_training_support_rate)}；启动窗口拒绝 ${c.rejected_start_window_count}；位置拒绝 ${c.rejected_position_count}\n碰拍 ${c.ball_hit_count}/${c.evaluated} = ${pct(c.ball_hit_rate)}，95% CI ${ci(c,'ball_hit')}；跟踪达标 ${pct(c.target_pass_rate)}，95% CI ${ci(c,'target_pass')}；合法回球 ${pct(c.legal_return_rate)}\nTTS 拍球距离均值 ${cm(c.racket_ball_distance_at_deadline_m_mean)}cm，中位数 ${cm(c.racket_ball_distance_at_deadline_m_median)}cm，P95 ${cm(c.racket_ball_distance_at_deadline_m_p95)}cm，n=${c.racket_ball_distance_at_deadline_m_n}；碰拍时刻距离均值 ${cm(c.racket_ball_distance_at_contact_m_mean)}cm，n=${c.racket_ball_distance_at_contact_m_n}\n拍心–target 均值 ${cm(c.position_error_m_mean)}cm；法向 ${num(c.normal_error_degrees_mean)}°；速度向量 ${num(c.velocity_error_mps_mean)}m/s`};g.onmouseenter=show;g.onclick=show;});
for(let i=0;i<=ny;i++)node(svg,'text',{x:100+i*w,y:45+plotHeight+25,'text-anchor':'middle','font-size':13},num(d.config.y_range[0]+i*(d.config.y_range[1]-d.config.y_range[0])/ny,2));for(let i=0;i<=nz;i++)node(svg,'text',{x:85,y:45+plotHeight-i*h+4,'text-anchor':'end','font-size':13},num(d.config.z_range[0]+i*(d.config.z_range[1]-d.config.z_range[0])/nz,2));node(svg,'text',{x:560,y:45+plotHeight+70,'text-anchor':'middle','font-size':17},'Y (m)');node(svg,'text',{x:50,y:25,'font-size':17},'Z (m)');node(svg,'text',{x:560,y:27,'text-anchor':'middle','font-size':18},`TTS=${t}s · ${m[1]}`);renderCells(selected);}
const headers=['TTS / 格子','已测/计划','接受率','碰拍率 [95% CI]','跟踪达标率','合法回球率','TTS 拍球距离 cm / n','碰拍距离 cm / n','拍心–target cm','法向 °','速度 m/s'];
function table(id,rows,label){el(id).innerHTML='<thead><tr>'+headers.map(s=>'<th>'+s+'</th>').join('')+'</tr></thead><tbody>'+rows.map(c=>'<tr>'+[label(c),`${c.evaluated}/${c.scheduled}`,pct(c.accepted_rate),pct(c.ball_hit_rate)+' '+ci(c,'ball_hit'),pct(c.target_pass_rate),pct(c.legal_return_rate),cm(c.racket_ball_distance_at_deadline_m_mean)+' / '+c.racket_ball_distance_at_deadline_m_n,cm(c.racket_ball_distance_at_contact_m_mean)+' / '+c.racket_ball_distance_at_contact_m_n,cm(c.position_error_m_mean),num(c.normal_error_degrees_mean),num(c.velocity_error_mps_mean)].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody>'}
function renderCells(rows){table('cells',rows,c=>`(${c.cell_y}, ${c.cell_z})`)}table('totals',d.totals,c=>c.tts+'s');
function lineChart(id,series,ylabel){let svg=el(id),left=58,top=45,w=575,h=235,max=Math.max(1,...series.flatMap(s=>d.totals.map(t=>(t[s[0]]||0)*s[2]))),xmin=Math.min(...times),xmax=Math.max(...times),X=t=>left+(t-xmin)/(xmax-xmin||1)*w,Y=v=>top+h-v/max*h;node(svg,'text',{x:330,y:23,'text-anchor':'middle','font-size':16},ylabel);for(let i=0;i<=4;i++){let y=top+h-i*h/4;node(svg,'line',{x1:left,y1:y,x2:left+w,y2:y,stroke:'#e2e8f0'});node(svg,'text',{x:left-7,y:y+4,'text-anchor':'end','font-size':11},num(max*i/4,1))}times.forEach(t=>node(svg,'text',{x:X(t),y:top+h+20,'text-anchor':'middle','font-size':10,transform:`rotate(-40 ${X(t)} ${top+h+20})`},t));series.forEach((s,i)=>{let pts=d.totals.filter(t=>t[s[0]]!=null).map(t=>[X(t.tts),Y(t[s[0]]*s[2])]);node(svg,'polyline',{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:s[3],'stroke-width':2});pts.forEach(p=>node(svg,'circle',{cx:p[0],cy:p[1],r:3,fill:s[3]}));node(svg,'text',{x:65+i*145,y:350,fill:s[3],'font-size':11},s[1])});}
lineChart('rates',[['accepted_rate','接受率',100,'#64748b'],['ball_hit_rate','碰拍率',100,'#0284c7'],['target_pass_rate','跟踪达标率',100,'#059669'],['legal_return_rate','合法回球率',100,'#d97706']],'各 TTS 成功率 (%)');lineChart('errors',[['position_error_m_mean','拍心–target',100,'#059669'],['racket_ball_distance_at_deadline_m_mean','TTS 拍球距离',100,'#0284c7'],['racket_ball_distance_at_contact_m_mean','碰拍时刻距离',100,'#9333ea']],'平均距离 (cm)');
el('metadata').textContent=JSON.stringify({config:d.config,statistics:d.statistics,metadata:d.metadata,suite_sha256:d.sha256},null,2);['tts','metric','stat'].forEach(id=>el(id).onchange=drawGrid);drawGrid();
</script></html>"""
