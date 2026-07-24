"""
Agent 6: Dashboard Generator — builds a single self-contained HTML dashboard
(Plotly via CDN, no build step) from the outputs of the other agents.

This module was referenced by app.py (`from agents import dashboard_agent`)
but did not exist — the dashboard route silently fell back to
"<p>No dashboard</p>" for every analysis. This is the real implementation.

Inputs are the same dicts already flowing through app.py's run_pipeline:
  det  — safe_ctx'd detective output (DataFrame already stripped out)
  ana  — analyst output (key_insights, ml_recommendation, warning_flags, ...)
  ml   — raw ml_engineer / clustering / forecasting output (may contain
         private "_"-prefixed model objects, which are filtered out here)
  rep  — reporter output (report, headline, risk_score, ...)
  industry — optional industry_intelligence output dict
"""
import json
import numpy as np
import pandas as pd


def _safe(v):
    """Recursively strip anything non-JSON-serializable (mirrors app.py's _safe_val,
    duplicated here so this module has no hard dependency on app.py internals)."""
    if isinstance(v, pd.DataFrame):
        return "__DATAFRAME__"
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.str_,)):
        return str(v)
    if isinstance(v, dict):
        return {k: _safe(vv) for k, vv in v.items() if not str(k).startswith("_")}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


def _kpi_cards(det: dict, ml: dict, industry: dict) -> list:
    shape = det.get("original_shape", [0, 0])
    qd = det.get("quality_dimensions", {})
    cards = [
        {"label": "Rows", "value": f"{shape[0]:,}"},
        {"label": "Columns", "value": str(shape[1])},
        {"label": "Data Quality", "value": f"{qd.get('composite_score', det.get('quality_score', 0))}%",
         "sub": qd.get("grade", "")},
    ]
    task_type = ml.get("task_type", "")
    if task_type in ("classification", "regression"):
        score = ml.get("best_score", 0)
        score_disp = f"{round(score*100,1)}%" if task_type == "classification" else f"R²={round(score,3)}"
        cards.append({"label": "Best Model", "value": ml.get("best_model", "N/A"), "sub": score_disp})
    elif task_type == "clustering":
        cards.append({"label": "Segments Found", "value": str(ml.get("n_clusters", 0)),
                       "sub": f"silhouette {ml.get('best_score', 0)}"})
    elif task_type == "forecasting":
        cards.append({"label": "Forecast Model", "value": ml.get("best_model", "N/A"),
                       "sub": f"MAE {ml.get('best_score', 0)}"})
    if industry and industry.get("available"):
        cards.append({"label": "Industry Mode", "value": industry.get("industry_label", "")})
    return cards


def _eda_charts(det: dict) -> dict:
    charts = {}
    dist = det.get("distribution_data", [])
    charts["histograms"] = [
        {"column": d["column"], "edges": d["edges"], "counts": d["histogram"]}
        for d in dist[:6]
    ]
    corr = det.get("correlation", {})
    if corr:
        labels = list(corr.keys())
        matrix = [[corr[r].get(c, 0) for c in labels] for r in labels]
        charts["correlation"] = {"labels": labels, "matrix": matrix}
    column_stats = det.get("column_stats", {})
    cat_cols = [c for c in det.get("categorical_cols", []) if column_stats.get(c, {}).get("type") == "categorical"]
    charts["categorical"] = []
    for c in cat_cols[:4]:
        s = column_stats.get(c, {})
        charts["categorical"].append({
            "column": c, "top": s.get("top", ""), "unique": s.get("unique", 0),
            "top_values": s.get("top_values", []),
        })
    return charts


def run(det: dict, ana: dict, ml: dict, rep: dict, query: str = "", industry: dict = None) -> dict:
    industry = industry or {}
    det_s = _safe(det)
    ana_s = _safe(ana)
    ml_s  = _safe(ml)
    rep_s = _safe(rep)
    ind_s = _safe(industry)

    payload = {
        "meta": {
            "filename": det_s.get("filename", "dataset.csv"),
            "generated": query,
            "task_type": ml_s.get("task_type", ""),
            "target_column": ml_s.get("target_column", ""),
        },
        "kpi_cards": _kpi_cards(det_s, ml_s, ind_s),
        "quality": det_s.get("quality_dimensions", {}),
        "eda": _eda_charts(det_s),
        "models": ml_s.get("models", []),
        "task_type": ml_s.get("task_type", ""),
        "feature_importances": ml_s.get("feature_importances", []),
        "shap_plots": ml_s.get("shap_plots", {"available": False}),
        "clustering": {
            "pca_projection": ml_s.get("pca_projection", []),
            "cluster_profiles": ml_s.get("cluster_profiles", []),
        } if ml_s.get("task_type") == "clustering" else None,
        "forecasting": {
            "historical_dates": ml_s.get("historical_dates", []),
            "historical_y": ml_s.get("historical_y", []),
            "future_dates": ml_s.get("future_dates", []),
            "future_y": ml_s.get("future_y", []),
            "future_lower": ml_s.get("future_lower", []),
            "future_upper": ml_s.get("future_upper", []),
        } if ml_s.get("task_type") == "forecasting" else None,
        "key_insights": ana_s.get("key_insights", []),
        "actions": ana_s.get("actions", []),
        "warning_flags": ana_s.get("warning_flags", []),
        "industry": ind_s,
        "report": {
            "headline": rep_s.get("headline", ""),
            "report": rep_s.get("report", ""),
            "should_i_worry": rep_s.get("should_i_worry", ""),
            "risk_score": rep_s.get("risk_score", 0),
            "next_steps": rep_s.get("next_steps", []),
        },
    }

    html = _build_html(payload)
    return {"html": html, "payload": payload}


def _build_html(d: dict) -> str:
    data_json = json.dumps(d, ensure_ascii=False)

    header = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>DataMind Dashboard — {d['meta'].get('filename','dataset')}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
  :root {{
    --bg: #0A0E16; --panel: rgba(255,255,255,0.045); --panel-border: rgba(255,255,255,0.09);
    --text: #ECEEF2; --muted: #9098AA; --primary: #F2A93B; --cyan: #4DD8E0; --success: #3DD68C;
    --warning: #F2A93B; --danger: #F2566B;
    --glow-a: rgba(242,169,59,0.14); --glow-b: rgba(77,216,224,0.11);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin:0; font-family:'Inter',-apple-system,"Segoe UI",Roboto,sans-serif; background: var(--bg); color: var(--text);
    position: relative;
  }}
  body::before {{
    content:''; position: fixed; inset:0; z-index:0; pointer-events:none;
    background: radial-gradient(circle at 10% 6%, var(--glow-a), transparent 38%),
                radial-gradient(circle at 90% 85%, var(--glow-b), transparent 42%);
  }}
  .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 60px; position: relative; z-index: 1; }}
  h1 {{ font-family:'Space Grotesk',sans-serif; font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}
  .grid {{ display: grid; gap: 16px; }}
  .kpi-grid {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-bottom: 24px; }}
  .mono {{ font-family:'JetBrains Mono',monospace; }}
  .card {{
    background: var(--panel); border: 1px solid var(--panel-border); border-radius: 16px;
    padding: 16px 18px; backdrop-filter: blur(18px) saturate(135%);
  }}
  .kpi-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
  .kpi-value {{ font-family:'JetBrains Mono',monospace; font-size: 23px; font-weight: 500; margin-top: 4px; }}
  .kpi-sub {{ color: var(--success); font-size: 12px; margin-top: 2px; }}
  .section {{ margin-top: 32px; }}
  .section h2 {{ font-family:'Space Grotesk',sans-serif; font-size: 16px; margin: 0 0 12px; color: var(--primary); }}
  .two-col {{ grid-template-columns: 1fr 1fr; }}
  .insight {{ padding: 12px 14px; border-radius: 10px; margin-bottom: 8px; font-size: 13px; }}
  .insight.positive {{ background: rgba(61,214,140,0.12); border-left: 3px solid var(--success); }}
  .insight.warning  {{ background: rgba(242,169,59,0.12); border-left: 3px solid var(--warning); }}
  .insight.info     {{ background: rgba(77,216,224,0.12); border-left: 3px solid var(--cyan); }}
  .caption {{ color: var(--muted); font-size: 12.5px; margin-top: 6px; line-height: 1.5; }}
  img.shap {{ max-width: 100%; border-radius: 10px; background: #fff; padding: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--panel-border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }}
  .badge {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge.best {{ background: var(--primary); color: #1A1106; }}
  .recs li {{ margin-bottom: 6px; font-size: 13px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{d['meta'].get('filename','dataset.csv')}</h1>
  <div class="sub">DataMind AI Dashboard &middot; task: {d['meta'].get('task_type','') or 'n/a'} &middot; target: {d['meta'].get('target_column','') or 'n/a'}</div>

  <div class="grid kpi-grid" id="kpi-cards"></div>

  <div class="section">
    <h2>Data Quality Center</h2>
    <div class="grid two-col">
      <div class="card"><div id="quality-gauge"></div></div>
      <div class="card"><div id="quality-radar"></div>
        <ul class="recs" id="quality-recs"></ul>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Exploratory Data Analysis</h2>
    <div class="grid two-col">
      <div class="card"><div id="hist-chart"></div></div>
      <div class="card"><div id="corr-chart"></div></div>
    </div>
    <div class="grid" id="categorical-grid" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin-top: 16px;"></div>
  </div>

  <div class="section" id="ml-section">
    <h2>Model Leaderboard</h2>
    <div class="card"><div id="leaderboard-chart"></div></div>
  </div>

  <div class="section" id="explain-section">
    <h2>Explainable AI</h2>
    <div class="grid two-col">
      <div class="card"><div id="fi-chart"></div></div>
      <div class="card" id="shap-card"></div>
    </div>
  </div>

  <div class="section" id="task-section"></div>

  <div class="section" id="industry-section"></div>

  <div class="section">
    <h2>Key Insights &amp; Actions</h2>
    <div class="grid two-col">
      <div class="card" id="insights-card"></div>
      <div class="card" id="actions-card"></div>
    </div>
  </div>

  <div class="section">
    <h2>Executive Take</h2>
    <div class="card" id="exec-card"></div>
  </div>
</div>

<script>
const DATA = {data_json};
</script>
"""

    body_script = """
<script>
const darkLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#ECEEF2', size: 11 }, margin: { t: 30, l: 50, r: 20, b: 40 },
};
const cfg = { displayModeBar: false, responsive: true };

// ── KPI cards ────────────────────────────────────────────────
const kpiWrap = document.getElementById('kpi-cards');
(DATA.kpi_cards || []).forEach(k => {
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = `<div class="kpi-label">${k.label}</div><div class="kpi-value">${k.value}</div>` +
                 (k.sub ? `<div class="kpi-sub">${k.sub}</div>` : '');
  kpiWrap.appendChild(el);
});

// ── Data Quality gauge + radar ──────────────────────────────────
const q = DATA.quality || {};
const composite = q.composite_score || 0;
Plotly.newPlot('quality-gauge', [{
  type: 'indicator', mode: 'gauge+number',
  value: composite,
  title: { text: `Composite Score (${q.grade || ''})`, font: { color: '#ECEEF2', size: 13 } },
  gauge: {
    axis: { range: [0, 100], tickcolor: '#9098AA' },
    bar: { color: '#F2A93B' },
    steps: [
      { range: [0, 60], color: 'rgba(240,68,68,0.25)' },
      { range: [60, 75], color: 'rgba(245,166,35,0.25)' },
      { range: [75, 90], color: 'rgba(0,200,150,0.20)' },
      { range: [90, 100], color: 'rgba(0,200,150,0.35)' },
    ],
  },
  number: { font: { color: '#ECEEF2' } },
}], { ...darkLayout, height: 220 }, cfg);

const radar = q.radar_chart || { labels: [], values: [] };
Plotly.newPlot('quality-radar', [{
  type: 'scatterpolar', r: [...radar.values, radar.values[0]], theta: [...radar.labels, radar.labels[0]],
  fill: 'toself', line: { color: '#F2A93B' }, fillcolor: 'rgba(242,169,59,0.22)',
}], { ...darkLayout, height: 220, polar: { radialaxis: { range: [0, 100], color: '#9098AA' }, bgcolor: 'rgba(0,0,0,0)' }, showlegend: false }, cfg);

const recsList = document.getElementById('quality-recs');
(q.recommendations || []).forEach(r => {
  const li = document.createElement('li');
  li.textContent = r;
  recsList.appendChild(li);
});

// ── EDA: histograms (small multiples as overlaid traces) + correlation heatmap ──
const hist = (DATA.eda || {}).histograms || [];
if (hist.length) {
  const traces = hist.map(h => ({
    x: h.edges.slice(0, -1).map((e, i) => (e + h.edges[i + 1]) / 2),
    y: h.counts, type: 'bar', name: h.column, opacity: 0.75,
  }));
  Plotly.newPlot('hist-chart', traces, { ...darkLayout, height: 280, barmode: 'overlay', title: { text: 'Distributions', font: { size: 12, color: '#ECEEF2' } } }, cfg);
} else {
  document.getElementById('hist-chart').innerHTML = '<p style="color:#9098AA;font-size:13px;">No numeric columns to chart.</p>';
}

const corr = (DATA.eda || {}).correlation;
if (corr && corr.labels && corr.labels.length > 1) {
  Plotly.newPlot('corr-chart', [{
    z: corr.matrix, x: corr.labels, y: corr.labels, type: 'heatmap',
    colorscale: 'RdBu', zmin: -1, zmax: 1, reversescale: true,
  }], { ...darkLayout, height: 280, title: { text: 'Correlation Heatmap', font: { size: 12, color: '#ECEEF2' } } }, cfg);
} else {
  document.getElementById('corr-chart').innerHTML = '<p style="color:#9098AA;font-size:13px;">Not enough numeric columns for a correlation matrix.</p>';
}

// ── Categorical breakdowns (bar chart per column, top 8 values) ─────
const catCols = (DATA.eda || {}).categorical || [];
const catGrid = document.getElementById('categorical-grid');
catCols.forEach((c, i) => {
  if (!c.top_values || !c.top_values.length) return;
  const card = document.createElement('div');
  card.className = 'card';
  const chartId = `cat-chart-${i}`;
  card.innerHTML = `<div id="${chartId}"></div>`;
  catGrid.appendChild(card);
  const sorted = [...c.top_values].sort((a, b) => b.count - a.count);
  Plotly.newPlot(chartId, [{
    x: sorted.map(v => v.count), y: sorted.map(v => v.value), type: 'bar', orientation: 'h',
    marker: { color: '#4DD8E0' },
  }], {
    ...darkLayout, height: Math.max(180, sorted.length * 30),
    title: { text: `${c.column} (${c.unique} unique)`, font: { size: 12, color: '#ECEEF2' } },
    yaxis: { autorange: 'reversed' },
  }, cfg);
});

// ── Model leaderboard ────────────────────────────────────────
const models = DATA.models || [];
if (models.length) {
  const sorted = [...models].sort((a, b) => (b.score || 0) - (a.score || 0));
  Plotly.newPlot('leaderboard-chart', [{
    x: sorted.map(m => m.score || 0), y: sorted.map(m => m.name), type: 'bar', orientation: 'h',
    marker: { color: sorted.map(m => m.is_best ? '#F2A93B' : 'rgba(242,169,59,0.32)') },
  }], { ...darkLayout, height: Math.max(220, sorted.length * 36), yaxis: { autorange: 'reversed' } }, cfg);
} else {
  document.getElementById('ml-section').style.display = 'none';
}

// ── Feature importance + SHAP ───────────────────────────────
const fi = DATA.feature_importances || [];
if (fi.length) {
  const top = [...fi].slice(0, 12).reverse();
  Plotly.newPlot('fi-chart', [{
    x: top.map(f => f.importance), y: top.map(f => f.feature), type: 'bar', orientation: 'h',
    marker: { color: '#3DD68C' },
  }], { ...darkLayout, height: Math.max(220, top.length * 26), title: { text: 'Feature Importance', font: { size: 12, color: '#ECEEF2' } } }, cfg);
} else {
  document.getElementById('fi-chart').innerHTML = '<p style="color:#9098AA;font-size:13px;">No feature importance available for this task type.</p>';
}

const shap = DATA.shap_plots || {};
const shapCard = document.getElementById('shap-card');
if (shap.available) {
  shapCard.innerHTML = `
    <p style="font-size:13px;font-weight:600;margin:0 0 6px;">Global: what drives predictions</p>
    <img class="shap" src="${shap.summary_plot}" />
    <p class="caption">${shap.summary_caption || ''}</p>
    <p style="font-size:13px;font-weight:600;margin:16px 0 6px;">Local: one prediction explained</p>
    <img class="shap" src="${shap.waterfall_plot}" />
    <p class="caption">${shap.waterfall_caption || ''}</p>
  `;
} else {
  shapCard.innerHTML = `<p style="color:#9098AA;font-size:13px;">SHAP explanations unavailable${shap.reason ? ': ' + shap.reason : '.'}</p>`;
}
if (!fi.length && !shap.available) {
  document.getElementById('explain-section').style.display = 'none';
}

// ── Task-specific panel: clustering or forecasting ──────────
const taskSection = document.getElementById('task-section');
if (DATA.task_type === 'clustering' && DATA.clustering) {
  taskSection.innerHTML = `<h2>Customer / Record Segments</h2>
    <div class="grid two-col">
      <div class="card"><div id="cluster-scatter"></div></div>
      <div class="card" id="cluster-profiles"></div>
    </div>`;
  const proj = DATA.clustering.pca_projection || [];
  const byCluster = {};
  proj.forEach(p => { (byCluster[p.cluster] = byCluster[p.cluster] || []).push(p); });
  const traces = Object.keys(byCluster).map(c => ({
    x: byCluster[c].map(p => p.x), y: byCluster[c].map(p => p.y),
    mode: 'markers', type: 'scatter', name: `Segment ${c}`, marker: { size: 6, opacity: 0.7 },
  }));
  Plotly.newPlot('cluster-scatter', traces, { ...darkLayout, height: 320, title: { text: 'PCA Projection (2D)', font: { size: 12, color: '#ECEEF2' } } }, cfg);
  const profiles = DATA.clustering.cluster_profiles || [];
  document.getElementById('cluster-profiles').innerHTML = profiles.map(p => `
    <div style="margin-bottom:14px;">
      <b>Segment ${p.cluster}</b> &middot; ${p.size} records (${p.pct}%)<br/>
      <span class="caption">Distinctive: ${(p.distinctive_features || []).slice(0,3).map(f => f.feature).join(', ')}</span>
    </div>`).join('');
} else if (DATA.task_type === 'forecasting' && DATA.forecasting) {
  taskSection.innerHTML = `<h2>Forecast</h2><div class="card"><div id="forecast-chart"></div></div>`;
  const f = DATA.forecasting;
  const histTrace = { x: f.historical_dates, y: f.historical_y, mode: 'lines', name: 'Historical', line: { color: '#9098AA' } };
  const futTrace  = { x: f.future_dates, y: f.future_y, mode: 'lines+markers', name: 'Forecast', line: { color: '#F2A93B' } };
  const upperTrace = { x: f.future_dates, y: f.future_upper, mode: 'lines', name: 'Upper 95%', line: { width: 0 }, showlegend: false };
  const lowerTrace = { x: f.future_dates, y: f.future_lower, mode: 'lines', name: 'Confidence band', fill: 'tonexty', fillcolor: 'rgba(124,111,247,0.18)', line: { width: 0 } };
  Plotly.newPlot('forecast-chart', [histTrace, upperTrace, lowerTrace, futTrace], { ...darkLayout, height: 340 }, cfg);
} else {
  taskSection.style.display = 'none';
}

// ── Industry KPIs ─────────────────────────────────────────────
const ind = DATA.industry || {};
const indSection = document.getElementById('industry-section');
if (ind.available) {
  indSection.innerHTML = `<h2>${ind.industry_label} Intelligence</h2>
    <div class="card">
      <table>
        <tr><th>KPI</th><th>Matched Column</th><th>Average</th></tr>
        ${(ind.matched_kpis || []).map(k => `<tr><td>${k.kpi}</td><td>${k.matched_column}</td><td>${k.mean != null ? Number(k.mean).toFixed(2) : '—'}</td></tr>`).join('')}
      </table>
      <ul class="recs" style="margin-top:14px;">
        ${(ind.recommendations || []).map(r => `<li>${r}</li>`).join('')}
      </ul>
    </div>`;
} else {
  indSection.style.display = 'none';
}

// ── Key insights + actions ────────────────────────────────────
const insightsCard = document.getElementById('insights-card');
insightsCard.innerHTML = (DATA.key_insights || []).map(i =>
  `<div class="insight ${i.type || 'info'}"><b>${i.title || ''}</b><br/>${i.description || ''}</div>`
).join('') || '<p style="color:#9098AA;font-size:13px;">No insights generated.</p>';

const actionsCard = document.getElementById('actions-card');
actionsCard.innerHTML = '<ol>' + (DATA.actions || []).map(a =>
  `<li style="margin-bottom:8px;font-size:13px;"><b>${a.action || ''}</b><div class="caption">${a.expected_outcome || ''}</div></li>`
).join('') + '</ol>' || '<p style="color:#9098AA;font-size:13px;">No actions generated.</p>';

// ── Executive take ────────────────────────────────────────────
const rep = DATA.report || {};
document.getElementById('exec-card').innerHTML = `
  <p style="font-size:15px;font-weight:700;margin:0 0 8px;">${rep.headline || ''}</p>
  <p style="font-size:13px;line-height:1.6;color:#CBD5E1;">${(rep.report || '').replace(/\\n/g, '<br/>')}</p>
  <p style="margin-top:12px;"><b>Should I worry?</b> ${rep.should_i_worry || ''}</p>
  <div style="margin-top:8px;"><span class="badge" style="background:${(rep.risk_score||0) > 50 ? '#F2566B' : '#3DD68C'};">Risk score: ${rep.risk_score != null ? rep.risk_score : 'n/a'}</span></div>
`;
</script>
</body>
</html>
"""

    return header + body_script
