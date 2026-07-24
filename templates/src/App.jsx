import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from "recharts";

// ─── CONFIG ──────────────────────────────────────────────────────
// Use the same machine hostname as the frontend, but talk to Flask on port 5000.
const API_BASE = `${window.location.protocol}//${window.location.hostname}:5000`;

const CHART_COLORS = ["#D98E2B","#2BA8AF","#4FB88A","#E0566F","#9B6FD9","#4FA8C9","#D9772E","#7AA85C"];

// ─── CSS VARIABLES & GLOBAL STYLES ───────────────────────────────
// Design direction: glassmorphism isn't decoration here — it's the same idea
// as the product's core feature (SHAP explainability: seeing through a model's
// decision instead of treating it as an opaque black box). Frosted, layered
// panels you can see depth through is the same metaphor in CSS. Palette is an
// amber/cyan duotone (deliberately not the generic violet-on-black every AI
// tool clone reaches for) — amber reads as "instrument/measurement" (calibration
// dials, oscilloscope phosphor), cyan as "data." The one signature move is the
// rotating glass-refraction ring on whichever agent is currently running.
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
  @property --angle { syntax: '<angle>'; initial-value: 0deg; inherits: false; }

  :root {
    --bg:#F3F1EC; --bg2:rgba(255,255,255,0.66); --bg3:rgba(255,255,255,0.85);
    --border:rgba(20,18,12,0.08); --border2:rgba(20,18,12,0.14);
    --text:#1C1A14; --text2:#6B6555; --text3:#9B9686;
    --accent:#A85E08; --accent-bg:rgba(168,94,8,0.10); --accent-text:#ffffff;
    --success:#188252; --success-bg:rgba(24,130,82,0.10);
    --warning:#A85E08; --warning-bg:rgba(168,94,8,0.10);
    --danger:#BC2540; --danger-bg:rgba(188,37,64,0.10);
    --purple:#0B7077; --purple-bg:rgba(11,112,119,0.10);
    --shadow:0 8px 28px rgba(20,18,12,0.07),inset 0 1px 0 rgba(255,255,255,0.6);
    --glow-a:rgba(168,94,8,0.10); --glow-b:rgba(11,112,119,0.08);
    --r:16px;--rs:10px;--blur:18px;
  }
  [data-theme="dark"] {
    --bg:#0A0E16; --bg2:rgba(255,255,255,0.045); --bg3:rgba(255,255,255,0.075);
    --border:rgba(255,255,255,0.09); --border2:rgba(255,255,255,0.16);
    --text:#ECEEF2; --text2:#9098AA; --text3:#5C6478;
    --accent:#F2A93B; --accent-bg:rgba(242,169,59,0.13); --accent-text:#1A1106;
    --success:#3DD68C; --success-bg:rgba(61,214,140,0.13);
    --warning:#F2A93B; --warning-bg:rgba(242,169,59,0.13);
    --danger:#F2566B; --danger-bg:rgba(242,86,107,0.13);
    --purple:#4DD8E0; --purple-bg:rgba(77,216,224,0.13);
    --shadow:0 8px 32px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.045);
    --glow-a:rgba(242,169,59,0.16); --glow-b:rgba(77,216,224,0.13);
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  html,body{height:100%;}
  body{
    font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);
    font-size:14px;line-height:1.5;position:relative;
  }
  /* Ambient glow — fixed, subtle, never competes with content */
  body::before{
    content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
    background:
      radial-gradient(circle at 12% 8%, var(--glow-a), transparent 38%),
      radial-gradient(circle at 88% 82%, var(--glow-b), transparent 42%);
  }
  #root{position:relative;z-index:1;}
  h1,h2,h3,.font-display{font-family:'Space Grotesk',sans-serif;letter-spacing:-0.01em;}
  .mono{font-family:'JetBrains Mono',monospace;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
  @keyframes typing{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-4px)}}
  @keyframes rotateAngle{to{--angle:360deg}}
  .nav-item{cursor:pointer;transition:background .15s,color .15s;user-select:none;}
  .nav-item:hover{background:var(--bg3)!important;color:var(--text)!important;}
  .nav-item.active{background:var(--accent-bg)!important;color:var(--accent)!important;}
  .btn-primary{
    background:linear-gradient(135deg,var(--accent),color-mix(in srgb,var(--accent) 75%,#000 10%));
    color:var(--accent-text);border:none;padding:9px 18px;border-radius:var(--rs);cursor:pointer;
    font-size:13px;font-weight:600;transition:filter .15s,transform .1s;font-family:inherit;
    display:inline-flex;align-items:center;gap:8px;box-shadow:0 4px 18px var(--glow-a);
  }
  .btn-primary:hover{filter:brightness(1.08);}
  .btn-primary:active{transform:translateY(1px);}
  .btn-primary:disabled{opacity:.5;cursor:not-allowed;box-shadow:none;}
  .btn-ghost{
    background:var(--bg2);backdrop-filter:blur(var(--blur));color:var(--text2);
    border:1px solid var(--border2);padding:8px 16px;border-radius:var(--rs);cursor:pointer;
    font-size:13px;transition:background .15s,border-color .15s;font-family:inherit;
    display:inline-flex;align-items:center;gap:8px;
  }
  .btn-ghost:hover{background:var(--bg3);border-color:var(--accent);color:var(--text);}
  .btn-ghost:disabled{opacity:.5;cursor:not-allowed;}
  .input{
    background:var(--bg2);backdrop-filter:blur(var(--blur));border:1px solid var(--border2);
    border-radius:var(--rs);padding:9px 12px;font-size:13px;color:var(--text);outline:none;
    transition:border-color .15s,box-shadow .15s;font-family:inherit;width:100%;
  }
  .input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-bg);}
  .card{
    background:var(--bg2);backdrop-filter:blur(var(--blur)) saturate(135%);
    border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow);
  }
  .scrollbar::-webkit-scrollbar{width:4px;}
  .scrollbar::-webkit-scrollbar-track{background:transparent;}
  .scrollbar::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
  .upload-zone:hover,.upload-zone.drag{border-color:var(--accent)!important;background:var(--accent-bg)!important;}
  .theme-btn{padding:4px 11px;border-radius:16px;font-size:11px;font-weight:500;cursor:pointer;border:none;background:transparent;color:var(--text3);transition:all .15s;font-family:inherit;}
  .theme-btn.active{background:var(--bg3);color:var(--text);}
  .tab-btn{padding:6px 15px;border-radius:8px;border:none;cursor:pointer;font-size:12px;font-weight:500;background:transparent;color:var(--text3);transition:all .15s;font-family:inherit;}
  .tab-btn.active{background:var(--bg3);color:var(--accent);}
  textarea.input{resize:vertical;min-height:72px;line-height:1.5;}

  /* Signature: rotating glass-refraction ring on the agent currently running */
  .agent-glow{position:relative;border-radius:13px;padding:1.4px;
    background:conic-gradient(from var(--angle,0deg),var(--accent),var(--purple),var(--accent));
    animation:rotateAngle 2.6s linear infinite;box-shadow:0 0 22px var(--glow-a);
  }
  .agent-glow-inner{border-radius:11.6px;background:var(--bg2);backdrop-filter:blur(var(--blur));height:100%;}
`;

// ─── UTILS ────────────────────────────────────────────────────────
const apiFetch = (path, opts = {}, token = "") =>
  fetch(`${API_BASE}${path}`, {
    ...opts,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(opts.headers || {}),
    },
  });

const formatScore = (score, task) => {
  if (score == null) return "—";
  if (task === "classification") return `${(score * 100).toFixed(1)}%`;
  return `R²=${score.toFixed(4)}`;
};

const agentStatusStyle = (s) => ({
  waiting: { bg: "var(--bg)", color: "var(--text3)", border: ".5px solid var(--border)" },
  running: { bg: "var(--accent-bg)", color: "var(--accent)", border: "none" },
  completed: { bg: "var(--success-bg)", color: "var(--success)", border: "none" },
  failed: { bg: "var(--danger-bg)", color: "var(--danger)", border: "none" },
}[s] || { bg: "var(--bg3)", color: "var(--text3)", border: "none" });

// ─── ATOMS ───────────────────────────────────────────────────────
function Spinner({ size = 15 }) {
  return <div style={{ width: size, height: size, border: "2px solid var(--border2)", borderTopColor: "var(--accent)", borderRadius: "50%", flexShrink: 0, animation: "spin .8s linear infinite" }} />;
}

function Badge({ type = "blue", children, style = {} }) {
  const map = {
    blue: { bg: "var(--accent-bg)", color: "var(--accent)" },
    green: { bg: "var(--success-bg)", color: "var(--success)" },
    yellow: { bg: "var(--warning-bg)", color: "var(--warning)" },
    red: { bg: "var(--danger-bg)", color: "var(--danger)" },
    purple: { bg: "var(--purple-bg)", color: "var(--purple)" },
    gray: { bg: "var(--bg3)", color: "var(--text2)" },
  };
  const c = map[type] || map.blue;
  return (
    <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 20, fontWeight: 500, background: c.bg, color: c.color, ...style }}>
      {children}
    </span>
  );
}

function KpiCard({ label, value, sub, highlight }) {
  return (
    <div className="card" style={{ padding: "14px 16px", borderLeft: highlight ? "3px solid var(--accent)" : "none" }}>
      <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.04em" }}>{label}</div>
      <div className="mono" style={{ fontSize: 21, fontWeight: 500, letterSpacing: "-0.3px", lineHeight: 1, color: highlight ? "var(--accent)" : "var(--text)" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--text3)", marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

function Empty({ icon, title, sub, action, onAction }) {
  return (
    <div style={{ textAlign: "center", padding: "48px 20px", color: "var(--text3)" }}>
      <div style={{ fontSize: 38, marginBottom: 12 }}>{icon}</div>
      <div style={{ fontSize: 14, fontWeight: 500, color: "var(--text2)", marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 12, marginBottom: action ? 16 : 0 }}>{sub}</div>
      {action && <button className="btn-primary" onClick={onAction}>{action}</button>}
    </div>
  );
}

// ─── AGENT PIPELINE ───────────────────────────────────────────────
const AGENTS = [
  { id: "detective", label: "Data Detective",      desc: "Cleans, profiles & scores data quality", icon: "🔍", step: 1 },
  { id: "analyst",    label: "AI Analyst",          desc: "LLaMA 3.3 70B insights",                  icon: "🧠", step: 2 },
  { id: "ml",         label: "ML Engineer",         desc: "AutoML, clustering/forecasting, SHAP",    icon: "⚙️", step: 3 },
  { id: "reporter",   label: "Reporter",            desc: "Executive summary & risk score",          icon: "📝", step: 4 },
  { id: "dashboard",  label: "Dashboard & Industry",desc: "Builds visual dashboard + industry KPIs", icon: "📊", step: 5 },
];

function fmtElapsed(sec) {
  if (sec == null) return "";
  if (sec < 1) return `${Math.round(sec * 1000)}ms`;
  return `${sec.toFixed(1)}s`;
}

function AgentPipeline({ step, jobStatus, stepHistory = [] }) {
  // Live-ticking clock so the "Running..." row's elapsed time updates without waiting on the next poll
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (jobStatus !== "running") return;
    const t = setInterval(() => setNow(Date.now() / 1000), 250);
    return () => clearInterval(t);
  }, [jobStatus]);

  const byStep = {};
  stepHistory.forEach(h => { byStep[h.step] = h; });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {AGENTS.map(ag => {
        let s = "waiting";
        if (jobStatus === "done" || ag.step < step) s = "completed";
        else if (ag.step === step && jobStatus === "running") s = "running";
        else if (jobStatus === "error" && ag.step === step) s = "failed";
        const { bg, color, border } = agentStatusStyle(s);

        const hist = byStep[ag.step];
        let timeLabel = "";
        if (s === "completed" && hist?.duration != null) timeLabel = fmtElapsed(hist.duration);
        else if (s === "running" && hist?.started_at) timeLabel = fmtElapsed(now - hist.started_at);

        const row = (
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 13px", borderRadius: s === "running" ? "11.6px" : 13, background: s === "running" ? "transparent" : "var(--bg3)", border: s === "running" ? "none" : "1px solid var(--border)" }}>
            <div style={{ width: 32, height: 32, borderRadius: 9, background: bg, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, flexShrink: 0 }}>{ag.icon}</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 500 }}>{ag.label}</div>
              <div style={{ fontSize: 11, color: "var(--text3)" }}>{ag.desc}</div>
            </div>
            {timeLabel && (
              <span className="mono" style={{ fontSize: 10.5, color: "var(--text3)", flexShrink: 0 }}>{timeLabel}</span>
            )}
            <span style={{
              fontSize: 11, fontWeight: 500, padding: "3px 9px", borderRadius: 20, background: bg, color, border,
              animation: s === "running" ? "pulse 1.3s ease infinite" : "none",
            }}>
              {s === "running" ? "Running..." : s.charAt(0).toUpperCase() + s.slice(1)}
            </span>
          </div>
        );
        return s === "running" ? (
          <div key={ag.id} className="agent-glow">
            <div className="agent-glow-inner">{row}</div>
          </div>
        ) : (
          <div key={ag.id}>{row}</div>
        );
      })}
    </div>
  );
}

// ─── AUTH SCREEN ──────────────────────────────────────────────────
function AuthScreen({ onAuth }) {
  const [tab, setTab]       = useState("login");
  const [name, setName]     = useState("");
  const [email, setEmail]   = useState("");
  const [pw, setPw]         = useState("");
  const [err, setErr]       = useState("");
  const [busy, setBusy]     = useState(false);

  async function submit(e) {
    e.preventDefault(); setErr(""); setBusy(true);
    try {
      const body = tab === "register" ? { name, email, password: pw } : { email, password: pw };
      const r = await fetch(`${API_BASE}/api/${tab}`, {
        method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) { setErr(d.error || "Error"); return; }
      onAuth(d);
    } catch { setErr("Cannot reach backend — is Flask running at " + API_BASE + "?"); }
    finally { setBusy(false); }
  }

  async function guest() {
    setBusy(true); setErr("");
    try {
      const r = await fetch(`${API_BASE}/api/guest`, { method: "POST", credentials: "include" });
      const d = await r.json();
      if (r.ok) onAuth(d); else setErr(d.error);
    } catch { setErr("Cannot reach backend at " + API_BASE); }
    finally { setBusy(false); }
  }

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}>
      <div className="card" style={{ width: "100%", maxWidth: 380, padding: 28 }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 24 }}>
          <div style={{ width: 34, height: 34, background: "var(--accent)", borderRadius: 9, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-text)" strokeWidth="2"><path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/></svg>
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, letterSpacing: "-0.3px" }}>DataMind AI</div>
            <div style={{ fontSize: 10, color: "var(--text3)" }}>Analytics Platform</div>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 3, background: "var(--bg3)", borderRadius: 8, padding: 3, marginBottom: 20 }}>
          {["login", "register"].map(t => (
            <button key={t} onClick={() => setTab(t)} className={`tab-btn${tab === t ? " active" : ""}`} style={{ flex: 1 }}>
              {t === "login" ? "Sign In" : "Register"}
            </button>
          ))}
        </div>

        <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {tab === "register" && <input className="input" placeholder="Full name" value={name} onChange={e => setName(e.target.value)} required />}
          <input className="input" type="email" placeholder="Email address" value={email} onChange={e => setEmail(e.target.value)} required />
          <input className="input" type="password" placeholder="Password" value={pw} onChange={e => setPw(e.target.value)} required />
          {err && <div style={{ fontSize: 12, color: "var(--danger)", background: "var(--danger-bg)", padding: "8px 10px", borderRadius: 6 }}>{err}</div>}
          <button className="btn-primary" type="submit" disabled={busy} style={{ justifyContent: "center", marginTop: 4 }}>
            {busy ? <Spinner size={13} /> : null}
            {tab === "login" ? "Sign In" : "Create Account"}
          </button>
        </form>

        <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "14px 0" }}>
          <div style={{ flex: 1, height: ".5px", background: "var(--border2)" }} />
          <span style={{ fontSize: 11, color: "var(--text3)" }}>or</span>
          <div style={{ flex: 1, height: ".5px", background: "var(--border2)" }} />
        </div>

        <button className="btn-ghost" onClick={guest} disabled={busy} style={{ width: "100%", justifyContent: "center" }}>
          {busy ? <Spinner size={13} /> : null} Continue as Guest
        </button>
      </div>
    </div>
  );
}

// ─── SIDEBAR ─────────────────────────────────────────────────────
const NAV = [
  { id: "dashboard", icon: "🏠", label: "Dashboard" },
  { id: "upload",    icon: "⬆️",  label: "Upload" },
  { id: "insights",  icon: "💡",  label: "Insights" },
  { id: "charts",    icon: "📊",  label: "Charts" },
  { id: "ml",        icon: "🤖",  label: "ML Results" },
  { id: "predict",   icon: "🎯",  label: "Predictions" },
  { id: "chat",      icon: "💬",  label: "Chat" },
  { id: "report",    icon: "📄",  label: "Reports" },
  { id: "history",   icon: "📈",  label: "History" },
];

function Sidebar({ page, setPage, user, onLogout }) {
  return (
    <div className="card" style={{ width: 208, display: "flex", flexDirection: "column", flexShrink: 0, height: "100%", overflow: "hidden" }}>
      <div style={{ padding: "18px 16px 14px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 10px var(--accent)", flexShrink: 0 }} />
          <div>
            <div className="font-display" style={{ fontSize: 15, fontWeight: 600, letterSpacing: "-0.2px" }}>DataMind</div>
            <div style={{ fontSize: 9.5, color: "var(--text3)", letterSpacing: "0.04em" }}>AI ANALYTICS</div>
          </div>
        </div>
      </div>

      <div className="scrollbar" style={{ flex: 1, padding: 8, overflowY: "auto" }}>
        <div style={{ fontSize: 10, fontWeight: 500, color: "var(--text3)", letterSpacing: ".08em", textTransform: "uppercase", padding: "4px 8px 8px" }}>Navigation</div>
        {NAV.map(n => (
          <div key={n.id} className={`nav-item${page === n.id ? " active" : ""}`}
            onClick={() => setPage(n.id)}
            style={{ display: "flex", alignItems: "center", gap: 9, padding: "7.5px 9px", borderRadius: 9, fontSize: 12.5, color: "var(--text2)", marginBottom: 2 }}>
            <span style={{ fontSize: 14, width: 16 }}>{n.icon}</span>
            {n.label}
          </div>
        ))}
      </div>

      <div style={{ padding: 12, borderTop: "1px solid var(--border)" }}>
        <div onClick={onLogout} style={{ display: "flex", alignItems: "center", gap: 9, padding: "6px 8px", borderRadius: 9, cursor: "pointer" }}
          className="nav-item">
          <div style={{ width: 28, height: 28, borderRadius: "50%", background: "linear-gradient(135deg,var(--accent),var(--purple))", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 600, color: "var(--accent-text)", flexShrink: 0 }}>
            {(user?.name || "G")[0].toUpperCase()}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, fontWeight: 500, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{user?.name || "Guest"}</div>
            <div style={{ fontSize: 10, color: "var(--text3)" }}>Sign out</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── TOPBAR ───────────────────────────────────────────────────────
function Topbar({ page, theme, setTheme, jobResult }) {
  return (
    <div className="card" style={{ height: 54, display: "flex", alignItems: "center", gap: 12, padding: "0 18px", flexShrink: 0 }}>
      <span className="font-display" style={{ fontSize: 14.5, fontWeight: 600 }}>{NAV.find(n => n.id === page)?.label}</span>
      {jobResult && (
        <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 10px", background: "var(--success-bg)", borderRadius: 20, fontSize: 11 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--success)" }} />
          <span style={{ color: "var(--success)", fontWeight: 500 }}>{jobResult.filename || "Analysis ready"}</span>
        </div>
      )}
      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 3, background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: 20, padding: 3 }}>
        {["light", "dark"].map(t => (
          <button key={t} onClick={() => setTheme(t)} className={`theme-btn${theme === t ? " active" : ""}`}>
            {t === "light" ? "☀ light" : "🌙 dark"}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── UPLOAD PAGE ──────────────────────────────────────────────────
function UploadPage({ token, onJobDone, setJobMeta }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile]         = useState(null);
  const [apiKey, setApiKey]     = useState("");
  const [query, setQuery]       = useState("Give me key insights about this data.");
  const [target, setTarget]     = useState("");
  const [loading, setLoading]   = useState(false);
  const [err, setErr]           = useState("");
  const [step, setStep]         = useState(0);
  const [stepName, setStepName] = useState("");
  const [stepHistory, setStepHistory] = useState([]);
  const [jobSt, setJobSt]       = useState("idle");
  const [logs, setLogs]         = useState([]);
  const poll    = useRef(null);
  const fileRef = useRef(null);

  const startPoll = useCallback((id) => {
    poll.current = setInterval(async () => {
      try {
        const r = await apiFetch(`/api/status/${id}`, {}, token);
        const d = await r.json();
        setStep(d.step || 0); setStepName(d.step_name || ""); setLogs(d.logs || []);
        setStepHistory(d.step_history || []);
        if (d.status === "done") {
          clearInterval(poll.current); setJobSt("done"); setLoading(false);
          onJobDone(d.result, id, d.step_history || []);
        } else if (d.status === "error") {
          clearInterval(poll.current); setJobSt("error"); setLoading(false);
          setErr(d.error || "Pipeline failed");
        } else setJobSt("running");
      } catch { clearInterval(poll.current); setJobSt("error"); setLoading(false); setErr("Connection lost"); }
    }, 900);
  }, [token, onJobDone]);

  useEffect(() => () => clearInterval(poll.current), []);

  async function analyse(overrideFile, overrideOpts) {
    const f = overrideFile || file;
    if (!f) { setErr("Please select a CSV file"); return; }
    setErr(""); setLoading(true); setJobSt("running"); setLogs([]); setStep(1);
    const fd = new FormData();
    fd.append("file", f); fd.append("query", (overrideOpts && overrideOpts.query) || query);
    if (apiKey) fd.append("api_key", apiKey);  // optional — server falls back to its own GROQ_API_KEY
    const t = (overrideOpts && overrideOpts.target) ?? target;
    if (t) fd.append("target_column", t);
    if (overrideOpts && overrideOpts.industry) fd.append("industry", overrideOpts.industry);
    try {
      const r = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST", credentials: "include",
        headers: { Authorization: `Bearer ${token}` }, body: fd,
      });
      const d = await r.json();
      if (!r.ok) { setErr(d.error); setLoading(false); setJobSt("idle"); return; }
      setJobMeta({ jobId: d.job_id, apiKey });
      startPoll(d.job_id);
    } catch { setErr("Flask backend unreachable — is it running at " + API_BASE + "?"); setLoading(false); setJobSt("idle"); }
  }

  const [sampleLoading, setSampleLoading] = useState(false);
  async function trySample() {
    setSampleLoading(true); setErr("");
    try {
      const r = await fetch(`${API_BASE}/api/sample`, { credentials: "include" });
      if (!r.ok) throw new Error("Sample dataset unavailable");
      const blob = await r.blob();
      const sampleFile = new File([blob], "employee_attrition_sample.csv", { type: "text/csv" });
      setFile(sampleFile);
      setTarget("attrition");
      setQuery("What's driving employee attrition, and what should we do about it?");
      await analyse(sampleFile, {
        target: "attrition",
        industry: "hr",
        query: "What's driving employee attrition, and what should we do about it?",
      });
    } catch {
      setErr("Couldn't load the sample dataset — try uploading your own CSV instead.");
    } finally {
      setSampleLoading(false);
    }
  }

  const drop = (e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && (f.name.endsWith(".csv") || f.name.endsWith(".xlsx"))) setFile(f);
    else setErr("Only CSV / Excel files supported");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
      {/* Drop zone */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Upload Dataset</div>
        <div className={`upload-zone${dragging ? " drag" : ""}`}
          style={{ border: "1.5px dashed var(--border2)", borderRadius: 10, padding: "28px 20px", textAlign: "center", cursor: "pointer", transition: "all .2s", background: "var(--bg)" }}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={drop}
          onClick={() => fileRef.current.click()}>
          <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" hidden onChange={e => e.target.files[0] && setFile(e.target.files[0])} />
          <div style={{ width: 42, height: 42, borderRadius: 10, background: "var(--bg3)", border: ".5px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 10px", fontSize: 22 }}>📁</div>
          {file ? (
            <>
              <div style={{ fontSize: 13, fontWeight: 500, color: "var(--accent)" }}>{file.name}</div>
              <div style={{ fontSize: 11, color: "var(--text3)", marginTop: 4 }}>{(file.size / 1024).toFixed(1)} KB · Click to change</div>
            </>
          ) : (
            <>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Drop CSV/Excel here or click to browse</div>
              <div style={{ fontSize: 12, color: "var(--text3)" }}>Supports CSV & Excel · Up to 100MB</div>
            </>
          )}
          <div style={{ display: "flex", gap: 6, justifyContent: "center", marginTop: 12 }}>
            {["CSV", "Excel", "UTF-8", "100MB max"].map(b => (
              <span key={b} style={{ fontSize: 10, padding: "2px 8px", borderRadius: 20, fontWeight: 500, background: "var(--bg3)", color: "var(--text2)", border: ".5px solid var(--border)" }}>{b}</span>
            ))}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 12 }}>
          <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
          <span style={{ fontSize: 11, color: "var(--text3)" }}>or</span>
          <div style={{ flex: 1, height: 1, background: "var(--border)" }} />
        </div>
        <button
          onClick={trySample}
          disabled={sampleLoading || loading}
          style={{
            marginTop: 12, width: "100%", padding: "11px 14px", borderRadius: "var(--rs)",
            background: "var(--accent-bg)", border: "1px solid var(--accent)",
            color: "var(--accent)", fontSize: 12.5, fontWeight: 600, cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
          }}>
          {sampleLoading ? <Spinner size={13} /> : "✨"} {sampleLoading ? "Loading sample…" : "Try it instantly — sample HR attrition dataset, no upload needed"}
        </button>
      </div>

      {/* Config */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Configuration</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div>
            <label style={{ fontSize: 11, color: "var(--text3)", display: "block", marginBottom: 4 }}>Groq API Key <span style={{ fontStyle: "italic" }}>(optional — leave blank to use this server's configured key)</span></label>
            <input className="input" type="password" placeholder="gsk_... (optional)" value={apiKey} onChange={e => setApiKey(e.target.value)} />
          </div>
          <div>
            <label style={{ fontSize: 11, color: "var(--text3)", display: "block", marginBottom: 4 }}>Analysis Query</label>
            <textarea className="input" value={query} onChange={e => setQuery(e.target.value)} />
          </div>
          <div>
            <label style={{ fontSize: 11, color: "var(--text3)", display: "block", marginBottom: 4 }}>Target Column <span style={{ fontStyle: "italic" }}>(optional — auto-detected)</span></label>
            <input className="input" placeholder="e.g. price, churn, target..." value={target} onChange={e => setTarget(e.target.value)} />
          </div>
          {err && <div style={{ fontSize: 12, color: "var(--danger)", background: "var(--danger-bg)", padding: "8px 12px", borderRadius: 6 }}>{err}</div>}
          <button className="btn-primary" onClick={() => analyse()} disabled={loading} style={{ alignSelf: "flex-start" }}>
            {loading ? <Spinner size={13} /> : "🚀"} {loading ? "Analysing..." : "Run Analysis"}
          </button>
        </div>
      </div>

      {/* Pipeline */}
      {jobSt !== "idle" && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
            Agent Pipeline
            {jobSt === "running" && <Spinner size={13} />}
            {jobSt === "done"    && <Badge type="green">Complete</Badge>}
            {jobSt === "error"   && <Badge type="red">Failed</Badge>}
          </div>
          <AgentPipeline step={step} jobStatus={jobSt} stepHistory={stepHistory} />
          {stepName && (
            <div style={{ marginTop: 10, padding: "6px 10px", background: "var(--bg3)", borderRadius: 6, fontSize: 11, color: "var(--text2)" }}>
              {jobSt === "running" ? "⏳" : "✓"} {stepName}
            </div>
          )}
          {logs.length > 0 && (
            <div className="scrollbar" style={{ marginTop: 10, maxHeight: 120, overflowY: "auto", background: "#0f0f0e", borderRadius: 6, padding: "8px 10px", fontFamily: "DM Mono, monospace", fontSize: 10.5, color: "#9b9890", display: "flex", flexDirection: "column", gap: 2 }}>
              {logs.slice(-14).map((l, i) => <div key={i}>{l}</div>)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── INSIGHTS PAGE ────────────────────────────────────────────────
function InsightsPage({ jobResult, setPage }) {
  if (!jobResult) return <Empty icon="💡" title="No analysis yet" sub="Upload a CSV to generate AI insights" action="⬆ Upload Dataset" onAction={() => setPage("upload")} />;

  const insights  = jobResult.key_insights   || [];
  const actions   = jobResult.actions        || [];
  const stats     = jobResult.stats          || {};
  const ml        = jobResult.ml_recommendation || {};
  const warnings  = ml.warning_flags         || [];
  const metricCards = ml.metric_cards        || [];

  const typeStyle = (t) => ({
    positive: { border: "var(--success)", badge: "green" },
    warning:  { border: "var(--warning)", badge: "yellow" },
  }[t] || { border: "var(--accent)", badge: "blue" });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
        <KpiCard label="Total Rows"     value={(stats.total_rows || 0).toLocaleString()} />
        <KpiCard label="Columns"        value={stats.total_cols || 0} />
        <KpiCard label="Data Quality"   value={`${stats.quality_score || 0}%`} sub={stats.quality_score > 80 ? "Good ↑" : "Needs improvement ↓"} />
        <KpiCard label="Insights Found" value={stats.insights_count || insights.length} />
      </div>

      {/* Summary */}
      {jobResult.summary && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Executive Summary</div>
          <div style={{ fontSize: 13, color: "var(--text2)", lineHeight: 1.65 }}>{jobResult.summary}</div>
        </div>
      )}

      {/* Metric cards from analyst */}
      {metricCards.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
          {metricCards.map((m, i) => (
            <div key={i} className="card" style={{ padding: "12px 14px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                <div style={{ fontSize: 11, color: "var(--text3)" }}>{m.label}</div>
                <span style={{ fontSize: 10 }}>{m.trend === "up" ? "↑" : m.trend === "down" ? "↓" : "→"}</span>
              </div>
              <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: "-0.4px", marginBottom: 4 }}>{m.value}</div>
              <div style={{ fontSize: 11, color: "var(--text3)" }}>{m.meaning}</div>
            </div>
          ))}
        </div>
      )}

      {/* Key insights */}
      {insights.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Key Insights</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 10 }}>
            {insights.map((ins, i) => {
              const { border, badge } = typeStyle(ins.type);
              return (
                <div key={i} style={{ padding: "12px 14px", background: "var(--bg3)", borderRadius: 8, borderLeft: `3px solid ${border}` }}>
                  <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8, marginBottom: 6 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, lineHeight: 1.4 }}>{ins.title}</div>
                    <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
                      <Badge type={badge}>{ins.type || "info"}</Badge>
                      {ins.impact && <Badge type={ins.impact === "high" ? "red" : ins.impact === "medium" ? "yellow" : "blue"}>{ins.impact}</Badge>}
                    </div>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text2)", lineHeight: 1.55 }}>{ins.description}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Actions */}
      {actions.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Recommended Actions</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {actions.map((a, i) => (
              <div key={i} style={{ display: "flex", gap: 12, padding: "10px 12px", background: "var(--bg3)", borderRadius: 8 }}>
                <div style={{ width: 24, height: 24, borderRadius: "50%", background: "var(--accent-bg)", color: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 600, flexShrink: 0 }}>
                  {a.priority || i + 1}
                </div>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 500 }}>{a.action}</div>
                  {a.expected_outcome && <div style={{ fontSize: 11, color: "var(--text3)", marginTop: 3 }}>→ {a.expected_outcome}</div>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Query suggestions */}
      {ml.query_suggestions?.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Try asking...</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {ml.query_suggestions.map((q, i) => (
              <span key={i} style={{ fontSize: 12, padding: "5px 12px", background: "var(--accent-bg)", color: "var(--accent)", borderRadius: 20, cursor: "default" }}>{q}</span>
            ))}
          </div>
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>⚠️ Warning Flags</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {warnings.map((w, i) => (
              <div key={i} style={{
                padding: "8px 12px", borderRadius: 6, fontSize: 12,
                background: w.severity === "critical" ? "var(--danger-bg)" : "var(--warning-bg)",
                color: w.severity === "critical" ? "var(--danger)" : "var(--warning)",
              }}>
                <span style={{ fontWeight: 500 }}>{w.severity?.toUpperCase()}</span>: {w.message}
                {w.column && <span style={{ opacity: .7, marginLeft: 8 }}>[{w.column}]</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── CHARTS PAGE ─────────────────────────────────────────────────
function ChartsPage({ jobResult, setPage }) {
  const [tab, setTab] = useState("dist");
  if (!jobResult) return <Empty icon="📊" title="No charts yet" sub="Run an analysis first" action="⬆ Upload" onAction={() => setPage("upload")} />;

  const ml   = jobResult.ml_recommendation || {};
  const dist = jobResult.distribution_data || [];
  const fi   = jobResult.feature_importances || ml.feature_importances || [];
  const mods = ml.models || [];
  const fp   = jobResult.future_prediction  || ml.future_prediction   || {};

  const tickStyle = { fontSize: 10, fill: "var(--text3)" };
  const ttStyle   = { background: "var(--bg2)", border: ".5px solid var(--border)", borderRadius: 6, fontSize: 11 };
  const TABS = [
    { id: "dist",    label: "Distributions" },
    { id: "models",  label: "Model Comparison" },
    { id: "feat",    label: "Feature Importance" },
    { id: "trend",   label: "Future Trend" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
      <div style={{ display: "flex", gap: 3, background: "var(--bg3)", borderRadius: 8, padding: 3, alignSelf: "flex-start" }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} className={`tab-btn${tab === t.id ? " active" : ""}`}>{t.label}</button>
        ))}
      </div>

      {/* Distributions */}
      {tab === "dist" && (
        dist.length === 0
          ? <Empty icon="📈" title="No distribution data" sub="Distribution data was not captured during analysis" />
          : <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 12 }}>
              {dist.slice(0, 6).map((d, i) => {
                const data = d.histogram
                  ? d.histogram.map((v, j) => ({ x: (d.edges?.[j] ?? j).toFixed(1), v }))
                  : (d.values || []).slice(0, 25).map((v, j) => ({ x: j, v }));
                return (
                  <div key={i} className="card" style={{ padding: 16 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
                      {d.column}
                      <Badge type="blue" style={{ fontSize: 9 }}>numeric</Badge>
                    </div>
                    <ResponsiveContainer width="100%" height={140}>
                      <BarChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="x" tick={tickStyle} tickLine={false} />
                        <YAxis tick={tickStyle} tickLine={false} />
                        <Tooltip contentStyle={ttStyle} />
                        <Bar dataKey="v" name="Count" fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                );
              })}
            </div>
      )}

      {/* Model comparison */}
      {tab === "models" && (
        mods.length === 0
          ? <Empty icon="🤖" title="No model data" sub="ML models were not trained" />
          : <div className="card" style={{ padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 6 }}>Model Score Comparison</div>
              <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 16 }}>Task: {ml.task_type} · Target: {ml.target_column}</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={mods.map(m => ({ name: m.name.replace(" Regressor","").replace(" Classifier",""), score: parseFloat(((m.score||0)*100).toFixed(1)), best: m.is_best }))}
                  margin={{ top: 0, right: 20, bottom: 30, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="name" tick={{ ...tickStyle, fontSize: 10 }} tickLine={false} angle={-15} textAnchor="end" />
                  <YAxis tick={tickStyle} tickLine={false} unit="%" />
                  <Tooltip contentStyle={ttStyle} formatter={v => [`${v}%`, "Score"]} />
                  <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                    {mods.map((m, i) => <Cell key={i} fill={m.is_best ? "var(--accent)" : "var(--border2)"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
      )}

      {/* Feature importance */}
      {tab === "feat" && (
        fi.length === 0
          ? <Empty icon="⭐" title="No feature data" sub="Feature importances not available" />
          : <div className="card" style={{ padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 16 }}>Top Feature Importances</div>
              <ResponsiveContainer width="100%" height={Math.max(200, Math.min(fi.length, 12) * 32)}>
                <BarChart layout="vertical" data={fi.slice(0, 12).map(f => ({ name: (f.feature || "").substring(0, 22), val: parseFloat((f.importance||0).toFixed(4)) }))}
                  margin={{ top: 0, right: 20, bottom: 0, left: 90 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis type="number" tick={tickStyle} tickLine={false} />
                  <YAxis type="category" dataKey="name" tick={tickStyle} tickLine={false} width={90} />
                  <Tooltip contentStyle={ttStyle} />
                  <Bar dataKey="val" name="Importance" fill="var(--purple)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
      )}

      {/* Future trend */}
      {tab === "trend" && (
        !fp?.future_y?.length
          ? <Empty icon="🔮" title="No forecast data" sub="Future prediction requires time-series data" />
          : <div className="card" style={{ padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Future Trend — {fp.target}</div>
              <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 16 }}>Trend R²: {fp.trend_r2} · Std Error: ±{fp.std_error} · {fp.future_steps} steps ahead</div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={fp.future_y.map((y, i) => ({
                  step: fp.future_x?.[i] ?? i,
                  predicted: parseFloat(y.toFixed(4)),
                  lower: parseFloat((fp.future_lower?.[i] ?? y).toFixed(4)),
                  upper: parseFloat((fp.future_upper?.[i] ?? y).toFixed(4)),
                }))} margin={{ top: 0, right: 20, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="step" tick={tickStyle} />
                  <YAxis tick={tickStyle} />
                  <Tooltip contentStyle={ttStyle} />
                  <Line type="monotone" dataKey="predicted" stroke="var(--accent)"  strokeWidth={2} dot={false} name="Predicted" />
                  <Line type="monotone" dataKey="upper"     stroke="var(--success)" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Upper 95%" />
                  <Line type="monotone" dataKey="lower"     stroke="var(--warning)" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Lower 95%" />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
      )}
    </div>
  );
}

// ─── ML RESULTS PAGE ─────────────────────────────────────────────
function MLResultsPage({ jobResult, setPage }) {
  if (!jobResult) return <Empty icon="🤖" title="No ML results" sub="Run an analysis to train models" action="⬆ Upload" onAction={() => setPage("upload")} />;

  const ml   = jobResult.ml_recommendation || {};
  const mods = ml.models          || [];
  const bm   = ml.best_metrics    || {};
  const fi   = jobResult.feature_importances || ml.feature_importances || [];
  const cm   = ml.confusion_matrix;
  const cml  = ml.confusion_labels || [];
  const isCls = ml.task_type === "classification";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
      {/* Best model banner */}
      {ml.best_model && (
        <div className="card" style={{ padding: 16, borderLeft: "3px solid var(--accent)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div>
              <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 4 }}>Best Model</div>
              <div style={{ fontSize: 22, fontWeight: 600, letterSpacing: "-0.5px" }}>{ml.best_model}</div>
              <div style={{ fontSize: 12, color: "var(--text3)", marginTop: 5 }}>
                Task: <span style={{ color: "var(--text2)" }}>{ml.task_type}</span>&nbsp;·&nbsp;
                Target: <span style={{ color: "var(--text2)" }}>{ml.target_column}</span>
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 30, fontWeight: 700, color: "var(--accent)", letterSpacing: "-1px" }}>{formatScore(ml.best_score, ml.task_type)}</div>
              <div style={{ fontSize: 11, color: "var(--text3)" }}>Score</div>
            </div>
          </div>
        </div>
      )}

      {/* Metric cards */}
      {Object.keys(bm).length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
          {isCls
            ? [["Accuracy","accuracy",true],["F1 Score","f1",true],["Precision","precision",true],["Recall","recall",true]].map(([l,k,pct]) => (
                <KpiCard key={k} label={l} value={bm[k] != null ? (pct ? `${(bm[k]*100).toFixed(1)}%` : bm[k].toFixed(4)) : "—"} />
              ))
            : [["R² Score","r2",false],["MAE","mae",false],["RMSE","rmse",false],["CV Score","cv_score",true]].map(([l,k,pct]) => (
                <KpiCard key={k} label={l} value={bm[k] != null ? (pct ? `${(bm[k]*100).toFixed(1)}%` : bm[k].toFixed(4)) : "—"} />
              ))
          }
        </div>
      )}

      {/* Models table */}
      {mods.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>All Models</div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: ".5px solid var(--border)" }}>
                {["Model", "Score", ""].map(h => (
                  <th key={h} style={{ textAlign: h === "Model" ? "left" : "center", padding: "6px 10px", fontSize: 11, color: "var(--text3)", fontWeight: 500 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {mods.map((m, i) => (
                <tr key={i} style={{ borderBottom: ".5px solid var(--border)", background: m.is_best ? "var(--accent-bg)" : "transparent" }}>
                  <td style={{ padding: "9px 10px", fontWeight: m.is_best ? 500 : 400 }}>{m.name}</td>
                  <td style={{ padding: "9px 10px", textAlign: "center", fontWeight: 600, color: m.is_best ? "var(--accent)" : "var(--text)" }}>{formatScore(m.score, ml.task_type)}</td>
                  <td style={{ padding: "9px 10px", textAlign: "center" }}>{m.is_best ? <Badge type="blue">★ Best</Badge> : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Confusion matrix */}
      {cm && isCls && cml.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Confusion Matrix</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ padding: "5px 10px", color: "var(--text3)", fontWeight: 500 }}>Actual \ Pred</th>
                  {cml.map(l => <th key={l} style={{ padding: "5px 10px", textAlign: "center", fontWeight: 500 }}>{l}</th>)}
                </tr>
              </thead>
              <tbody>
                {cm.map((row, ri) => (
                  <tr key={ri}>
                    <td style={{ padding: "5px 10px", fontWeight: 500 }}>{cml[ri]}</td>
                    {row.map((v, ci) => (
                      <td key={ci} style={{ padding: "7px 14px", textAlign: "center", borderRadius: 4, fontWeight: ri === ci ? 700 : 400,
                        background: ri === ci ? "var(--success-bg)" : "var(--bg3)",
                        color: ri === ci ? "var(--success)" : "var(--text)" }}>{v}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Feature importance bars */}
      {fi.length > 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Feature Importances</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {fi.slice(0, 12).map((f, i) => {
              const pct = Math.min(100, Math.round((f.importance / (fi[0]?.importance || 1)) * 100));
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ fontSize: 11, color: "var(--text2)", width: 130, flexShrink: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{f.feature}</div>
                  <div style={{ flex: 1, background: "var(--bg3)", borderRadius: 99, height: 6, overflow: "hidden" }}>
                    <div style={{ width: `${pct}%`, height: "100%", background: "var(--accent)", borderRadius: 99, transition: "width .8s ease" }} />
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text3)", width: 52, textAlign: "right", fontFamily: "DM Mono, monospace" }}>{f.importance?.toFixed(4)}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── PREDICT PAGE ─────────────────────────────────────────────────
function PredictPage({ token, jobId, setPage }) {
  const [feats, setFeats]     = useState([]);
  const [inputs, setInputs]   = useState({});
  const [taskType, setTask]   = useState("");
  const [target, setTarget]   = useState("");
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr]         = useState("");
  const [ready, setReady]     = useState(false);

  useEffect(() => {
    if (!jobId) return;
    apiFetch(`/api/predict_info/${jobId}`, {}, token)
      .then(r => r.json()).then(d => {
        if (d.features) {
          setFeats(d.features); setTask(d.task_type); setTarget(d.target);
          const def = {}; d.features.forEach(f => { def[f.name] = f.mean?.toFixed(3) ?? "0"; });
          setInputs(def); setReady(true);
        }
      }).catch(() => setErr("Could not load predict_info — job may not have a trained model"));
  }, [jobId, token]);

  async function predict() {
    setLoading(true); setErr(""); setResult(null);
    try {
      const nums = {}; Object.keys(inputs).forEach(k => { nums[k] = parseFloat(inputs[k]) || 0; });
      const r = await apiFetch(`/api/predict/${jobId}`, { method: "POST", body: JSON.stringify({ inputs: nums }) }, token);
      const d = await r.json();
      if (!r.ok) { setErr(d.error); return; }
      setResult(d);
    } catch { setErr("Prediction request failed"); }
    finally { setLoading(false); }
  }

  if (!jobId) return <Empty icon="🎯" title="No model trained" sub="Upload and analyse a dataset first" action="⬆ Upload" onAction={() => setPage("upload")} />;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Make a Prediction</div>
        {target && <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 16 }}>Predicting: <span style={{ color: "var(--accent)", fontWeight: 500 }}>{target}</span> · Task: {taskType}</div>}
        {!ready && !err && <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text3)", fontSize: 12, padding: "12px 0" }}><Spinner />Loading feature inputs...</div>}
        {feats.length > 0 && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 10, marginBottom: 14 }}>
            {feats.slice(0, 12).map(f => (
              <div key={f.name}>
                <label style={{ fontSize: 11, color: "var(--text3)", display: "block", marginBottom: 3 }}>
                  {f.name} <span style={{ fontSize: 10, opacity: .7 }}>({f.min?.toFixed(1)} – {f.max?.toFixed(1)})</span>
                </label>
                <input className="input" type="number" step="any"
                  value={inputs[f.name] ?? ""}
                  onChange={e => setInputs(p => ({ ...p, [f.name]: e.target.value }))}
                  placeholder={f.mean?.toFixed(3)} />
              </div>
            ))}
          </div>
        )}
        {err && <div style={{ fontSize: 12, color: "var(--danger)", background: "var(--danger-bg)", padding: "8px 12px", borderRadius: 6, marginBottom: 10 }}>{err}</div>}
        {ready && (
          <button className="btn-primary" onClick={predict} disabled={loading}>
            {loading ? <Spinner size={13} /> : "🎯"} {loading ? "Predicting..." : "Run Prediction"}
          </button>
        )}
      </div>

      {result && (
        <div className="card" style={{ padding: 16, borderLeft: "3px solid var(--accent)", animation: "fadeIn .3s ease" }}>
          <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 4 }}>Prediction Result</div>
          <div style={{ fontSize: 32, fontWeight: 700, color: "var(--accent)", letterSpacing: "-1px", marginBottom: 10 }}>{String(result.prediction)}</div>
          {result.confidence != null && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <div style={{ fontSize: 12, color: "var(--text2)", flexShrink: 0 }}>Confidence</div>
              <div style={{ flex: 1, background: "var(--bg3)", borderRadius: 99, height: 7, overflow: "hidden" }}>
                <div style={{ width: `${result.confidence}%`, height: "100%", borderRadius: 99, background: result.confidence > 80 ? "var(--success)" : result.confidence > 60 ? "var(--warning)" : "var(--danger)" }} />
              </div>
              <div style={{ fontSize: 12, fontWeight: 600, flexShrink: 0 }}>{result.confidence}%</div>
            </div>
          )}
          <div style={{ fontSize: 11, color: "var(--text3)" }}>
            Task: {result.task_type} · Features used: {result.features_used}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── CHAT PAGE ────────────────────────────────────────────────────
function ChatPage({ token, jobId, savedApiKey }) {
  const [msgs, setMsgs]       = useState([{ role: "assistant", content: "Hi! I'm DataMind AI. Upload a dataset then ask me anything about your data." }]);
  const [input, setInput]     = useState("");
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey]   = useState(savedApiKey || "");
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  async function send() {
    if (!input.trim() || loading) return;
    const txt = input.trim(); setInput("");
    setMsgs(p => [...p, { role: "user", content: txt }]);
    setLoading(true);
    try {
      const r = await apiFetch("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: txt, job_id: jobId || "", api_key: apiKey }),
      }, token);
      const d = await r.json();
      setMsgs(p => [...p, { role: "assistant", content: d.reply || d.error || "Error — please try again." }]);
    } catch {
      setMsgs(p => [...p, { role: "assistant", content: "Connection error — is Flask running?" }]);
    } finally { setLoading(false); }
  }

  const Bubble = ({ m }) => (
    <div style={{ display: "flex", gap: 10, flexDirection: m.role === "user" ? "row-reverse" : "row", animation: "fadeIn .2s ease" }}>
      <div style={{
        width: 30, height: 30, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
        background: m.role === "user" ? "linear-gradient(135deg,var(--accent),var(--purple))" : "var(--bg3)",
        color: m.role === "user" ? "var(--accent-text)" : "var(--text2)",
      }}>
        {m.role === "user" ? (token ? "U" : "G") : "🤖"}
      </div>
      <div style={{
        maxWidth: "72%", padding: "9px 13px", borderRadius: 10, fontSize: 13, lineHeight: 1.55,
        background: m.role === "user" ? "var(--accent)" : "var(--bg3)",
        color: m.role === "user" ? "var(--accent-text)" : "var(--text)",
        borderBottomRightRadius: m.role === "user" ? 2 : 10,
        borderBottomLeftRadius:  m.role === "user" ? 10 : 2,
      }}>
        {m.content}
      </div>
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, height: "calc(100vh - 136px)", animation: "fadeIn .2s ease" }}>
      {!savedApiKey && (
        <div className="card" style={{ padding: "10px 14px" }}>
          <input className="input" type="password" placeholder="Groq API key for chat (gsk_...)" value={apiKey} onChange={e => setApiKey(e.target.value)} />
        </div>
      )}
      <div className="card scrollbar" style={{ flex: 1, padding: 16, overflowY: "auto", display: "flex", flexDirection: "column", gap: 12 }}>
        {msgs.map((m, i) => <Bubble key={i} m={m} />)}
        {loading && (
          <div style={{ display: "flex", gap: 10 }}>
            <div style={{ width: 30, height: 30, borderRadius: "50%", background: "var(--bg3)", display: "flex", alignItems: "center", justifyContent: "center" }}>🤖</div>
            <div style={{ padding: "10px 14px", background: "var(--bg3)", borderRadius: 10, display: "flex", gap: 4, alignItems: "center" }}>
              {[0,1,2].map(i => <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--text3)", animation: `typing 1s ease ${i*0.2}s infinite` }} />)}
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>
      <div className="card" style={{ padding: "8px 14px", display: "flex", gap: 10, alignItems: "center" }}>
        <input className="input" style={{ flex: 1, border: "none", background: "transparent", outline: "none", fontSize: 13 }}
          placeholder="Ask anything about your dataset..."
          value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()} />
        <button className="btn-primary" onClick={send} disabled={loading || !input.trim()} style={{ padding: "7px 14px" }}>
          {loading ? <Spinner size={13} /> : "Send →"}
        </button>
      </div>
    </div>
  );
}

// ─── REPORT PAGE ──────────────────────────────────────────────────
function ReportPage({ token, jobId, jobResult, setPage }) {
  const [dlPDF, setDlPDF] = useState(false);
  const [dlCSV, setDlCSV] = useState(false);

  async function download(type) {
    if (type === "pdf") setDlPDF(true); else setDlCSV(true);
    try {
      const url = `${API_BASE}/api/${type === "pdf" ? "download_report" : "download_cleaned"}/${jobId}`;
      const r = await fetch(url, { credentials: "include", headers: { Authorization: `Bearer ${token}` } });
      if (!r.ok) { alert(type === "pdf" ? "PDF failed — install reportlab: pip install reportlab" : "CSV not found"); return; }
      const blob = await r.blob();
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
      a.download = type === "pdf" ? "DataMind_Report.pdf" : "cleaned_dataset.csv"; a.click();
    } catch (e) { alert("Download error: " + e.message); }
    finally { if (type === "pdf") setDlPDF(false); else setDlCSV(false); }
  }

  if (!jobResult) return <Empty icon="📄" title="No report yet" sub="Run an analysis to generate a report" action="⬆ Upload" onAction={() => setPage("upload")} />;

  const ml    = jobResult.ml_recommendation || {};
  const stats = jobResult.stats || {};
  const rep   = jobResult.analysis_data || jobResult;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Analysis Report</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 16 }}>
          {[
            ["Dataset",  jobResult.filename || "—"],
            ["Rows",     (stats.total_rows || 0).toLocaleString()],
            ["Quality",  `${stats.quality_score || 0}%`],
            ["Model",    ml.best_model?.split(" ").slice(-1)[0] || "—"],
            ["Score",    formatScore(ml.best_score, ml.task_type)],
            ["Task",     ml.task_type || "—"],
          ].map(([l, v]) => (
            <div key={l} style={{ padding: "10px 12px", background: "var(--bg3)", borderRadius: 8 }}>
              <div style={{ fontSize: 10, color: "var(--text3)", textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 3 }}>{l}</div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>{v}</div>
            </div>
          ))}
        </div>
        {jobResult.summary && (
          <div style={{ fontSize: 13, color: "var(--text2)", lineHeight: 1.65, padding: "12px 14px", background: "var(--bg3)", borderRadius: 8, marginBottom: 16 }}>
            {jobResult.summary}
          </div>
        )}
        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn-primary" onClick={() => download("pdf")} disabled={dlPDF || !jobId}>
            {dlPDF ? <Spinner size={13} /> : "📥"} {dlPDF ? "Generating..." : "Download PDF Report"}
          </button>
          <button className="btn-ghost" onClick={() => download("csv")} disabled={dlCSV || !jobId}>
            {dlCSV ? <Spinner size={13} /> : "📊"} Cleaned CSV
          </button>
        </div>
      </div>

      {/* AI source badges */}
      {jobResult.ai_sources && (
        <div className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: "var(--text3)" }}>AI sources:</span>
          <Badge type={jobResult.ai_sources.analyst === "groq" ? "green" : "yellow"}>Analyst: {jobResult.ai_sources.analyst}</Badge>
          <Badge type={jobResult.ai_sources.reporter === "groq" ? "green" : "yellow"}>Reporter: {jobResult.ai_sources.reporter}</Badge>
          {jobResult.degraded_mode && <Badge type="red">Degraded mode</Badge>}
        </div>
      )}
    </div>
  );
}

// ─── DASHBOARD PAGE ───────────────────────────────────────────────
// ─── HISTORY (experiment tracking log) ─────────────────────────────
function formatRunScore(run) {
  if (run.best_score == null) return "—";
  if (run.scoring_metric === "accuracy") return `${(run.best_score * 100).toFixed(1)}%`;
  if (run.scoring_metric === "silhouette") return `silhouette ${run.best_score.toFixed(3)}`;
  if (run.scoring_metric === "mae") return `MAE ${run.best_score.toFixed(3)}`;
  return `R²=${run.best_score.toFixed(3)}`;
}

function HistoryPage({ token, setPage }) {
  const [runs, setRuns]       = useState(null);
  const [err, setErr]         = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await apiFetch("/api/history", {}, token);
        const d = await r.json();
        if (!cancelled) setRuns(r.ok ? d.runs : []);
        if (!r.ok) setErr(d.error || "Could not load history");
      } catch {
        if (!cancelled) { setRuns([]); setErr("Connection lost"); }
      }
    })();
    return () => { cancelled = true; };
  }, [token]);

  if (runs === null) return <div style={{ padding: 40, textAlign: "center" }}><Spinner size={20} /></div>;
  if (runs.length === 0) {
    return <Empty icon="📈" title="No analysis history yet" sub="Every completed run gets logged here automatically — run an analysis to start building a track record." action="⬆ Upload Dataset" onAction={() => setPage("upload")} />;
  }

  const chartData = [...runs].reverse().map((r, i) => ({
    idx: i + 1,
    quality: r.quality_score,
    label: r.filename,
  }));

  const avgQuality = (runs.reduce((s, r) => s + (r.quality_score || 0), 0) / runs.length).toFixed(1);
  const modelCounts = {};
  runs.forEach(r => { if (r.best_model) modelCounts[r.best_model] = (modelCounts[r.best_model] || 0) + 1; });
  const topModel = Object.entries(modelCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "—";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
        <KpiCard label="Total Runs" value={runs.length} />
        <KpiCard label="Avg Data Quality" value={`${avgQuality}%`} />
        <KpiCard label="Most-Used Model" value={topModel} highlight />
        <KpiCard label="Last Run" value={new Date(runs[0].created_at).toLocaleDateString()} sub={new Date(runs[0].created_at).toLocaleTimeString()} />
      </div>

      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Data Quality Over Time</div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="idx" tick={{ fontSize: 11 }} label={{ value: "Run #", position: "insideBottom", fontSize: 11, dy: 10 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v, n, p) => [`${v}%`, p.payload.label]} />
            <Line type="monotone" dataKey="quality" stroke="var(--accent)" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>All Runs</div>
        {err && <div style={{ fontSize: 12, color: "var(--danger)", marginBottom: 8 }}>{err}</div>}
        <div className="scrollbar" style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--text3)", borderBottom: ".5px solid var(--border)" }}>
                {["Date", "Dataset", "Task", "Best Model", "Score", "Quality", "Industry"].map(h => (
                  <th key={h} style={{ padding: "8px 10px", fontWeight: 500 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map(r => (
                <tr key={r.id} style={{ borderBottom: ".5px solid var(--border)" }}>
                  <td style={{ padding: "8px 10px", color: "var(--text3)" }}>{new Date(r.created_at).toLocaleDateString()}</td>
                  <td style={{ padding: "8px 10px" }}>{r.filename}</td>
                  <td style={{ padding: "8px 10px" }}><Badge type="blue">{r.task_type}</Badge></td>
                  <td style={{ padding: "8px 10px" }}>{r.best_model || "—"}</td>
                  <td style={{ padding: "8px 10px", fontWeight: 500 }}>{formatRunScore(r)}</td>
                  <td style={{ padding: "8px 10px" }}>{r.quality_score}%</td>
                  <td style={{ padding: "8px 10px", color: "var(--text3)" }}>{r.industry || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DashboardPage({ jobResult, jobStep, jobStatus, jobStepHistory, setPage }) {
  const stats = jobResult?.stats || {};
  const ml    = jobResult?.ml_recommendation || {};

  if (!jobResult) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
        <div className="card" style={{ padding: 36, textAlign: "center" }}>
          <div style={{ fontSize: 42, marginBottom: 14 }}>🧠</div>
          <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: "-0.3px", marginBottom: 8 }}>Welcome to DataMind AI</div>
          <div style={{ fontSize: 13, color: "var(--text3)", marginBottom: 24, lineHeight: 1.6 }}>
            Upload a CSV or Excel dataset to start your AI-powered analysis.<br/>
            Four intelligent agents will clean, analyse, model, and summarise your data.
          </div>
          <button className="btn-primary" onClick={() => setPage("upload")}>⬆ Upload Dataset</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
          {[["🔍","Data Detective","Cleans & profiles your CSV automatically"],["🧠","AI Analyst","Groq LLaMA 3.3 70B generates business insights"],["⚙️","ML Engineer","Trains & compares 5+ ML models with SHAP"],["📝","Reporter","Writes plain-English executive summaries"]].map(([i,t,d]) => (
            <div key={t} className="card" style={{ padding: "14px 16px" }}>
              <div style={{ fontSize: 24, marginBottom: 8 }}>{i}</div>
              <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 4 }}>{t}</div>
              <div style={{ fontSize: 11, color: "var(--text3)", lineHeight: 1.4 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeIn .2s ease" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
        <KpiCard label="Rows Analysed" value={(stats.total_rows || 0).toLocaleString()} />
        <KpiCard label="Columns"       value={stats.total_cols || 0} />
        <KpiCard label="Data Quality"  value={`${stats.quality_score || 0}%`} />
        <KpiCard label="Best Model"    value={ml.best_model?.split(" ").slice(-1)[0] || "—"} sub={formatScore(ml.best_score, ml.task_type)} highlight />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.6fr", gap: 12 }}>
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Agent Pipeline</div>
          <AgentPipeline step={jobStep} jobStatus={jobStatus} stepHistory={jobStepHistory} />
        </div>
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Quick Navigation</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {[["💡","Insights","insights"],["📊","Charts","charts"],["🤖","ML Results","ml"],["🎯","Predict","predict"],["💬","Chat AI","chat"],["📄","Reports","report"]].map(([ico,lab,pg]) => (
              <button key={pg} className="btn-ghost" onClick={() => setPage(pg)} style={{ justifyContent: "center", padding: 10 }}>
                {ico} {lab}
              </button>
            ))}
          </div>
        </div>
      </div>

      {ml.domain && (
        <div className="card" style={{ padding: "12px 16px", display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 12, color: "var(--text3)" }}>Detected domain:</span>
          <Badge type="purple">{ml.domain}</Badge>
          <span style={{ fontSize: 12, color: "var(--text3)", marginLeft: 4 }}>Data health:</span>
          <Badge type={ml.data_health === "excellent" || ml.data_health === "good" ? "green" : "yellow"}>{ml.data_health}</Badge>
        </div>
      )}
    </div>
  );
}

// ─── APP SHELL ────────────────────────────────────────────────────
function AppShell({ user, token, onLogout }) {
  const [page, setPage]         = useState("dashboard");
  const [theme, setTheme]       = useState("light");
  const [jobId, setJobId]       = useState(null);
  const [savedApiKey, setSavedKey] = useState("");
  const [jobResult, setResult]  = useState(null);
  const [jobStep, setStep]      = useState(0);
  const [jobStatus, setStatus]  = useState("idle");
  const [jobStepHistory, setJobStepHistory] = useState([]);

  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); }, [theme]);

  function handleJobDone(result, id, stepHist) {
    setResult(result); setStatus("done"); setJobStepHistory(stepHist || []); setPage("insights");
  }

  function setJobMeta({ jobId, apiKey }) {
    setJobId(jobId); setSavedKey(apiKey); setStep(1); setStatus("running");
  }

  return (
    <div style={{ display: "flex", height: "100vh", background: "var(--bg)", overflow: "hidden", padding: 14, gap: 14 }}>
      <Sidebar page={page} setPage={setPage} user={user} onLogout={onLogout} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0, gap: 14 }}>
        <Topbar page={page} theme={theme} setTheme={setTheme} jobResult={jobResult} />
        <div className="scrollbar" style={{ flex: 1, overflowY: "auto", padding: "2px 4px 4px" }}>
          {page === "dashboard" && <DashboardPage jobResult={jobResult} jobStep={jobStep} jobStatus={jobStatus} jobStepHistory={jobStepHistory} setPage={setPage} />}
          {page === "upload"    && <UploadPage token={token} onJobDone={handleJobDone} setJobMeta={setJobMeta} />}
          {page === "insights"  && <InsightsPage  jobResult={jobResult} setPage={setPage} />}
          {page === "charts"    && <ChartsPage    jobResult={jobResult} setPage={setPage} />}
          {page === "ml"        && <MLResultsPage jobResult={jobResult} setPage={setPage} />}
          {page === "predict"   && <PredictPage   token={token} jobId={jobId} setPage={setPage} />}
          {page === "chat"      && <ChatPage      token={token} jobId={jobId} savedApiKey={savedApiKey} />}
          {page === "report"    && <ReportPage    token={token} jobId={jobId} jobResult={jobResult} setPage={setPage} />}
          {page === "history"   && <HistoryPage   token={token} setPage={setPage} />}
        </div>
      </div>
    </div>
  );
}

// ─── ROOT APP ─────────────────────────────────────────────────────
export default function App() {
  const [user, setUser]   = useState(null);
  const [token, setToken] = useState("");

  function handleAuth(d) { setUser({ name: d.name, email: d.email }); setToken(d.token || ""); }

  function handleLogout() {
    fetch(`${API_BASE}/api/logout`, { method: "POST", credentials: "include", headers: { Authorization: `Bearer ${token}` } }).catch(() => {});
    setUser(null); setToken("");
  }

  return (
    <>
      <style>{CSS}</style>
      {!user
        ? <AuthScreen onAuth={handleAuth} />
        : <AppShell user={user} token={token} onLogout={handleLogout} />
      }
    </>
  );
}
