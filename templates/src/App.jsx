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
  const [jobResult, setResult]  = useState(null);
  const [jobStep, setStep]      = useState(0);
  const [jobStatus, setStatus]  = useState("idle");
  const [jobStepHistory, setJobStepHistory] = useState([]);

  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); }, [theme]);

  function handleJobDone(result, id, stepHist) {
    setResult(result); setStatus("done"); setJobStepHistory(stepHist || []); setPage("insights");
  }

  // NOTE: no longer accepts/stores an apiKey — the chat API key box was
  // removed from ChatPage, and /api/analyze never needed one to begin with.
  function setJobMeta({ jobId }) {
    setJobId(jobId); setStep(1); setStatus("running");
  }

  return (
    <div className="app-root" style={{ display: "flex", height: "100vh", background: "var(--bg)", overflow: "hidden", padding: 14, gap: 14 }}>
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
          {page === "chat"      && <ChatPage      token={token} jobId={jobId} />}
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
