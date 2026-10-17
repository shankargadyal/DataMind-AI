import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          padding: 24,
          background: "#0f172a",
          color: "#e2e8f0",
          fontFamily: "system-ui, sans-serif",
        }}>
          <div style={{
            maxWidth: 720,
            width: "100%",
            background: "#111827",
            border: "1px solid #334155",
            borderRadius: 14,
            padding: 24,
            lineHeight: 1.6,
          }}>
            <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 10 }}>DataMind could not render</div>
            <div style={{ color: "#cbd5e1", marginBottom: 14 }}>
              The app hit a runtime error while loading. This usually means one component is crashing before the page can show.
            </div>
            <pre style={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              background: "#0b1220",
              border: "1px solid #1f2937",
              borderRadius: 10,
              padding: 16,
              color: "#fca5a5",
              fontSize: 12,
              margin: 0,
            }}>
              {String(this.state.error?.stack || this.state.error?.message || this.state.error)}
            </pre>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function showFatalOverlay(title, detail) {
  const el = document.getElementById("fatal-overlay") || document.createElement("div");
  el.id = "fatal-overlay";
  el.style.cssText = [
    "position:fixed",
    "inset:0",
    "z-index:99999",
    "display:grid",
    "place-items:center",
    "background:#0f172a",
    "color:#e2e8f0",
    "font-family:system-ui,sans-serif",
    "padding:24px",
  ].join(";");
  el.innerHTML = `
    <div style="max-width:760px;width:100%;background:#111827;border:1px solid #334155;border-radius:14px;padding:24px;line-height:1.6">
      <div style="font-size:20px;font-weight:700;margin-bottom:10px">${title}</div>
      <pre style="white-space:pre-wrap;word-break:break-word;background:#0b1220;border:1px solid #1f2937;border-radius:10px;padding:16px;color:#fca5a5;font-size:12px;margin:0">${detail}</pre>
    </div>
  `;
  document.body.appendChild(el);
}

window.addEventListener("error", (event) => {
  showFatalOverlay("JavaScript error", String(event.error?.stack || event.error?.message || event.message || "Unknown error"));
});

window.addEventListener("unhandledrejection", (event) => {
  showFatalOverlay("Unhandled promise rejection", String(event.reason?.stack || event.reason?.message || event.reason || "Unknown rejection"));
});

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
