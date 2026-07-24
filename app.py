"""
DataMind v2.4 — Flask Backend
Bug-fixes applied in this version:
  BUG-01  Hardcoded secret key → read from env
  BUG-02  SESSION_COOKIE_SECURE hardcoded False → auto-True on production
  BUG-03  Plain SHA-256 passwords (no salt) → bcrypt (graceful fallback kept)
  BUG-04  No auth guard on /api/analyze, /api/chat, /api/predict, /api/whatif
  BUG-05  jobs dict grows forever (memory leak) → capped at MAX_JOBS with LRU eviction
  BUG-06  _chats dict grows forever → same cap
  BUG-07  _tokens dict grows forever → pruned on login
  BUG-08  Upload folder inside app dir (web-accessible risk) → /tmp by default
  BUG-09  No rate-limiting on login/register → flask-limiter (optional install)
  BUG-10  CORS wildcard fallback missing → explicit allowed origins only
  BUG-11  np.bool_ / np.str_ not caught by _safe_ctx → added
  BUG-12  reporter called with raw det (has DataFrame) → use safe_det
  BUG-13  Duplicate json import (json + _json) → unified
  BUG-14  _safe_ctx doesn't recurse into dicts/lists → fixed
"""

import os, time, threading, re, secrets, json, tempfile, uuid, socket
from collections import OrderedDict
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import io
import numpy as np
import pandas as pd

load_dotenv()

# ── BUG-01: Secret key from env, never hardcoded ──────────────────
_SECRET = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

# ── BUG-03: bcrypt with graceful SHA-256 fallback ─────────────────
try:
    import bcrypt as _bcrypt
    HAS_BCRYPT = True
except ImportError:
    import hashlib as _hashlib
    HAS_BCRYPT = False
    print("[WARN] bcrypt not installed — using SHA-256. Run: pip install bcrypt")

def _hash(pw: str) -> str:
    if HAS_BCRYPT:
        return _bcrypt.hashpw(pw.encode(), _bcrypt.gensalt()).decode()
    import hashlib
    return hashlib.sha256(pw.encode()).hexdigest()

def _check_pw(pw: str, stored: str) -> bool:
    if HAS_BCRYPT:
        try:
            return _bcrypt.checkpw(pw.encode(), stored.encode())
        except Exception:
            pass
    import hashlib
    return hashlib.sha256(pw.encode()).hexdigest() == stored

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

app = Flask(__name__)
app.secret_key = _SECRET
_ROOT_DIR = os.path.dirname(__file__)
_FRONTEND_DIR = os.path.join(_ROOT_DIR, "templates")
_FRONTEND_DIST = os.path.join(_FRONTEND_DIR, "dist")
_FRONTEND_DIST_INDEX = os.path.join(_FRONTEND_DIST, "index.html")

# ── BUG-02: Secure cookie only on production ──────────────────────
_IS_PROD = os.environ.get("FLASK_ENV", "development") == "production"
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=_IS_PROD,
)

# ── BUG-10: Strict CORS ───────────────────────────────────────────
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
).split(",") if o.strip()]
CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS)

# ── BUG-08: Upload folder outside app dir ────────────────────────
_DEFAULT_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "datamind_uploads")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", _DEFAULT_UPLOAD_DIR)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# ── Database (replaces the old users.json flat file) ───────────────
# DATABASE_URL lets this swap to Postgres with zero code changes for a
# real multi-instance deployment — see ARCHITECTURE.md §4.
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datamind.db")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", f"sqlite:///{_DEFAULT_DB_PATH}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

from models import db, User, get_user_by_email, create_user, log_experiment_run, get_runs_for_user
db.init_app(app)
with app.app_context():
    db.create_all()

# ── BUG-05/06: LRU-capped dicts ───────────────────────────────────
MAX_JOBS = int(os.environ.get("MAX_JOBS", 200))

class _LRUDict(OrderedDict):
    def __init__(self, maxsize=200):
        super().__init__()
        self.maxsize = maxsize
    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            del self[next(iter(self))]

jobs   = _LRUDict(MAX_JOBS)
_chats = _LRUDict(MAX_JOBS)
_tokens: dict[str, dict] = {}  # small enough; pruned per-user on login

# ── BUG-09: Optional rate limiting ───────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=[])
    HAS_LIMITER = True
except ImportError:
    HAS_LIMITER = False

def _rate_limit(limit_str):
    def decorator(f):
        if HAS_LIMITER:
            return limiter.limit(limit_str)(f)
        return f
    return decorator

def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

# ── Users now live in the database (models.py) — see init above ───

# ── Agents ────────────────────────────────────────────────────────
from agents import detective, analyst, ml_engineer, reporter
from agents import rag_agent, evaluator

HAS_PDF = False
generate_pdf_report = None
try:
    from agents.report_generator import generate_pdf_report
    HAS_PDF = True
except ImportError:
    # ── Inline fallback PDF generator using reportlab ─────────────────
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        import reportlab.lib.colors as _rl_colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable,
        )

        def generate_pdf_report(analysis: dict, ctx_safe: dict, ml_safe: dict, industry: dict = None, eval_result: dict = None, rag_info: dict = None, llm_calls: list = None) -> bytes:
            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            accent = _rl_colors.HexColor("#6366F1")

            h1   = ParagraphStyle("H1",  parent=styles["Heading1"],  fontSize=20,
                                  textColor=accent, spaceAfter=8)
            h2   = ParagraphStyle("H2",  parent=styles["Heading2"],  fontSize=13,
                                  textColor=accent, spaceAfter=6, spaceBefore=14)
            body = ParagraphStyle("Body",parent=styles["Normal"],     fontSize=9,
                                  leading=14, spaceAfter=4)
            bold = ParagraphStyle("Bold",parent=body, fontName="Helvetica-Bold")

            def _tbl(data, col_widths):
                t = Table(data, colWidths=col_widths)
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0), accent),
                    ("TEXTCOLOR",     (0,0), (-1,0), _rl_colors.white),
                    ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                    ("FONTSIZE",      (0,0), (-1,-1), 9),
                    ("GRID",          (0,0), (-1,-1), 0.5, _rl_colors.lightgrey),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1),
                     [_rl_colors.white, _rl_colors.HexColor("#F8F8FF")]),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                ]))
                return t

            story = []

            # Title
            story.append(Paragraph("DataMind Analysis Report", h1))
            story.append(Paragraph(f"File: <b>{analysis.get('filename','Dataset')}</b>", body))
            story.append(HRFlowable(width="100%", thickness=1, color=accent, spaceAfter=12))

            # Stats overview
            stats = analysis.get("stats", {})
            if stats:
                story.append(Paragraph("Dataset Overview", h2))
                story.append(_tbl([
                    ["Metric", "Value"],
                    ["Total Rows",    str(stats.get("total_rows", "—"))],
                    ["Total Columns", str(stats.get("total_cols", "—"))],
                    ["Quality Score", f"{stats.get('quality_score', 0):.1f}%"],
                    ["Insights Found",str(stats.get("insights_count", "—"))],
                ], [6*cm, 8*cm]))
                story.append(Spacer(1, 10))

            # Executive summary
            summary = analysis.get("summary", "")
            if summary:
                story.append(Paragraph("Executive Summary", h2))
                for para in summary.split("\n\n"):
                    if para.strip():
                        story.append(Paragraph(para.strip(), body))
                story.append(Spacer(1, 6))

            # Key insights
            insights = analysis.get("key_insights", [])
            if insights:
                story.append(Paragraph("Key Insights", h2))
                for i, ins in enumerate(insights, 1):
                    if isinstance(ins, dict):
                        title  = ins.get("title", ins.get("insight", f"Insight {i}"))
                        detail = ins.get("detail", ins.get("description", ""))
                        impact = ins.get("impact", "")
                        story.append(Paragraph(f"<b>{i}. {title}</b>", bold))
                        if detail:  story.append(Paragraph(detail, body))
                        if impact:  story.append(Paragraph(f"Impact: {impact}", body))
                    else:
                        story.append(Paragraph(f"{i}. {ins}", body))
                    story.append(Spacer(1, 4))

            # Recommended actions
            actions = analysis.get("actions", [])
            if actions:
                story.append(Paragraph("Recommended Actions", h2))
                for a in actions:
                    if isinstance(a, dict):
                        story.append(Paragraph(
                            f"<b>{a.get('action', a.get('title',''))}</b>"
                            f" — {a.get('reason', a.get('detail',''))}", body))
                    else:
                        story.append(Paragraph(f"\u2022 {a}", body))

            # ML results
            ml_rec = analysis.get("ml_recommendation", {})
            if ml_rec.get("best_model"):
                story.append(Paragraph("ML Model Results", h2))
                score = ml_rec.get("best_score", 0) or 0
                score_pct = f"{score*100:.1f}%" if score <= 1 else f"{score:.1f}%"
                story.append(Paragraph(
                    f"Best Model: <b>{ml_rec['best_model']}</b>  |  "
                    f"Score: <b>{score_pct}</b>  |  "
                    f"Task: {ml_rec.get('task_type','—')}  |  "
                    f"Target: {ml_rec.get('target_column','—')}", body))

                fi = (ml_rec.get("feature_importances")
                      or ml_safe.get("feature_importances", []))
                if fi:
                    story.append(Spacer(1, 6))
                    story.append(Paragraph("Top Feature Importances", h2))
                    rows = [["Feature", "Importance"]]
                    for f in fi[:12]:
                        rows.append([
                            str(f.get("feature", f.get("name", ""))),
                            f"{float(f.get('importance', f.get('value', 0))):.4f}",
                        ])
                    story.append(_tbl(rows, [9*cm, 5*cm]))
                    story.append(Spacer(1, 8))

            # Column statistics
            col_stats = ctx_safe.get("column_stats", {})
            if col_stats:
                story.append(Paragraph("Column Statistics", h2))
                rows = [["Column", "Type", "Min", "Max", "Mean", "Missing"]]
                for col, st in list(col_stats.items())[:20]:
                    rows.append([
                        col,
                        str(st.get("dtype", "—")),
                        str(round(st["min"],  2)) if "min"  in st else "—",
                        str(round(st["max"],  2)) if "max"  in st else "—",
                        str(round(st["mean"], 2)) if "mean" in st else "—",
                        str(st.get("missing", 0)),
                    ])
                story.append(_tbl(rows, [3.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2.5*cm]))
                story.append(Spacer(1, 8))

            # Footer
            story.append(Spacer(1, 16))
            story.append(HRFlowable(width="100%", thickness=0.5, color=_rl_colors.lightgrey))
            story.append(Paragraph("Generated by DataMind AI Analytics Platform", body))

            doc.build(story)
            return buf.getvalue()

        HAS_PDF = True
    except ImportError:
        print("[WARN] reportlab not installed — PDF disabled. Run: pip install reportlab")

HAS_DASHBOARD = False
dashboard_agent = None
try:
    from agents import dashboard_agent
    HAS_DASHBOARD = True
except ImportError:
    pass

HAS_INDUSTRY = False
industry_intelligence = None
try:
    from agents import industry_intelligence
    HAS_INDUSTRY = True
except ImportError:
    pass

# ── Logging (Section 12: production-readiness) ─────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("datamind")

# ── Utils ─────────────────────────────────────────────────────────
def _valid_email(e): return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e))
def _new_token():    return secrets.token_hex(32)

def _make_auth_response(name: str, email: str) -> dict:
    session["user_email"] = email
    # BUG-07: remove stale tokens for this user
    for t in [t for t, u in _tokens.items() if u["email"] == email]:
        del _tokens[t]
    token = _new_token()
    _tokens[token] = {"email": email, "name": name}
    return {"name": name, "email": email, "token": token}

def _current_user() -> dict | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        user  = _tokens.get(token)
        if user:
            return user
    email = session.get("user_email", "")
    if email == "guest@datamind.ai":
        return {"name": "Guest", "email": email}
    if email:
        u = get_user_by_email(email)
        if u:
            return {"name": u.name, "email": u.email}
    name  = request.headers.get("X-User-Name", "")
    email = request.headers.get("X-User-Email", "")
    if name and email:
        return {"name": name, "email": email}
    return None

def _require_auth():
    user = _current_user()
    if not user:
        return None, (jsonify({"error": "Authentication required"}), 401)
    return user, None

def _get_owned_job(job_id: str, require_done: bool = False):
    user, err = _require_auth()
    if err:
        return None, err
    job = jobs.get(job_id)
    if not job:
        return None, (jsonify({"error": "Job not found"}), 404)
    if job.get("user") not in ("guest@datamind.ai", user["email"]):
        return None, (jsonify({"error": "Forbidden"}), 403)
    if require_done and job.get("status") != "done":
        return None, (jsonify({"error": "Analysis not complete"}), 404)
    return job, None

# ── BUG-11/14: Recursive safe serializer ─────────────────────────
def _safe_val(v):
    if isinstance(v, pd.DataFrame):    return "__DATAFRAME__"
    if isinstance(v, np.ndarray):      return v.tolist()
    if isinstance(v, np.integer):      return int(v)
    if isinstance(v, np.floating):     return float(v)
    if isinstance(v, np.bool_):        return bool(v)
    if isinstance(v, np.str_):         return str(v)
    if isinstance(v, tuple):           return [_safe_val(i) for i in v]
    if isinstance(v, list):            return [_safe_val(i) for i in v]
    if isinstance(v, dict):            return {k: _safe_val(vv) for k, vv in v.items()}
    return v

def _safe_ctx(det: dict) -> dict:
    return {k: _safe_val(v) for k, v in det.items()}

def _safe_ml(ml: dict) -> dict:
    return {k: _safe_val(v) for k, v in ml.items() if not k.startswith("_")}


# ══════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/api/register", methods=["POST"])
@_rate_limit("10/minute")
def register():
    b     = request.get_json(force=True)
    name  = b.get("name", "").strip()
    email = b.get("email", "").strip().lower()
    pw    = b.get("password", "").strip()
    if not name:                return jsonify({"error": "Name required"}), 400
    if not _valid_email(email): return jsonify({"error": "Invalid email"}), 400
    if len(pw) < 6:             return jsonify({"error": "Password too short"}), 400
    if get_user_by_email(email): return jsonify({"error": "Email already registered"}), 409
    create_user(name, email, _hash(pw))
    return jsonify(_make_auth_response(name, email))


@app.route("/api/login", methods=["POST"])
@_rate_limit("10/minute")
def login():
    b     = request.get_json(force=True)
    email = b.get("email", "").strip().lower()
    pw    = b.get("password", "").strip()
    user  = get_user_by_email(email)
    if not user or not _check_pw(pw, user.password_hash):
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify(_make_auth_response(user.name, user.email))


@app.route("/api/logout", methods=["POST"])
def logout():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        _tokens.pop(auth_header[7:].strip(), None)
    session.pop("user_email", None)
    return jsonify({"message": "Logged out"})


@app.route("/api/guest", methods=["POST"])
def guest_login():
    return jsonify(_make_auth_response("Guest", "guest@datamind.ai"))


SAMPLE_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "employee_attrition_sample.csv")


@app.route("/api/sample")
def sample_dataset():
    """Serves a bundled synthetic HR dataset so anyone can try the full
    pipeline without uploading their own file or having a Groq key — the
    'Try it instantly' button on the upload page hits this."""
    user, err = _require_auth()
    if err: return err
    if not os.path.exists(SAMPLE_DATASET_PATH):
        return jsonify({"error": "Sample dataset not found on server"}), 404
    return send_file(SAMPLE_DATASET_PATH, mimetype="text/csv",
                      as_attachment=False, download_name="employee_attrition_sample.csv")


@app.route("/api/me")
def me():
    user = _current_user()
    if user:
        return jsonify({"name": user["name"], "email": user["email"]})
    return jsonify({"error": "Not logged in"}), 401


# ══════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if os.path.exists(_FRONTEND_DIST_INDEX):
        return send_from_directory(_FRONTEND_DIST, "index.html")
    if _port_open("127.0.0.1", 5173):
        return """
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8">
            <meta http-equiv="refresh" content="0; url=http://127.0.0.1:5173">
            <title>DataMind</title>
            <style>
              body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;display:grid;place-items:center;min-height:100vh;margin:0;padding:24px}
              .box{max-width:560px;background:#111827;border:1px solid #334155;border-radius:12px;padding:24px}
              a{color:#60a5fa}
            </style>
          </head>
          <body>
            <div class="box">
              <h1>Opening DataMind</h1>
              <p>The frontend is running on port 5173, so this page is redirecting you there.</p>
              <p>If it does not redirect, open <a href="http://127.0.0.1:5173">http://127.0.0.1:5173</a>.</p>
            </div>
          </body>
        </html>
        """
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>DataMind</title>
        <style>
          body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;display:grid;place-items:center;min-height:100vh;margin:0;padding:24px}
          .box{max-width:640px;background:#111827;border:1px solid #334155;border-radius:12px;padding:28px;line-height:1.6}
          code{background:#0b1220;padding:2px 6px;border-radius:6px}
          a{color:#60a5fa}
        </style>
      </head>
      <body>
        <div class="box">
          <h1>DataMind is split into two parts</h1>
          <p>The backend is running, but the UI is a separate Vite app.</p>
          <p>Open the frontend at <a href="http://127.0.0.1:5173">http://127.0.0.1:5173</a> after starting it with <code>npm run dev</code> inside <code>templates</code>.</p>
          <p>If you want a single-command run, use the new <code>start-dev.ps1</code> script from the project root.</p>
        </div>
      </body>
    </html>
    """


@app.route("/assets/<path:filename>")
def frontend_assets(filename):
    asset_dir = os.path.join(_FRONTEND_DIST, "assets")
    if os.path.exists(asset_dir):
        return send_from_directory(asset_dir, filename)
    return jsonify({"error": "Frontend build not found. Run `npm run build` in `templates/`."}), 404


# ══════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════

@app.route("/api/analyze", methods=["POST"])
def analyze():
    user, err = _require_auth()   # BUG-04
    if err: return err

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    query         = request.form.get("query", "Give me key insights about this data.")
    api_key       = request.form.get("api_key", "").strip() or GROQ_API_KEY
    target_column = request.form.get("target_column", "").strip()
    mode          = request.form.get("mode", "auto").strip().lower()
    industry      = request.form.get("industry", "").strip().lower()
    if mode not in ("auto", "classification", "regression", "clustering", "forecasting"):
        mode = "auto"
    if not api_key:
        return jsonify({"error": "Groq API key required"}), 400

    filename = secure_filename(file.filename)
    uid      = uuid.uuid4().hex
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_{filename}")
    file.save(filepath)

    job_id = f"job_{uid}"
    jobs[job_id] = {
        "status": "running", "step": 0, "step_name": "Starting...",
        "logs": [], "result": None, "error": None,
        "filename": filename, "user": user["email"], "target_column": target_column,
        "step_history": [], "total_duration": None,
    }
    threading.Thread(
        target=run_pipeline,
        args=(job_id, filepath, query, api_key, filename, target_column, mode, industry),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job, err = _get_owned_job(job_id)
    if err: return err
    out = {
        "status": job["status"], "step": job["step"],
        "step_name": job.get("step_name", ""),
        "logs": job["logs"], "error": job.get("error"),
        "step_history": job.get("step_history", []),
        "total_duration": job.get("total_duration"),
        "server_time": time.time(),
    }
    if job["status"] in ("done", "needs_review") and job["result"]:
        ad = job["result"].get("analysis_data", {})
        ml_safe = job["result"].get("ml_safe", {})
        ctx_safe = job["result"].get("ctx_safe", {})
        out["result"] = {
            "analysis_data":    ad,
            # Hoist top-level fields so the frontend can read them directly
            # without deep-diving into analysis_data
            "key_insights":     ad.get("key_insights", []),
            "actions":          ad.get("actions", []),
            "distribution_data":ad.get("distribution_data", [])
                                or ctx_safe.get("distribution_data", []),
            "ml_recommendation":ad.get("ml_recommendation", {}),
            "stats":            ad.get("stats", {}),
            "summary":          ad.get("summary", ""),
            "filename":         ad.get("filename", ""),
            # ML extras for charts
            "feature_importances": ml_safe.get("feature_importances", []),
            "future_prediction":   ml_safe.get("future_prediction", {}),
        }
        out["eval"] = job["result"].get("eval", {})
        if job["status"] == "needs_review":
            out["review_required"] = True
            out["review"] = job.get("review")
    return jsonify(out)


@app.route("/api/dashboard/<job_id>")
def get_dashboard(job_id):
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err
    return job.get("result", {}).get("dashboard_html", "<p>No dashboard</p>"), 200


@app.route("/api/download_cleaned/<job_id>")
def download_cleaned(job_id):
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err
    path = job["result"].get("cleaned_path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Cleaned file not found"}), 404
    return send_file(path, as_attachment=True)


@app.route("/api/download_report/<job_id>")
def download_report(job_id):
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err
    if not HAS_PDF:
        return jsonify({"error": "PDF not installed (pip install reportlab)"}), 500
    r = job["result"]
    analysis, ctx_safe, ml_safe = r.get("analysis_data",{}), r.get("ctx_safe",{}), r.get("ml_safe",{})
    industry = r.get("industry", {})
    if not analysis:
        return jsonify({"error": "No analysis data"}), 500
    try:
        rag_idx = job.get("rag_index")
        rag_info = {"chunk_count": len(rag_idx.chunks)} if rag_idx is not None else {"chunk_count": 0}
        try:
            from models import get_llm_calls_for_job
            llm_calls = [c.to_dict() for c in get_llm_calls_for_job(job_id)]
        except Exception:
            llm_calls = []
        pdf_bytes = generate_pdf_report(analysis, ctx_safe, ml_safe, industry,
                                         eval_result=r.get("eval"), rag_info=rag_info, llm_calls=llm_calls)
        if not pdf_bytes or len(pdf_bytes) < 100:
            return jsonify({"error": "PDF empty"}), 500
        buf = io.BytesIO(pdf_bytes); buf.seek(0)
        safe = job.get("filename", "report").replace(".csv", "")
        return send_file(buf, mimetype="application/pdf", as_attachment=True,
                         download_name=f"DataMind_Report_{safe}.pdf")
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"PDF failed: {e}"}), 500


# ══════════════════════════════════════════════════════════════════
# EXPLORE
# ══════════════════════════════════════════════════════════════════

@app.route("/api/explore/<job_id>")
def explore(job_id):
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err

    r   = job["result"] or {}
    ctx = r.get("ctx_safe", {})
    ad  = r.get("analysis_data", {})

    # distribution_data: prefer ctx_safe, fall back to analysis_data
    dist_data = ctx.get("distribution_data") or ad.get("distribution_data", [])

    explore_data = {
        "column_stats":         ctx.get("column_stats", {}),
        "distribution_data":    dist_data,
        "correlation":          ctx.get("correlation", {}),
        "numeric_cols":         ctx.get("numeric_cols", []),
        "categorical_cols":     ctx.get("categorical_cols", []),
        "original_shape":       ctx.get("original_shape", [0, 0]),
        "quality_score":        ctx.get("quality_score", 0),
        "quality_dimensions":   ctx.get("quality_dimensions", {}),
        "missing_values_fixed": ctx.get("missing_values_fixed", {}),
        "outlier_summary":      ctx.get("outlier_summary", {}),
        "categorical_counts":   {},
        "industry":             r.get("industry", {"available": False}),
    }

    # Try cleaned CSV first, then fall back to original upload path
    det_path = r.get("det_df_path") or r.get("cleaned_path")
    if det_path and not os.path.exists(det_path):
        det_path = None  # will build counts from ctx_safe instead

    if det_path and os.path.exists(det_path):
        try:
            df = pd.read_csv(det_path)
            for col in ctx.get("categorical_cols", [])[:8]:
                if col in df.columns:
                    vc = df[col].value_counts().head(10)
                    explore_data["categorical_counts"][col] = {
                        "labels": [str(x) for x in vc.index.tolist()],
                        "values": vc.values.tolist(),
                    }
            # Also build distribution_data from df if ctx had none
            if not explore_data["distribution_data"]:
                dist = []
                for col in ctx.get("numeric_cols", [])[:8]:
                    if col in df.columns:
                        vals = df[col].dropna().tolist()
                        dist.append({"column": col, "values": vals[:500]})
                explore_data["distribution_data"] = dist
        except Exception as ex:
            print(f"[explore] CSV read error: {ex}")
    else:
        # Build categorical_counts from column_stats when no CSV is available
        col_stats = ctx.get("column_stats", {})
        for col in ctx.get("categorical_cols", [])[:8]:
            st = col_stats.get(col, {})
            top = st.get("top_values", {})
            if top:
                explore_data["categorical_counts"][col] = {
                    "labels": list(top.keys()),
                    "values": list(top.values()),
                }

    return jsonify(explore_data)


# ══════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════

@app.route("/api/predict/<job_id>", methods=["POST"])
def predict(job_id):
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err
    artifacts = job["result"].get("ml_artifacts", {})
    model, scaler = artifacts.get("model"), artifacts.get("scaler")
    feat_cols = artifacts.get("feature_cols", [])
    task_type = artifacts.get("task_type", "regression")
    le_target = artifacts.get("le_target")
    if model is None or not feat_cols:
        return jsonify({"error": "No trained model available"}), 500
    inputs = request.get_json(force=True).get("inputs", {})
    try:
        row = []
        for col in feat_cols:
            try: row.append(float(inputs.get(col, 0)))
            except: row.append(0.0)
        X_scaled = scaler.transform(np.array([row]))
        pred_raw = model.predict(X_scaled)[0]
        if task_type == "classification" and le_target is not None:
            pred_label = le_target.inverse_transform([int(pred_raw)])[0]
        else:
            pred_label = round(float(pred_raw), 4)
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = round(float(max(model.predict_proba(X_scaled)[0])) * 100, 1)
        return jsonify({"prediction": pred_label, "confidence": confidence,
                        "task_type": task_type, "target": artifacts.get("target_column"),
                        "features_used": len(feat_cols)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_info/<job_id>")
def predict_info(job_id):
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err
    artifacts = job["result"].get("ml_artifacts", {})
    ctx = job["result"].get("ctx_safe", {})
    feat_cols = artifacts.get("feature_cols", [])
    col_stats = ctx.get("column_stats", {})
    features = []
    for col in feat_cols:
        st = col_stats.get(col, {})
        features.append({"name": col, "min": st.get("min", 0), "max": st.get("max", 100),
                         "mean": st.get("mean", 0), "median": st.get("median", 0)})
    return jsonify({"features": features, "task_type": artifacts.get("task_type", "regression"),
                    "target": artifacts.get("target_column", ""),
                    "best_model": artifacts.get("best_model", ""),
                    "best_score": artifacts.get("best_score", 0)})


@app.route("/api/whatif/<job_id>", methods=["POST"])
def whatif(job_id):
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err
    artifacts = job["result"].get("ml_artifacts", {})
    model, scaler = artifacts.get("model"), artifacts.get("scaler")
    feat_cols = artifacts.get("feature_cols", [])
    task_type = artifacts.get("task_type", "regression")
    le_target = artifacts.get("le_target")
    if model is None or not feat_cols:
        return jsonify({"error": "No trained model"}), 500
    scenarios = request.get_json(force=True).get("scenarios", [])
    if not scenarios:
        return jsonify({"error": "No scenarios provided"}), 400
    try:
        rows = [[float(sc.get(col, 0)) for col in feat_cols] for sc in scenarios]
        X_scaled = scaler.transform(np.array(rows))
        preds_raw = model.predict(X_scaled)
        results = []
        for i, pred_raw in enumerate(preds_raw):
            if task_type == "classification" and le_target is not None:
                pred_label = le_target.inverse_transform([int(pred_raw)])[0]
            else:
                pred_label = round(float(pred_raw), 4)
            results.append({"scenario": i + 1, "prediction": pred_label})
        return jsonify({"results": results})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
@_rate_limit("30/minute")
def chat():
    user, err = _require_auth()   # BUG-04
    if err: return err

    from groq import Groq
    b       = request.get_json(force=True)
    message = b.get("message", "").strip()
    job_id  = b.get("job_id", "")
    api_key = b.get("api_key", "").strip() or GROQ_API_KEY
    if not message: return jsonify({"error": "Empty message"}), 400
    if not api_key: return jsonify({"error": "API key required"}), 400

    job = jobs.get(job_id, {})
    if job_id:
        if not job:
            return jsonify({"error": "Job not found"}), 404
        if job.get("user") not in ("guest@datamind.ai", user["email"]):
            return jsonify({"error": "Forbidden"}), 403
    ad  = (job.get("result") or {}).get("analysis_data", {})
    ml_rec = ad.get("ml_recommendation", {})

    # RAG: retrieve only the chunks relevant to this specific question,
    # searched across the FULL analysis output (all insights, all actions,
    # SHAP captions, quality report, industry KPIs) — not a fixed
    # first-N slice like before.
    rag_index = job.get("rag_index")
    retrieved_text = "None available."
    if rag_index is not None:
        hits = rag_index.retrieve(message, k=6)
        if hits:
            retrieved_text = "\n".join(f"- ({h['source']}) {h['text']}" for h in hits)

    system_prompt = (
        "You are DataMind AI, an expert data analyst assistant embedded in a SaaS analytics platform.\n"
        f"Dataset: {ad.get('filename', 'Unknown')}\n"
        f"ML Target: {ml_rec.get('target_column','unknown')}  |  "
        f"Best Model: {ml_rec.get('best_model','—')}  |  "
        f"Task: {ml_rec.get('task_type','—')}\n"
        f"Relevant retrieved context for this question:\n{retrieved_text}\n"
        "Answer in 2-5 clear sentences, using only the retrieved context above plus the "
        "model/target info. If the retrieved context doesn't cover the question, say what "
        "you don't have rather than guessing. If no data is loaded, say so and ask the "
        "user to upload a CSV first."
    )

    # Use job_id as chat key; fall back to user email so chat persists across page refreshes
    chat_key = job_id if job_id and job_id in jobs else f"user_{(user or {}).get('email','anon')}"
    history = _chats.setdefault(chat_key, [])
    history.append({"role": "user", "content": message})
    try:
        from agents import llmops
        client = Groq(api_key=api_key)
        with llmops.track_llm_call(job_id, user["email"], "chat") as ctx:
            comp = client.chat.completions.create(
                model="llama-3.3-70b-versatile", max_tokens=600, temperature=0.5,
                messages=[{"role": "system", "content": system_prompt}] + history[-12:],
            )
            ctx["response"] = comp
        reply = comp.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply, "chat_key": chat_key})
    except Exception as e:
        import traceback; traceback.print_exc()
        err_str = str(e)
        if "api_key" in err_str.lower() or "authentication" in err_str.lower():
            return jsonify({"error": "Invalid Groq API key. Please check your key in settings."}), 401
        if "rate" in err_str.lower():
            return jsonify({"error": "Groq rate limit hit. Please wait a moment and try again."}), 429
        return jsonify({"error": f"Chat error: {err_str}"}), 500




# ══════════════════════════════════════════════════════════════════
# INSIGHTS / ACTIONS / DISTRIBUTION  (dedicated endpoints)
# ══════════════════════════════════════════════════════════════════

@app.route("/api/jobs/<job_id>/review", methods=["POST"])
def review_job(job_id):
    """HIL endpoint: approve or reject a job that Guardrails/Eval held for
    human review. Approving unlocks it exactly like a normal 'done' job —
    every existing require_done=True endpoint starts working the moment
    status flips to 'done', so no other route needed changing."""
    user, err = _require_auth()
    if err: return err
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("user") not in ("guest@datamind.ai", user["email"]):
        return jsonify({"error": "Forbidden"}), 403
    if job.get("status") != "needs_review":
        return jsonify({"error": "This job isn't awaiting review"}), 400

    data     = request.get_json(silent=True) or {}
    decision = (data.get("decision") or "").lower()
    note     = data.get("note", "")
    if decision not in ("approve", "reject"):
        return jsonify({"error": "decision must be 'approve' or 'reject'"}), 400

    job["review"] = {
        "decision": decision,
        "note": note,
        "reviewed_by": user["email"],
        "reviewed_at": time.time(),
    }
    job["status"] = "done" if decision == "approve" else "rejected"
    log(job_id, f"[HIL] Reviewed by {user['email']}: {decision}" + (f" — {note}" if note else ""))
    return jsonify({"status": job["status"], "review": job["review"]})


@app.route("/api/llm_calls/<job_id>")
def get_llm_calls(job_id):
    """LLMOps trace: every individual Groq call made for this job — which
    agent, latency, token usage, success/failure. Job-level summary lives
    in /api/history; this is the per-call detail underneath it."""
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err
    try:
        from models import get_llm_calls_for_job
        calls = get_llm_calls_for_job(job_id)
        return jsonify({"calls": [c.to_dict() for c in calls], "count": len(calls)})
    except Exception as e:
        return jsonify({"calls": [], "count": 0, "error": str(e)})


@app.route("/api/history")
def get_history():
    """Experiment tracking log: every completed analysis run for the current
    user, most recent first. Lets the UI show model performance over time
    instead of discarding it once a job finishes."""
    user, err = _require_auth()
    if err: return err
    limit = min(int(request.args.get("limit", 50)), 200)
    runs = get_runs_for_user(user["email"], limit=limit)
    return jsonify({"runs": [r.to_dict() for r in runs], "count": len(runs)})


@app.route("/api/insights/<job_id>")
def get_insights(job_id):
    """Return key_insights, actions, and distribution_data for a completed job."""
    job, err = _get_owned_job(job_id, require_done=True)
    if err: return err
    ad       = (job["result"] or {}).get("analysis_data", {})
    ctx_safe = (job["result"] or {}).get("ctx_safe", {})
    ml_safe  = (job["result"] or {}).get("ml_safe", {})
    dist     = ad.get("distribution_data") or ctx_safe.get("distribution_data", [])
    return jsonify({
        "key_insights":      ad.get("key_insights", []),
        "actions":           ad.get("actions", []),
        "distribution_data": dist,
        "ml_recommendation": ad.get("ml_recommendation", {}),
        "feature_importances": (
            (ad.get("ml_recommendation") or {}).get("feature_importances")
            or ml_safe.get("feature_importances", [])
        ),
        "future_prediction": (
            (ad.get("ml_recommendation") or {}).get("future_prediction")
            or ml_safe.get("future_prediction", {})
        ),
        "summary": ad.get("summary", ""),
        "eval": (job["result"] or {}).get("eval", {"status": "NOT_EVALUATED", "flags": []}),
    })


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    """Clear chat history for a given job_id or chat_key."""
    user, err = _require_auth()
    if err: return err
    b = request.get_json(force=True)
    key = b.get("job_id") or b.get("chat_key") or f"user_{(user or {}).get('email','anon')}"
    _chats.pop(key, None)
    return jsonify({"cleared": True, "key": key})


# ══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════

def log(job_id, msg):
    jobs[job_id]["logs"].append(msg)
    logger.info(f"[{job_id}] {msg}")


def set_step(job_id, step, name):
    """Advance the job to a new pipeline stage and record a timestamp for it,
    so the frontend can show live per-agent elapsed time (Section 1: agent
    monitoring dashboard) instead of just a static progress label."""
    now = time.time()
    job = jobs[job_id]
    history = job.setdefault("step_history", [])
    if history:
        history[-1]["ended_at"] = now
        history[-1]["duration"] = round(now - history[-1]["started_at"], 2)
    history.append({"step": step, "name": name, "started_at": now, "ended_at": None, "duration": None})
    job["step"] = step
    job["step_name"] = name


def run_pipeline(job_id, filepath, query, api_key, filename, target_column="", mode="auto", industry=""):
    jobs[job_id]["pipeline_started_at"] = time.time()
    try:
        set_step(job_id, 1, "Cleaning & profiling data")
        log(job_id, "[AG1] Data Detective loading CSV...")

        det = detective.run_detective(filepath)
        det["filename"] = filename
        df  = det.get("df")
        if df is None:
            raise ValueError("DataFrame missing from detective output")

        log(job_id, f"[AG1] {det['original_shape'][0]} rows, {det['original_shape'][1]} cols, quality {det.get('quality_score',0)}%")

        cleaned_path = filepath.replace(".csv", "_cleaned.csv")
        try:
            df.to_csv(cleaned_path, index=False)
        except Exception:
            cleaned_path = filepath

        set_step(job_id, 2, "Finding AI insights")
        log(job_id, "[AG2] Sending to Groq LLaMA 3.3 70B...")

        safe_det = _safe_ctx(det)
        ana = analyst.run_analyst(safe_det, query, api_key, job_id=job_id, user_email=jobs[job_id].get("user"))
        if target_column:
            ana.setdefault("ml_recommendation", {})["target_column"] = target_column
        log(job_id, f"[AG2] {len(ana.get('key_insights',[]))} insights generated")

        set_step(job_id, 3, "Training ML models")
        log(job_id, "[AG3] Feature engineering + model training...")

        ml   = ml_engineer.run_ml_engineer(df, det, ana, mode_override=("" if mode == "auto" else mode))
        best = next((m for m in ml.get("models", []) if m.get("is_best")), None)
        log(job_id, f"[AG3] Best: {best['name'] if best else 'N/A'} @ {round((best.get('score',0) if best else 0)*100,1)}%")
        if ml.get("future_prediction", {}).get("future_y"):
            log(job_id, f"[AG3] Future prediction: {ml['future_prediction']['future_steps']} steps ahead")

        set_step(job_id, 4, "Generating report")
        log(job_id, "[AG4] Writing executive report...")
        # BUG-12: pass safe_det, not raw det (which contains DataFrame)
        rep = reporter.run_reporter(safe_det, ana, ml, query, api_key, job_id=job_id, user_email=jobs[job_id].get("user"))
        degraded_mode = bool(ana.get("degraded_mode") or rep.get("degraded_mode"))
        if degraded_mode:
            log(job_id, "[WARN] AI fallback mode active - Groq response unavailable")

        # [AG4b] Evaluator / Guardrails — checks Reporter's claims (confidence,
        # should_i_worry, risk_score) for internal consistency and overconfidence
        # before the result reaches the user. Never fails the pipeline itself.
        try:
            eval_result = evaluator.run_evaluator(rep, safe_det, ml)
            if eval_result["status"] != "PASSED":
                for f in eval_result["flags"]:
                    log(job_id, f"[AG4b][{eval_result['status']}] {f['check']}: {f['message']}")
            else:
                log(job_id, "[AG4b] Guardrails check passed — no flags")
        except Exception as eval_err:
            log(job_id, f"[WARN] Evaluator skipped: {eval_err}")
            eval_result = {"status": "NOT_EVALUATED", "flags": [], "checked_fields": []}

        set_step(job_id, 5, "Building dashboard")
        industry_result = {"available": False}
        if HAS_INDUSTRY and industry:
            try:
                industry_result = industry_intelligence.get_industry_insights(industry, det, ana, ml)
            except Exception as ie:
                log(job_id, f"[WARN] Industry intelligence skipped: {ie}")

        dash_html = ""
        if HAS_DASHBOARD:
            try:
                dash      = dashboard_agent.run(safe_det, ana, ml, rep, query, industry=industry_result)
                dash_html = dash.get("html", "")
            except Exception as de:
                log(job_id, f"[WARN] Dashboard skipped: {de}")

        ctx_safe = _safe_ctx(det)
        ml_safe  = _safe_ml(ml)

        ml_artifacts = {
            "model":         ml.get("_model_obj"),
            "scaler":        ml.get("_scaler_obj"),
            "feature_cols":  ml.get("_feature_cols", []),
            "task_type":     ml.get("task_type"),
            "target_column": ml.get("target_column"),
            "best_model":    ml.get("best_model"),
            "best_score":    ml.get("best_score", 0),
            "le_target":     ml.get("_le_target"),
            "numeric_cols":  ml.get("_numeric_cols", []),
        }

        ml_rec = ana.get("ml_recommendation", {})
        if ml.get("models"):
            best_metrics = {}
            confusion_matrix = None
            confusion_labels = []
            for m in ml["models"]:
                if m.get("is_best"):
                    best_metrics = m.get("metrics", {})
                    confusion_matrix = best_metrics.get("confusion_matrix")
                    break
            if ml.get("task_type") == "classification":
                le = ml.get("_le_target")
                if le is not None and hasattr(le, "classes_"):
                    confusion_labels = [str(c) for c in le.classes_]
            ml_rec["models"] = [
                {
                    "name": m.get("name", "?"),
                    "score": m.get("score", m.get("mae", 0)),
                    "estimated_score": m.get("score", m.get("mae", 0)),
                    "metrics": m.get("metrics", {}),
                    "is_best": m.get("is_best", False),
                    "error": m.get("error"),
                }
                for m in ml["models"]
            ]
            ml_rec.update({
                "task_type":           ml.get("task_type"),
                "target_column":       ml.get("target_column"),
                "best_model":          ml.get("best_model"),
                "best_score":          ml.get("best_score"),
                "best_metrics":        best_metrics,
                "confusion_matrix":    confusion_matrix,
                "confusion_labels":    confusion_labels,
                "class_imbalance":     ml_safe.get("class_imbalance", {}),
                "feature_importances": ml_safe.get("feature_importances", []),
                "shap_plots":          ml_safe.get("shap_plots", {"available": False}),
                "engineered_features": ml_safe.get("engineered_features", []),
                "future_prediction":   ml_safe.get("future_prediction", {}),
                "feature_count":       ml.get("feature_count", 0),
                "scoring_metric": (
                    "Accuracy / Precision / Recall / F1"
                    if ml.get("task_type") == "classification"
                    else "R2 / MAE / RMSE"
                ),
            })

        # Ensure distribution_data is populated from multiple sources
        dist_data = (
            ctx_safe.get("distribution_data")
            or det.get("distribution_data")
            or []
        )
        # If still empty, build from column_stats
        if not dist_data:
            for col, st in (ctx_safe.get("column_stats") or {}).items():
                if st.get("dtype", "").startswith("float") or st.get("dtype", "").startswith("int"):
                    hist_vals = st.get("histogram_values") or st.get("sample_values") or []
                    if hist_vals:
                        dist_data.append({"column": col, "values": hist_vals})

        analysis_data = {
            "filename":          filename,
            "summary":           rep.get("report", "Analysis complete."),
            "key_insights":      ana.get("key_insights", []),
            "distribution_data": dist_data,
            "ml_recommendation": ml_rec,
            "actions":           ana.get("actions", []),
            "degraded_mode":     degraded_mode,
            "ai_sources": {
                "analyst": ana.get("generated_by", "unknown"),
                "reporter": rep.get("generated_by", "unknown"),
            },
            "stats": {
                "total_rows":     int(det["original_shape"][0]),
                "total_cols":     int(det["original_shape"][1]),
                "quality_score":  float(det.get("quality_score", 0)),
                "insights_count": len(ana.get("key_insights", [])),
                "missing_fixed":  ctx_safe.get("missing_values_fixed", {}),
            },
            "quality_dimensions": ctx_safe.get("quality_dimensions", {}),
            "risk_flags": rep.get("risk_flags", []),
            "risk_score": rep.get("risk_score", 0),
            "next_steps": rep.get("next_steps", []),
        }

        jobs[job_id]["result"] = {
            "analysis_data":  analysis_data,
            "ctx_safe":       ctx_safe,
            "ml_safe":        ml_safe,
            "ml_artifacts":   ml_artifacts,
            "det_df_path":    cleaned_path,
            "cleaned_path":   cleaned_path,
            "dashboard_html": dash_html,
            "industry":       industry_result,
            "eval":           eval_result,
        }

        # [AG7] Build the RAG index over the full analysis output so /api/chat
        # can retrieve relevant chunks instead of stuffing a fixed slice into
        # the prompt. Never fails the pipeline if indexing has a problem.
        try:
            jobs[job_id]["rag_index"] = rag_agent.run_rag_indexing(analysis_data, ml_safe, ctx_safe)
            log(job_id, f"[AG7] RAG index built: {len(jobs[job_id]['rag_index'].chunks)} chunks")
        except Exception as rag_err:
            log(job_id, f"[WARN] RAG indexing skipped: {rag_err}")
            jobs[job_id]["rag_index"] = None

        # [HIL] Human-in-the-loop checkpoint: if Guardrails flagged or failed
        # anything, don't auto-complete — hold the job for a human decision.
        # A NOT_EVALUATED result (evaluator itself errored) fails open, since
        # blocking on our own tooling breaking is worse than the risk it guards.
        if eval_result.get("status") in ("FLAGGED", "FAILED"):
            jobs[job_id]["status"] = "needs_review"
            jobs[job_id]["review"] = {"decision": None, "note": None, "reviewed_at": None}
            log(job_id, f"[HIL] Job held for human review — eval status: {eval_result['status']}")
        else:
            jobs[job_id]["status"] = "done"
        set_step(job_id, 6, "Complete")
        history = jobs[job_id].get("step_history", [])
        if history:
            history[-1]["ended_at"] = time.time()
            history[-1]["duration"] = 0.0
        jobs[job_id]["total_duration"] = round(time.time() - jobs[job_id]["pipeline_started_at"], 2)
        log(job_id, "[DONE] Analysis complete!")

        # ── Experiment tracking: log this run so performance is visible over time ──
        try:
            scoring_metric = {
                "classification": "accuracy", "clustering": "silhouette",
                "forecasting": "mae",
            }.get(ml.get("task_type", ""), "r2")
            with app.app_context():
                log_experiment_run(
                    job_id=job_id,
                    user_email=jobs[job_id].get("user", ""),
                    filename=filename,
                    mode=mode,
                    task_type=ml.get("task_type", ""),
                    target_column=ml.get("target_column", "") or "",
                    industry=industry or "",
                    rows=int(det["original_shape"][0]),
                    cols=int(det["original_shape"][1]),
                    quality_score=float(det.get("quality_score", 0)),
                    best_model=ml.get("best_model", "") or "",
                    best_score=float(ml.get("best_score", 0) or 0),
                    scoring_metric=scoring_metric,
                    total_duration=jobs[job_id]["total_duration"],
                )
        except Exception as log_err:
            log(job_id, f"[WARN] Experiment logging skipped: {log_err}")

    except Exception as e:
        import traceback; traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        history = jobs[job_id].get("step_history", [])
        if history and history[-1]["ended_at"] is None:
            history[-1]["ended_at"] = time.time()
            history[-1]["duration"] = round(history[-1]["ended_at"] - history[-1]["started_at"], 2)
        log(job_id, f"[ERR] {e}")


# ── Global exception handling (Section 12: production-readiness) ───
@app.errorhandler(Exception)
def _handle_unexpected_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e  # let Flask's normal 404/405/etc handling proceed
    logger.exception("Unhandled exception")
    return jsonify({"error": "Internal server error. Please try again or contact support."}), 500


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"DataMind v2.4  →  http://localhost:5000")
    print(f"  bcrypt:  {'yes' if HAS_BCRYPT else 'NO  →  pip install bcrypt'}")
    print(f"  limiter: {'yes' if HAS_LIMITER else 'NO  →  pip install flask-limiter'}")
    print(f"  uploads: {UPLOAD_FOLDER}")
    app.run(debug=not _IS_PROD, host="0.0.0.0", port=5000)
