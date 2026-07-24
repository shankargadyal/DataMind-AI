"""
Database models — SQLAlchemy, defaulting to a local SQLite file.

This replaces the original users.json flat-file auth store. The interface
(`get_user_by_email`, `create_user`) is deliberately small so app.py's auth
routes barely had to change shape — the storage backend moved, the call
sites mostly didn't.

Swapping to Postgres for real multi-instance deployment later is a one-line
change: set DATABASE_URL to a postgres:// URL instead of the sqlite default.
See ARCHITECTURE.md §4 for why SQLite-on-one-instance is the right call today.
"""
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at    = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {"name": self.name, "email": self.email}


class ExperimentRun(db.Model):
    """One row per completed analysis — the experiment-tracking log.

    The point isn't to replace MLflow/W&B; it's to demonstrate the habit
    of tracking model performance over time rather than discarding it the
    moment a job finishes, which is the same instinct that real MLOps work
    requires at any scale.
    """
    __tablename__ = "experiment_runs"

    id              = db.Column(db.Integer, primary_key=True)
    job_id          = db.Column(db.String(64), index=True)
    user_email      = db.Column(db.String(255), index=True)
    filename        = db.Column(db.String(255))
    created_at      = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    mode            = db.Column(db.String(32))       # auto / classification / regression / clustering / forecasting
    task_type       = db.Column(db.String(32))       # what actually ran
    target_column   = db.Column(db.String(255))
    industry        = db.Column(db.String(32))

    rows            = db.Column(db.Integer)
    cols            = db.Column(db.Integer)
    quality_score   = db.Column(db.Float)

    best_model      = db.Column(db.String(120))
    best_score      = db.Column(db.Float)
    scoring_metric  = db.Column(db.String(64))        # "accuracy", "r2", "silhouette", "mae" — keeps best_score interpretable

    total_duration  = db.Column(db.Float)             # seconds, from the pipeline timing (see app.py set_step)

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "filename": self.filename,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "mode": self.mode,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "industry": self.industry,
            "rows": self.rows,
            "cols": self.cols,
            "quality_score": self.quality_score,
            "best_model": self.best_model,
            "best_score": self.best_score,
            "scoring_metric": self.scoring_metric,
            "total_duration": self.total_duration,
        }


class LLMCallLog(db.Model):
    """One row per Groq call — the LLMOps trace.

    ExperimentRun (above) logs one summary row per completed job. This logs
    every individual LLM call within that job (Analyst, Reporter, Chat),
    which is what's needed to see latency/token/failure patterns per agent
    rather than only the end-to-end job duration.
    """
    __tablename__ = "llm_call_logs"

    id                 = db.Column(db.Integer, primary_key=True)
    job_id             = db.Column(db.String(64), index=True)
    user_email         = db.Column(db.String(255), index=True)
    agent_name         = db.Column(db.String(64))     # "analyst" / "reporter" / "chat"
    model              = db.Column(db.String(120))
    created_at         = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    latency_ms         = db.Column(db.Float)
    prompt_tokens      = db.Column(db.Integer, nullable=True)
    completion_tokens  = db.Column(db.Integer, nullable=True)
    success            = db.Column(db.Boolean, default=True)
    error              = db.Column(db.String(500), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "success": self.success,
            "error": self.error,
        }


# ── Convenience helpers used by app.py ──────────────────────────────────

def get_user_by_email(email: str):
    return User.query.filter_by(email=(email or "").strip().lower()).first()


def create_user(name: str, email: str, password_hash: str) -> User:
    user = User(name=name, email=email.strip().lower(), password_hash=password_hash)
    db.session.add(user)
    db.session.commit()
    return user


def log_experiment_run(**fields) -> ExperimentRun:
    run = ExperimentRun(**fields)
    db.session.add(run)
    db.session.commit()
    return run


def get_runs_for_user(email: str, limit: int = 100):
    return (ExperimentRun.query
            .filter_by(user_email=email)
            .order_by(ExperimentRun.created_at.desc())
            .limit(limit)
            .all())


def log_llm_call(**fields) -> LLMCallLog:
    call = LLMCallLog(**fields)
    db.session.add(call)
    db.session.commit()
    return call


def get_llm_calls_for_job(job_id: str):
    return (LLMCallLog.query
            .filter_by(job_id=job_id)
            .order_by(LLMCallLog.created_at.asc())
            .all())
