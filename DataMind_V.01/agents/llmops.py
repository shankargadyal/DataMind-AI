"""
LLMOps: per-call tracking for every Groq request.
────────────────────────────────────────────────
ExperimentRun (models.py) already logs one row per completed job. That's
job-level, not call-level — it can't tell you that Reporter is consistently
slower than Analyst, or that a specific agent is silently failing more
often. This wraps every individual Groq call (Analyst, Reporter, Chat) to
capture latency, token usage, and success/failure — the minimum a real
LLMOps setup tracks.

Never blocks the caller: if the DB write fails (e.g. running outside an
app context, or in a unit test), it logs to an in-memory list instead
and moves on. The pipeline's actual result is never affected by whether
this tracking succeeds.
"""

from __future__ import annotations
import time
from contextlib import contextmanager

# Fallback store used when the DB isn't available (e.g. plain unit tests
# that call agents directly without a Flask app context).
_fallback_log: list[dict] = []


def _persist(record: dict):
    try:
        from models import log_llm_call
        log_llm_call(**record)
    except Exception:
        _fallback_log.append(record)


@contextmanager
def track_llm_call(job_id: str | None, user_email: str | None, agent_name: str,
                    model: str = "llama-3.3-70b-versatile"):
    """
    Usage:
        with llmops.track_llm_call(job_id, user_email, "reporter") as ctx:
            response = client.chat.completions.create(...)
            ctx["response"] = response   # so token usage can be read off it
    """
    start = time.time()
    ctx = {"response": None}
    error = None
    try:
        yield ctx
    except Exception as e:
        error = str(e)
        raise
    finally:
        latency_ms = round((time.time() - start) * 1000, 1)
        usage = getattr(ctx.get("response"), "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        _persist({
            "job_id": job_id,
            "user_email": user_email,
            "agent_name": agent_name,
            "model": model,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "success": error is None,
            "error": error,
        })
