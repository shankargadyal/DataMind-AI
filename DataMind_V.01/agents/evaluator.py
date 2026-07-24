"""
Agent: Evaluator / Guardrails
──────────────────────────────
Reporter (agents/reporter.py) asks an LLM to independently produce
`confidence`, `should_i_worry`, `risk_score`, and a plain-English `report`.
Nothing currently checks whether those actually agree with each other or
with the real numbers (quality_score, best_score, cv_score, sample size).
This agent is that checkpoint — it runs after Reporter and before the
result is shown to the user, and never blocks the pipeline: on failure it
degrades to "not evaluated" rather than crashing the run.

Checks performed:
  1. Confidence vs. reality — does reporter's self-reported "high/medium/low"
     confidence match quality_score, best_score, and sample size?
  2. risk_score vs. should_i_worry — do these two fields agree with each
     other? (e.g. risk_score=75 but should_i_worry starts with "No")
  3. Small-sample flag — rows below a threshold get flagged regardless of
     what the LLM claimed, since small samples make any score unreliable.
  4. Output completeness — required keys present and non-empty.

This doubles as a lightweight LLMOps trace: every run's eval verdict gets
logged, so — like the existing experiment history — you're not throwing
away whether a report was trustworthy the moment the job finishes.
"""

from __future__ import annotations

REQUIRED_KEYS = [
    "report", "headline", "score_meaning", "top_recommendation",
    "confidence", "should_i_worry", "risk_score", "next_steps",
]

SMALL_SAMPLE_THRESHOLD = 100


def _check_completeness(rep: dict) -> list[dict]:
    flags = []
    for key in REQUIRED_KEYS:
        val = rep.get(key)
        if val is None or (isinstance(val, (str, list)) and len(val) == 0):
            flags.append({
                "check": "completeness",
                "severity": "fail",
                "message": f"Reporter output missing or empty field: '{key}'",
            })
    return flags


def _check_confidence_vs_reality(rep: dict, quality: float, best_score: float, rows: int) -> list[dict]:
    flags = []
    claimed = (rep.get("confidence") or "").lower()
    if claimed == "high" and (quality < 70 or best_score < 0.65 or rows < SMALL_SAMPLE_THRESHOLD):
        flags.append({
            "check": "confidence_vs_reality",
            "severity": "flag",
            "message": (
                f"Reporter claims 'high' confidence, but quality={quality}%, "
                f"best_score={round(best_score,3)}, rows={rows} don't fully support that."
            ),
        })
    if claimed == "low" and quality > 90 and best_score > 0.85 and rows >= SMALL_SAMPLE_THRESHOLD:
        flags.append({
            "check": "confidence_vs_reality",
            "severity": "flag",
            "message": "Reporter claims 'low' confidence despite strong quality/score/sample size — possibly overcautious.",
        })
    return flags


def _check_risk_vs_worry(rep: dict) -> list[dict]:
    flags = []
    risk_score = rep.get("risk_score", 0) or 0
    worry_text = (rep.get("should_i_worry") or "").strip().lower()
    says_no = worry_text.startswith("no")
    says_yes = worry_text.startswith("yes")
    if risk_score >= 50 and says_no:
        flags.append({
            "check": "risk_vs_worry",
            "severity": "flag",
            "message": f"risk_score={risk_score} (elevated) but should_i_worry says 'No' — contradiction.",
        })
    if risk_score < 20 and says_yes:
        flags.append({
            "check": "risk_vs_worry",
            "severity": "flag",
            "message": f"risk_score={risk_score} (low) but should_i_worry says 'Yes' — contradiction.",
        })
    return flags


def _check_small_sample(rows: int) -> list[dict]:
    if rows < SMALL_SAMPLE_THRESHOLD:
        return [{
            "check": "small_sample",
            "severity": "flag",
            "message": f"Only {rows} rows — any score/confidence claim should be treated as provisional.",
        }]
    return []


def run_evaluator(rep: dict, det: dict, ml: dict) -> dict:
    """Entry point: called from run_pipeline() right after Reporter finishes."""
    quality    = det.get("quality_score", 0) or 0
    rows       = (det.get("original_shape") or [0])[0]
    best_score = ml.get("best_score", 0) or 0

    flags = []
    flags += _check_completeness(rep)
    flags += _check_confidence_vs_reality(rep, quality, best_score, rows)
    flags += _check_risk_vs_worry(rep)
    flags += _check_small_sample(rows)

    if any(f["severity"] == "fail" for f in flags):
        status = "FAILED"
    elif flags:
        status = "FLAGGED"
    else:
        status = "PASSED"

    return {
        "status": status,
        "flags": flags,
        "checked_fields": REQUIRED_KEYS,
    }
