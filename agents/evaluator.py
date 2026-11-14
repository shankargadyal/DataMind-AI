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


# ─────────────────────────────────────────────────────────────────
# Numerical-consistency checks (deterministic — no LLM involved)
# ─────────────────────────────────────────────────────────────────
import re as _re


def _flag(check: str, message: str, severity: str = "fail") -> dict:
    return {"check": check, "severity": severity, "message": message}


def _check_missing_consistency(det: dict, dataset_summary: dict) -> list[dict]:
    flags = []
    orig = det.get("missing_original")
    if not orig or not dataset_summary:
        return flags
    chat_side = dataset_summary.get("missing_values") or {}
    for col, n in orig.items():
        if col in chat_side and int(chat_side[col]) != int(n):
            flags.append(_flag("missing_value_consistency",
                               f"Missing count for '{col}' differs: profiling={n}, chat summary={chat_side[col]}."))
    total_orig = sum(int(v) for v in orig.values())
    if det.get("missing_before") is not None and int(det["missing_before"]) != total_orig:
        flags.append(_flag("missing_value_consistency",
                           f"Total missing before cleaning ({det['missing_before']}) != sum of per-column counts ({total_orig})."))
    for col, comp in (det.get("column_completeness") or {}).items():
        rows = max((det.get("original_shape") or [0])[0], 1)
        expected = round((1 - int(orig.get(col, 0)) / rows) * 100, 1)
        if abs(float(comp) - expected) > 0.11:
            flags.append(_flag("completeness_consistency",
                               f"Column '{col}' completeness {comp}% disagrees with original missing count ({expected}% expected)."))
    return flags


def _check_model_consistency(ml: dict, rep: dict) -> list[dict]:
    flags = []
    models = [m for m in (ml.get("models") or []) if not m.get("error")]
    if not models:
        return flags
    metric = ml.get("score_metric") or ("accuracy" if ml.get("task_type") == "classification" else "r2")
    best = [m for m in models if m.get("is_best")]
    if len(best) != 1:
        return [_flag("model_consistency", f"Expected exactly one best model, found {len(best)}.")]
    b = best[0]
    if ml.get("best_model") and ml["best_model"] != b.get("name"):
        flags.append(_flag("model_consistency", f"Reported best model '{ml['best_model']}' != flagged best '{b.get('name')}'."))
    hold = (b.get("metrics") or {}).get(metric)
    if hold is not None:
        if abs(float(b.get("score", 0)) - float(hold)) > 1e-3:
            flags.append(_flag("metric_consistency",
                               f"Comparison score {b.get('score')} != detailed {metric} {hold} for {b.get('name')} (different evaluation protocols)."))
        if abs(float(ml.get("best_score", 0) or 0) - float(hold)) > 1e-3:
            flags.append(_flag("metric_consistency", f"best_score {ml.get('best_score')} != detailed {metric} {hold}."))
    cm = (b.get("metrics") or {}).get("confusion_matrix")
    if cm and ml.get("task_type") == "classification":
        total = sum(sum(r) for r in cm)
        n_test = (b.get("metrics") or {}).get("n_test")
        if n_test and total != n_test:
            flags.append(_flag("confusion_matrix_consistency", f"Confusion matrix sums to {total} but test size is {n_test}."))
        if total:
            acc = sum(cm[i][i] for i in range(len(cm))) / total
            if hold is not None and abs(acc - float(hold)) > 1e-3:
                flags.append(_flag("confusion_matrix_consistency",
                                   f"Confusion-matrix accuracy {acc:.4f} != reported accuracy {hold}."))
    # Percent claims about accuracy in the Reporter's prose must match a real stored score.
    text = " ".join(str(rep.get(k, "")) for k in ("report", "headline", "score_meaning"))
    allowed = []
    for m in models:
        for v in ((m.get("metrics") or {}).get(metric), (m.get("metrics") or {}).get("cv_score"), m.get("score")):
            if v is not None:
                allowed.append(float(v) * 100 if abs(float(v)) <= 1 else float(v))
    for m in _re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:accuracy|accurate)|accuracy[^.\d]{0,25}(\d{1,3}(?:\.\d+)?)\s*%", text, _re.I):
        val = float(m.group(1) or m.group(2))
        if allowed and not any(abs(val - a) <= 0.6 for a in allowed):
            flags.append(_flag("report_vs_metrics", f"Report text claims {val}% accuracy, which matches no stored evaluation result.", "flag"))
    return flags


def _check_summary_consistency(dataset_summary: dict) -> list[dict]:
    flags = []
    if not dataset_summary:
        return flags
    dist = dataset_summary.get("target_distribution")
    tgt = dataset_summary.get("target_column")
    if dist and tgt:
        expected = dataset_summary.get("num_rows", 0) - int((dataset_summary.get("missing_values") or {}).get(tgt, 0))
        if sum(int(v) for v in dist.values()) != expected:
            flags.append(_flag("group_count_consistency",
                               f"Target class counts sum to {sum(dist.values())}, expected {expected} (rows with a non-missing target)."))
    return flags



def _check_degenerate_evaluation(ml: dict) -> list[dict]:
    """A hold-out set that contains no examples of some class makes that class's precision/recall/F1
    (and any 100% accuracy) meaningless: the model may simply be predicting the majority class."""
    flags = []
    if ml.get("task_type") != "classification":
        return flags
    models = [m for m in (ml.get("models") or []) if not m.get("error")]
    best = next((m for m in models if m.get("is_best")), None)
    cm = ((best or {}).get("metrics") or {}).get("confusion_matrix")
    if not cm:
        return flags
    labels = ml.get("confusion_labels") or list(range(len(cm)))
    absent = [str(labels[i]) if i < len(labels) else str(i) for i, row in enumerate(cm) if sum(row) == 0]
    if absent:
        flags.append(_flag(
            "degenerate_evaluation",
            f"The hold-out set contains no examples of class(es) {', '.join(absent)}, so precision/recall/F1 and the "
            f"reported score are not meaningful for them (the model may only be predicting the majority class). "
            f"More data for the rare class is needed for a reliable evaluation.", "warn"))
    return flags



def _check_no_better_than_baseline(ml: dict) -> list[dict]:
    """A classifier that does not beat 'always predict the most common class' has learned nothing useful
    (typical causes: dirty/over-split target labels, no signal in the features, tiny hold-out)."""
    if ml.get("task_type") != "classification":
        return []
    models = [m for m in (ml.get("models") or []) if not m.get("error")]
    best = next((m for m in models if m.get("is_best")), None)
    mt = (best or {}).get("metrics") or {}
    cm, acc = mt.get("confusion_matrix"), mt.get("accuracy")
    if not cm or acc is None:
        return []
    total = sum(sum(r) for r in cm)
    if total <= 0:
        return []
    baseline = max(sum(r) for r in cm) / total
    margin = max(0.02, (baseline * (1 - baseline) / total) ** 0.5)   # ~1 standard error: small hold-outs are noisy
    if baseline < 1.0 and acc <= baseline + margin:
        predicted = sum(1 for j in range(len(cm[0])) if sum(row[j] for row in cm) > 0)
        single = " The selected model predicts a single class for every hold-out row." if predicted == 1 else ""
        return [_flag("no_better_than_baseline",
                      f"Best model accuracy ({acc:.1%}) is not better than always predicting the most common class "
                      f"({baseline:.1%}) on the hold-out set.{single} Accuracy alone can hide a useful model on imbalanced "
                      f"data, so also check per-class recall / macro F1; if recall is poor, look for inconsistent target "
                      f"labels (e.g. 'Yes'/'yes'/'Y') or weak features.", "warn")]
    return []


def _check_leakage(ml: dict) -> list[dict]:
    """The model's score may be inflated because the target is (almost) a formula of its own inputs."""
    rel = ml.get("data_relationships") or {}
    leaks = rel.get("target_leakage") or []
    if not leaks:
        return []
    lines = [x for x in (rel.get("summary") or []) if x.startswith(("Target", "Feature"))] or ["Target leakage suspected."]
    return [_flag("leakage_suspected",
                  " ".join(lines[:2]) + " The reported score is likely inflated; remove the flagged column(s) and re-run "
                  "to see the honest performance.", "warn")]


def _check_selection_mismatch(ml: dict) -> list[dict]:
    """The model chosen as 'best' is selected by cross-validated score on the TRAINING split, but the
    leaderboard displays hold-out score. On a real distribution shift (e.g. a time-series regime change)
    the two can disagree, and the report would star a model that another row visibly beats on its own
    table — with no explanation. This does not say the selection is WRONG (CV-based selection is the more
    honest choice for model picking); it only says so plainly, so the reader isn't confused by the table."""
    models = [m for m in (ml.get("models") or []) if not m.get("error") and m.get("score") is not None]
    best = next((m for m in models if m.get("is_best")), None)
    if not best or len(models) < 2:
        return []
    metric = ml.get("evaluation_protocol", {}).get("primary_metric") or best.get("score_metric") or "score"
    top_holdout = max(models, key=lambda m: m["score"])
    if top_holdout["name"] != best["name"] and top_holdout["score"] - best["score"] > 1e-9:
        sel_metric = ml.get("selection_metric", "selection_score")
        return [_flag("selection_metric_differs_from_displayed_score",
                      f"'{best['name']}' was selected as the best model by {sel_metric} "
                      f"({best.get('selection_score')}), but '{top_holdout['name']}' has a higher displayed "
                      f"hold-out {metric} ({top_holdout['score']} vs {best['score']}). The leaderboard's "
                      f"score column and the selection rule are not the same metric — see "
                      f"evaluation_protocol.selection_rule for which one chose the winner.", "warn")]
    return []


def _check_regression_no_skill(ml: dict) -> list[dict]:
    """R² <= 0 means the model is no better than (or worse than) always predicting the mean — it has
    learned nothing usable. This is the regression counterpart of _check_no_better_than_baseline, and is
    what should stop a report from calling a negative-R² model's output a reliable forecast."""
    if ml.get("task_type") != "regression":
        return []
    best = next((m for m in (ml.get("models") or []) if m.get("is_best")), None)
    if not best or best.get("score") is None:
        return []
    r2 = best["score"]
    if r2 <= 0:
        return [_flag("regression_no_skill",
                      f"Best model R² ({r2:.3f}) is at or below zero on the hold-out set: it predicts no better "
                      f"than always guessing the average value, so its forecasts should not be called reliable. "
                      f"Likely causes: the target has little relationship to the available features, not enough "
                      f"history, or (for time-series data) a regime change between the training and test periods.",
                      "fail" if r2 < -0.05 else "warn")]
    return []


def _check_degraded_mode(rep: dict) -> list[dict]:
    """When the LLM call failed, the Reporter falls back to generic template text (e.g. 'the AI has analysed
    your data...reliably forecast future outcomes') that does not reflect the actual model quality. The
    report must say this happened instead of presenting fallback text as a normal AI-generated summary."""
    if rep.get("degraded_mode"):
        return [_flag("degraded_mode_not_disclosed",
                      "The AI narrative (executive summary / insights / recommendations) could not be generated "
                      "and generic fallback text was used instead. The report must disclose this rather than "
                      "presenting the fallback text as a normal AI-written summary.", "warn")]
    return []


def _check_forecast_sanity(ml: dict) -> list[dict]:
    """A forecast section with no target name and Trend R² / Std Error of exactly 0 is very likely printing
    dict.get(key, 0) defaults for keys that were never computed — not a genuine zero-variance result."""
    fut = ml.get("future_prediction") or {}
    if not fut:
        return []
    flags = []
    if not fut.get("target"):
        flags.append(_flag("forecast_missing_target",
                           "The forecast section has no target column name recorded.", "warn"))
    if fut.get("trend_r2") in (0, 0.0, None) and fut.get("std_error") in (0, 0.0, None):
        flags.append(_flag("forecast_suspected_default_values",
                           "Trend R² and Std Error are both exactly 0, which is almost always a missing-data "
                           "default rather than a genuine calculation — check whether these fields are actually "
                           "computed for this forecast.", "warn"))
    return flags

def _check_requested_target(requested_target: str | None, ml: dict) -> list[dict]:
    """The user asked for a specific target column but the pipeline used a different one."""
    used = ml.get("target_column")
    if requested_target and used and requested_target != used:
        return [_flag("target_column_not_used",
                      f"Requested target column '{requested_target}' was not found in the dataset; "
                      f"'{used}' was auto-selected instead. Check the column name.", "warn")]
    return []

def run_consistency_checks(rep: dict, det: dict, ml: dict, dataset_summary: dict | None = None,
                           requested_target: str | None = None) -> list[dict]:
    flags = []
    flags += _check_missing_consistency(det, dataset_summary or {})
    flags += _check_model_consistency(ml, rep)
    flags += _check_summary_consistency(dataset_summary or {})
    flags += _check_degenerate_evaluation(ml)
    flags += _check_no_better_than_baseline(ml)
    flags += _check_leakage(ml)
    flags += _check_requested_target(requested_target, ml)
    flags += _check_selection_mismatch(ml)
    flags += _check_regression_no_skill(ml)
    # NOTE: _check_degraded_mode() exists and is unit-tested but is deliberately NOT wired into the
    # blocking review chain here. merge_flags() treats ANY non-empty flag list as FLAGGED (holds the job
    # for human review), and now that the fallback narrative text itself is conditioned on real model
    # skill (see reporter.py), bare degraded_mode no longer implies an unsupported claim reached the
    # report -- forcing a manual review on every LLM timeout would be a disproportionate UX regression.
    # Disclosure instead happens directly in the PDF (see report_generator.py's degraded-mode note).
    flags += _check_forecast_sanity(ml)
    return flags


def merge_flags(eval_result: dict, extra_flags: list[dict]) -> dict:
    """Merge extra flags and recompute status. FAILED if any flag is a 'fail', never PASSED with flags."""
    flags = list(eval_result.get("flags", [])) + list(extra_flags)
    if any(f["severity"] == "fail" for f in flags):
        status = "FAILED"
    elif flags:
        status = "FLAGGED"
    else:
        status = eval_result.get("status", "PASSED") if eval_result.get("status") != "NOT_EVALUATED" else "PASSED"
    out = dict(eval_result)
    out.update({"status": status, "flags": flags,
                "checked_fields": list(dict.fromkeys(list(eval_result.get("checked_fields", [])) +
                                                     ["missing_values", "model_metrics", "confusion_matrix", "group_counts"]))})
    return out


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
