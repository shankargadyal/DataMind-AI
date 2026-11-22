"""
Agent 4: Reporter v3.0 — Plain-English executive summary.
New in v3:
  - "Should I be worried?" section with risk score
  - Exploratory mode reporting
  - Confidence level explanation tied to CV scores
  - Warning flag integration from analyst
  - Score meaning tied to cross-validation
"""
import json
import re
from agents.analysis_evidence import qualitative_text_is_safe


_REPORT_PROMPT = """You are a trusted business advisor writing a report for a non-technical business owner.
Based on the analysis results below, write a clear, friendly executive report.

Dataset: {filename}
Composite data-quality score: {quality}/100
Data completeness: {completeness}
Rows analysed: {rows}
Task: {task_type} — predicting '{target}'
Best model: {best_model} (score: {score_str}, cross-validation: {cv_str})
Key insights: {insights}
Recommended actions: {actions}
Domain: {domain}
Warning flags: {warnings}

Write a professional yet friendly report in this exact JSON format:
{{
  "report": "2-3 paragraph plain English summary. First paragraph: what the data shows. Second: what the AI found. Third: what to do next.",
  "headline": "One punchy sentence summarising the most important finding",
  "score_meaning": "Plain English explanation of the model score — e.g. 'Your AI model correctly predicts X 87% of the time. Cross-validation confirms this holds on unseen data.'",
  "top_recommendation": "The single most important action to take right now",
  "risk_flags": ["Any data quality issues or warnings the business owner should know"],
  "confidence": "high|medium|low",
  "should_i_worry": "Direct, honest answer: 'No — your data is solid and predictions are reliable.' or 'Yes — here is what to fix first...'",
  "risk_score": 0-100,
  "next_steps": ["Step 1", "Step 2", "Step 3"]
}}

Rules:
- Never use technical jargon
- Speak directly to the business owner ("your data", "you should")
- Keep it encouraging but honest
- risk_score: 0 = no risk, 100 = severe issues
- should_i_worry must be direct and plain — no hedging
"""


_EDA_REPORT_PROMPT = """You are a data analyst explaining exploratory analysis to a non-technical business owner.

Dataset: {filename}
Composite data-quality score: {quality}/100
Data completeness: {completeness}
Rows analysed: {rows}
Key insights: {insights}
Domain: {domain}
Warning flags: {warnings}

Return this JSON:
{{
  "report": "2-3 paragraph plain English summary of what patterns and trends were found.",
  "headline": "One punchy sentence about the most interesting finding",
  "score_meaning": "Exploratory analysis does not train a prediction model — it reveals patterns instead.",
  "top_recommendation": "The most actionable insight from the exploratory analysis",
  "risk_flags": ["Any data quality issues"],
  "confidence": "high|medium|low",
  "should_i_worry": "Honest answer about data quality and completeness",
  "risk_score": 0-100,
  "next_steps": ["Explore specific patterns", "Collect more data on key variables", "Consider defining a prediction goal"]
}}
"""


_CLUSTERING_REPORT_PROMPT = """You are a data analyst explaining customer/record segmentation to a non-technical business owner.

Dataset: {filename}
Composite data-quality score: {quality}/100
Data completeness: {completeness}
Rows analysed: {rows}
Segments found: {best_model} (silhouette score: {score_str} — closer to 1.0 means more clearly separated segments)
Key insights: {insights}
Domain: {domain}
Warning flags: {warnings}

Return this JSON:
{{
  "report": "2-3 paragraph plain English summary of the segments found and what makes each one distinct.",
  "headline": "One punchy sentence about the most useful segment found",
  "score_meaning": "Plain English explanation of what the silhouette score means for how distinct these segments are.",
  "top_recommendation": "The most actionable way to use these segments (e.g. targeted marketing, different treatment per segment)",
  "risk_flags": ["Any data quality issues"],
  "confidence": "high|medium|low",
  "should_i_worry": "Honest answer about whether the segments are clear/reliable enough to act on",
  "risk_score": 0-100,
  "next_steps": ["Profile each segment further", "Test a different action per segment", "Re-run segmentation as more data arrives"]
}}
"""

_FORECASTING_REPORT_PROMPT = """You are a data analyst explaining a time-series forecast to a non-technical business owner.

Dataset: {filename}
Composite data-quality score: {quality}/100
Data completeness: {completeness}
Rows analysed: {rows}
Forecasting target: '{target}'
Best model: {best_model} (backtest error: {score_str} — lower is better)
Key insights: {insights}
Domain: {domain}
Warning flags: {warnings}

Return this JSON:
{{
  "report": "2-3 paragraph plain English summary of the trend found and what the forecast implies for the business.",
  "headline": "One punchy sentence about where the forecast says '{target}' is heading",
  "score_meaning": "Plain English explanation of the backtest error and how much to trust the forecast.",
  "top_recommendation": "The most actionable thing to do based on this forecast",
  "risk_flags": ["Any data quality issues or forecast caveats"],
  "confidence": "high|medium|low",
  "should_i_worry": "Honest answer about how reliable this forecast is",
  "risk_score": 0-100,
  "next_steps": ["Monitor actuals against forecast", "Re-forecast as new data arrives", "Plan capacity/budget around the forecasted trend"]
}}
"""


def run_reporter(det: dict, ana: dict, ml: dict, query: str, api_key: str, job_id: str = None, user_email: str = None) -> dict:
    filename    = det.get("filename", "") or "dataset"
    quality_data = det.get("quality_dimensions", {})
    quality = quality_data.get("composite_score", det.get("quality_score", 0))
    completeness = quality_data.get("dimensions", {}).get("completeness")
    completeness_text = f"{completeness:g}%" if completeness is not None else "unavailable"
    rows        = det.get("original_shape", [0])[0]
    task_type   = ml.get("task_type") or "exploratory"
    target      = ml.get("target_column", "outcome")
    best_model  = ml.get("best_model", "AI model")
    best_score  = ml.get("best_score") or 0
    domain      = ana.get("domain", "business")
    exploratory = ml.get("exploratory_mode", False) or task_type == "exploratory"

    # Score string
    if task_type == "classification":
        score_str = f"{round(best_score * 100, 1)}% accuracy"
    elif task_type == "clustering":
        score_str = f"silhouette {round(best_score, 3)}"
    elif task_type == "forecasting":
        score_str = f"MAE {round(best_score, 3)}"
    elif task_type == "exploratory":
        score_str = "N/A (exploratory mode)"
    else:
        score_str = f"R² = {round(best_score, 3)}"

    # Cross-validation score string
    best_model_data = next((m for m in ml.get("models", []) if m.get("is_best")), {})
    cv_score = (best_model_data.get("metrics") or {}).get("cv_score", 0)
    cv_std   = (best_model_data.get("metrics") or {}).get("cv_std", 0)
    cv_str   = (f"{cv_score:.3f} ± {cv_std:.3f} (cross-validation score)" if cv_score is not None else "not available")

    insights_text = "; ".join([
        f"{i.get('title','')}: {i.get('description','')}"
        for i in ana.get("key_insights", [])[:4]
    ])
    actions_text = "; ".join([
        f"({a.get('priority','')}) {a.get('action','')}"
        for a in ana.get("actions", [])[:3]
    ])
    warning_text = "; ".join([
        w.get("message", "")
        for w in ana.get("warning_flags", [])[:3]
    ]) or "none"

    # Derive a simple risk score from quality + outliers + warnings
    warning_count = len(ana.get("warning_flags", []))
    critical_count = sum(1 for w in ana.get("warning_flags", []) if w.get("severity") == "critical")
    risk_score = min(100, int((100 - quality) * 0.5 + warning_count * 5 + critical_count * 15))

    if exploratory:
        body = "Exploratory analysis has revealed key patterns and trends in your data."
        headline = f"Exploratory analysis complete — key patterns discovered in your {domain} data"
        score_meaning = "Exploratory mode — no prediction model trained. Focus is on understanding patterns."
    elif task_type == "clustering":
        n_clusters = ml.get("n_clusters", 0)
        body = f"The AI grouped your records into {n_clusters} distinct segments ({best_model}, {score_str})."
        headline = f"{n_clusters} segments found in your {domain} data"
        score_meaning = f"A silhouette score of {round(best_score,3)} indicates how cleanly separated these {n_clusters} segments are — closer to 1.0 is better, closer to 0 means segments overlap."
    elif task_type == "forecasting":
        body = f"The AI forecasted '{target}' forward using {best_model} ({score_str} on backtesting)."
        headline = f"Forecast ready for '{target}' — {best_model}"
        score_meaning = f"The backtest error ({score_str}) shows how far off the model's predictions were on recent known data — lower is more trustworthy."
    elif not best_model_data:
        body = f"Model evaluation for '{target}' is unavailable."
        headline = "Model evaluation unavailable"
        score_meaning = "This run did not produce hold-out model results."
    elif task_type == "regression" and best_score <= 0:
        # R2 <= 0 is no better than always predicting the mean -- the model has no real predictive skill,
        # so the fallback text must not claim it "reliably" forecasts anything (this was previously
        # unconditional and produced "...meaning it can reliably forecast future outcomes" even when the
        # score was negative).
        body = (
            f"The predictive model achieved {score_str}, which is at or below zero — it does not predict "
            f"'{target}' any better than always guessing the average value, so its forecasts should not be "
            f"relied on."
        )
        headline = f"No reliable model found for '{target}' ({score_str})"
        score_meaning = f"{score_str} means the model has no usable predictive skill for '{target}' on this data. Cross-validation: {cv_str}."
    else:
        body = (f"The model achieved {score_str} on the evaluation data. This describes observed model fit, "
                f"not prediction confidence or proof of reliable future forecasts. Independent validation is recommended.")
        headline = f"Model evaluation for '{target}': {score_str}"
        score_meaning = f"Hold-out {score_str}. Cross-validation: {cv_str}. Validate on more representative data."

    metrics = best_model_data.get("metrics", {})
    next_steps = list(quality_data.get("recommendations", []))
    if metrics.get("positive_recall") is not None and metrics["positive_recall"] < 0.60:
        next_steps.append("Review missed positive cases and assess their operational cost.")
    if ml.get("feature_availability"):
        next_steps.append("Use the restricted comparison for decisions made before the flagged features become available.")
    if not next_steps:
        next_steps.append("Check hold-out errors against the intended use case." if best_model_data
                          else "Review the data profile before selecting a prediction outcome.")
    fallback = {
        "report": (
            f"The uploaded dataset contains {rows:,} records. Composite data-quality score: {quality}/100. "
            f"Data completeness: {completeness_text}. " + body
        ),
        "headline": headline,
        "score_meaning": score_meaning,
        "top_recommendation": next_steps[0],
        "risk_flags": [item for item in next_steps if item.startswith("Review missed")],
        "confidence": "medium" if best_model_data and quality > 70 else "low",
        "should_i_worry": "Review data quality, hold-out performance, and timing assumptions before operational use.",
        "risk_score": risk_score,
        "next_steps": list(dict.fromkeys(next_steps)),
        "generated_by": "fallback",
        "degraded_mode": True,
    }

    if not api_key:
        return fallback

    if exploratory:
        prompt_template = _EDA_REPORT_PROMPT
    elif task_type == "clustering":
        prompt_template = _CLUSTERING_REPORT_PROMPT
    elif task_type == "forecasting":
        prompt_template = _FORECASTING_REPORT_PROMPT
    else:
        prompt_template = _REPORT_PROMPT
    prompt = prompt_template.format(
        filename=filename, quality=quality, completeness=completeness_text, rows=rows,
        task_type=task_type, target=target,
        best_model=best_model, score_str=score_str, cv_str=cv_str,
        insights=insights_text, actions=actions_text,
        domain=domain, warnings=warning_text,
    )

    try:
        from groq import Groq
        from agents import llmops

        client   = Groq(api_key=api_key, timeout=45.0, max_retries=1)
        with llmops.track_llm_call(job_id, user_email, "reporter") as ctx:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=1000,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": "Return JSON. The report field is qualitative interpretation only. Omit numerical claims, scores, target claims, causation and guarantees."},
                    {"role": "user",   "content": prompt},
                ],
            )
            ctx["response"] = response
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        interpretation = result.get("report", "")
        if (not isinstance(interpretation, str) or not interpretation.strip()
                or not qualitative_text_is_safe(interpretation)
                or re.search(r"quality|completeness|accuracy|precision|recall|f1|baseline|\btarget\b|\bmodel\b", interpretation, re.I)):
            return fallback
        return {**fallback, "report": fallback["report"] + "\n\nAI interpretation: " + interpretation.strip(),
                "generated_by": "groq", "degraded_mode": False}
    except Exception as e:
        print(f"[Reporter] Error: {e}")
        return fallback
