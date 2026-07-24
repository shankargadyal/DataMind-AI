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


_REPORT_PROMPT = """You are a trusted business advisor writing a report for a non-technical business owner.
Based on the analysis results below, write a clear, friendly executive report.

Dataset: {filename}
Data Quality: {quality}%
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
Data Quality: {quality}%
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
Data Quality: {quality}%
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
Data Quality: {quality}%
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
    quality     = det.get("quality_score", 0)
    rows        = det.get("original_shape", [0])[0]
    task_type   = ml.get("task_type", "regression")
    target      = ml.get("target_column", "outcome")
    best_model  = ml.get("best_model", "AI model")
    best_score  = ml.get("best_score", 0)
    domain      = ana.get("domain", "business")
    exploratory = ml.get("exploratory_mode", False)

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
    cv_str   = f"{round(cv_score * 100, 1)}% ± {round(cv_std * 100, 1)}%" if cv_score else "not available"

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
    else:
        body = f"The predictive model achieved {score_str}, meaning it can reliably forecast future outcomes."
        headline = f"Your data is ready — AI model achieves {score_str}"
        score_meaning = f"The model correctly predicts '{target}' with {score_str}. Cross-validation: {cv_str}."

    fallback = {
        "report": (
            f"Your {filename} dataset contains {rows:,} records with a data quality score of {quality}%. "
            f"The AI has analysed your data and identified key patterns to help your business. " + body
        ),
        "headline": headline,
        "score_meaning": score_meaning,
        "top_recommendation": (
            (ana.get("actions", [{}])[0].get("action") if ana.get("actions") else "Review the key insights above.")
        ),
        "risk_flags": [w.get("message", "") for w in ana.get("warning_flags", [])[:3]],
        "confidence": "high" if best_score > 0.8 and quality > 85 else ("medium" if best_score > 0.6 else "low"),
        "should_i_worry": (
            "No — your data quality is solid and the model performs well. You can act on these insights confidently."
            if risk_score < 30
            else f"Yes — there are {warning_count} flag(s) to address before fully trusting predictions. See risk flags."
        ),
        "risk_score": risk_score,
        "next_steps": [
            a.get("action", "") for a in ana.get("actions", [])[:3]
        ] or ["Review insights", "Check warning flags", "Export the report"],
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
        filename=filename, quality=quality, rows=rows,
        task_type=task_type, target=target,
        best_model=best_model, score_str=score_str, cv_str=cv_str,
        insights=insights_text, actions=actions_text,
        domain=domain, warnings=warning_text,
    )

    try:
        from groq import Groq
        from agents import llmops

        client   = Groq(api_key=api_key)
        with llmops.track_llm_call(job_id, user_email, "reporter") as ctx:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=1000,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": "You are a business advisor. Return only valid JSON, no markdown."},
                    {"role": "user",   "content": prompt},
                ],
            )
            ctx["response"] = response
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        for k in fallback:
            if k not in result:
                result[k] = fallback[k]
        # Always derive risk_score from our logic (more reliable than LLM)
        result["risk_score"] = risk_score
        result["generated_by"] = "groq"
        result["degraded_mode"] = False
        return result
    except Exception as e:
        print(f"[Reporter] Error: {e}")
        return fallback
