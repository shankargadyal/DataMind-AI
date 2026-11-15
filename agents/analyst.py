"""
Agent 2: Analyst v3.0 — Groq LLaMA insight generation.
New in v3:
  - "Should I be worried?" warning system with severity levels
  - Plain-English metric cards (no jargon)
  - Smart query suggestions based on dataset domain
  - Outlier-aware insights
  - Exploratory mode support (no ML target needed)
"""
import json
import re


_INSIGHT_PROMPT = """You are a trusted business advisor helping a non-technical business owner understand their data.
Analyse the dataset profile below and return ONLY valid JSON (no markdown, no extra text).

Dataset profile:
{profile}

User question: {query}

Return this exact JSON structure:
{{
  "key_insights": [
    {{"title": "Short title", "description": "Plain English 1-2 sentence insight", "type": "positive|warning|info", "impact": "high|medium|low"}}
  ],
  "ml_recommendation": {{
    "target_column": "best column to predict",
    "task_type": "classification|regression",
    "reason": "why this target makes sense",
    "business_value": "what predicting this helps the business do"
  }},
  "actions": [
    {{"priority": 1, "action": "Specific actionable recommendation", "expected_outcome": "what it will achieve"}}
  ],
  "dataset_summary": "2-3 sentence plain-English overview of what this data represents",
  "data_health": "excellent|good|fair|poor",
  "domain": "retail|finance|healthcare|marketing|operations|hr|logistics|real_estate|education|other",
  "warning_flags": [
    {{"message": "Plain-English warning the business owner must see", "severity": "critical|warning|info", "column": "optional column name"}}
  ],
  "metric_cards": [
    {{"label": "Short metric name", "value": "human-readable value", "meaning": "plain English explanation of what this means for the business", "trend": "up|down|neutral"}}
  ],
  "query_suggestions": [
    "What are the top factors driving sales?",
    "Which products have the highest return rates?"
  ]
}}

Rules:
- key_insights: 4-6 insights, business-friendly language (no jargon)
- actions: 3-5 prioritised actions
- target_column must be one of the actual column names listed
- warning_flags: flag issues like >10% missing data, extreme outliers, low data quality, class imbalance
- metric_cards: 3-5 cards translating raw stats into business meaning
- query_suggestions: 4-6 natural questions a business owner would ask about THIS specific dataset
- "Should I be worried?" — use severity "critical" only for genuine data problems
- Use ONLY the column names from the profile, never invent new ones
"""


_EDA_PROMPT = """You are a data analyst in exploratory mode — there is no clear prediction target.
Analyse this dataset profile and return ONLY valid JSON.

Dataset profile:
{profile}

User question: {query}

Return this exact JSON:
{{
  "key_insights": [
    {{"title": "Short title", "description": "Plain English insight about patterns, trends, or anomalies", "type": "positive|warning|info", "impact": "high|medium|low"}}
  ],
  "ml_recommendation": {{
    "target_column": "",
    "task_type": "exploratory",
    "reason": "No clear prediction target found — running exploratory analysis",
    "business_value": "Understanding patterns and distributions in the data"
  }},
  "actions": [
    {{"priority": 1, "action": "Actionable insight based on EDA", "expected_outcome": "expected business benefit"}}
  ],
  "dataset_summary": "2-3 sentence overview of what this data represents",
  "data_health": "excellent|good|fair|poor",
  "domain": "retail|finance|healthcare|marketing|operations|hr|other",
  "warning_flags": [
    {{"message": "Any data quality issues", "severity": "critical|warning|info", "column": ""}}
  ],
  "metric_cards": [
    {{"label": "Short metric name", "value": "human-readable value", "meaning": "plain English explanation", "trend": "up|down|neutral"}}
  ],
  "query_suggestions": [
    "What patterns exist in this data?",
    "Are there any unusual values I should know about?"
  ]
}}
"""


def _build_profile(det: dict) -> str:
    cols = det.get("columns", [])
    numeric_cols = det.get("numeric_cols", [])
    categorical_cols = det.get("categorical_cols", [])
    shape = det.get("original_shape", [0, 0])
    quality = det.get("quality_score", 0)
    stats = det.get("column_stats", {})
    is_ts = det.get("is_timeseries", False)
    outliers = det.get("outlier_summary", {})
    top_corr = det.get("top_correlations", [])

    lines = [
        f"Rows: {shape[0]}, Columns: {shape[1]}",
        f"Data quality: {quality}%",
        f"Time-series: {is_ts}",
        f"Suggested target: {det.get('suggested_target', 'none')}",
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:20])}",
        f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:20])}",
        "Column statistics:",
    ]
    for col in list(numeric_cols)[:15]:
        s = stats.get(col, {})
        out = outliers.get(col, {})
        out_str = f", outliers={out.get('count', 0)} ({out.get('pct', 0)}%)" if out else ""
        lines.append(
            f"  {col}: mean={s.get('mean','?')}, min={s.get('min','?')}, "
            f"max={s.get('max','?')}, unique={s.get('unique','?')}{out_str}"
        )
    for col in list(categorical_cols)[:10]:
        s = stats.get(col, {})
        lines.append(f"  {col}: {s.get('unique','?')} unique values, top='{s.get('top','?')}'")

    if top_corr:
        lines.append("Top correlations:")
        for p in top_corr[:5]:
            lines.append(f"  {p['col1']} ↔ {p['col2']}: r={p['r']}")

    return "\n".join(lines)


def run_analyst(det: dict, query: str, api_key: str, job_id: str = None, user_email: str = None) -> dict:
    profile = _build_profile(det)
    exploratory = det.get("exploratory_mode", False)
    suggested_target = det.get("suggested_target") or ""
    lower_target = suggested_target.lower()
    fallback_task = "classification" if lower_target in {"target", "label", "outcome", "diagnosis", "class", "status"} else "regression"
    prompt_template = _EDA_PROMPT if exploratory else _INSIGHT_PROMPT
    prompt = prompt_template.format(profile=profile, query=query)

    # Deterministic, evidence-grounded fallback when the language model is unavailable.
    shape = det.get("original_shape") or [0, 0]
    row_count, col_count = (list(shape) + [0, 0])[:2]
    col_stats = det.get("column_stats") or {}
    numeric = det.get("numeric_cols") or []
    categorical = det.get("categorical_cols") or []
    quality = det.get("quality_score", 0)
    evidence = [{"title": "Dataset profile", "description":
                 f"The uploaded dataset contains {row_count} records and {col_count} columns "
                 f"({len(numeric)} numeric, {len(categorical)} categorical).",
                 "type": "info", "impact": "medium"}]
    missing = [(name, st.get("missing", 0)) for name, st in col_stats.items()
               if isinstance(st, dict) and isinstance(st.get("missing", 0), (int, float)) and st.get("missing", 0) > 0]
    if missing:
        name, count = max(missing, key=lambda pair: pair[1])
        evidence.append({"title": "Missing data detected", "description":
                         f"Column '{name}' has {int(count)} missing values. Review the cleaning log and check whether missingness is systematic.",
                         "type": "warning", "impact": "high"})
    else:
        evidence.append({"title": "No missing values reported", "description":
                         "The profiler did not report missing values in its column statistics. This does not rule out invalid or mislabeled data.",
                         "type": "info", "impact": "low"})
    for name in numeric[:2]:
        st = col_stats.get(name) or {}
        lo, hi = st.get("min"), st.get("max")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            evidence.append({"title": f"Range of {name}", "description":
                             f"Observed {name} values range from {lo:g} to {hi:g}. Investigate extreme or unexpected values before relying on the model.",
                             "type": "info", "impact": "medium"})
    if row_count < 100:
        evidence.append({"title": "Small dataset", "description":
                         f"Only {row_count} rows were supplied. Predictive evaluation may be unstable; gather more representative observations.",
                         "type": "warning", "impact": "high"})
    fallback = {
        "key_insights": evidence,
        "ml_recommendation": {
            "target_column": suggested_target or (numeric[0] if numeric else ""),
            "task_type": "exploratory" if exploratory else fallback_task,
            "reason": "Auto-selected from available columns; confirm that this target matches your goal",
            "business_value": "Explore associations; prediction requires validation",
        },
        "actions": [
            {"priority": 1, "action": "Verify the selected prediction target and column meanings", "expected_outcome": "Avoid modeling the wrong outcome"},
            {"priority": 2, "action": "Review missing, invalid, and unusual values", "expected_outcome": "Understand potential data-quality limitations"},
            {"priority": 3, "action": "Validate results on additional representative records", "expected_outcome": "Assess generalization beyond this dataset"},
        ],
        "dataset_summary": (f"Dataset contains {row_count} rows and {col_count} columns. "
                            "The language model was unavailable, so these are deterministic profile observations, not AI-inferred business findings."),
        "data_health": "good" if quality >= 85 else ("fair" if quality >= 65 else "poor"),
        "domain": "other",
        "warning_flags": ([{"message": f"Small dataset: {row_count} rows; evaluation may be unstable.", "severity": "warning", "column": ""}] if row_count < 100 else []),
        "metric_cards": [
            {"label": "Data Quality", "value": f"{quality}%", "meaning": "Profiler quality score; not a model accuracy score", "trend": "neutral"},
            {"label": "Rows", "value": str(row_count), "meaning": "Observations in the uploaded dataset", "trend": "neutral"},
        ],
        "query_suggestions": [
            "Which columns have missing values?",
            "What are the observed ranges of the numeric columns?",
            "What additional data is needed to validate the model?",
        ],
        "generated_by": "fallback",
        "degraded_mode": True,
    }

    if not api_key:
        return fallback

    try:
        from groq import Groq
        from agents import llmops

        client = Groq(api_key=api_key, timeout=45.0, max_retries=1)
        with llmops.track_llm_call(job_id, user_email, "analyst") as ctx:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Always respond with valid JSON only, no markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            ctx["response"] = response
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        # Ensure required keys exist
        for k in fallback:
            if k not in result:
                result[k] = fallback[k]
        result["generated_by"] = "groq"
        result["degraded_mode"] = False
        return result
    except Exception as e:
        print(f"[Analyst] Error: {e}")
        return fallback
