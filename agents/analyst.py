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

    fallback = {
        "key_insights": [
            {
                "title": "Data Loaded Successfully",
                "description": "Your dataset has been cleaned and is ready for analysis.",
                "type": "positive",
                "impact": "high",
            }
        ],
        "ml_recommendation": {
            "target_column": suggested_target or (det.get("numeric_cols", [""])[0] if det.get("numeric_cols") else ""),
            "task_type": "exploratory" if exploratory else fallback_task,
            "reason": "Auto-selected from available columns",
            "business_value": "Identify key drivers and forecast future values",
        },
        "actions": [
            {"priority": 1, "action": "Review the data quality score", "expected_outcome": "Better predictions"}
        ],
        "dataset_summary": (
            f"Dataset with {det.get('original_shape', [0, 0])[0]} rows "
            f"and {det.get('original_shape', [0, 0])[1]} columns."
        ),
        "data_health": "good",
        "domain": "other",
        "warning_flags": [],
        "metric_cards": [
            {
                "label": "Data Quality",
                "value": f"{det.get('quality_score', 0)}%",
                "meaning": "How complete and reliable your data is",
                "trend": "up" if det.get("quality_score", 0) > 80 else "down",
            }
        ],
        "query_suggestions": [
            "What are the most important patterns in this data?",
            "Which columns have the most missing values?",
            "What should I focus on to improve results?",
        ],
        "generated_by": "fallback",
        "degraded_mode": True,
    }

    if not api_key:
        return fallback

    try:
        from groq import Groq
        from agents import llmops

        client = Groq(api_key=api_key)
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
