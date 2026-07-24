"""
Business-friendly utilities v3.0.
New in v3:
  - warning_card() — formats "Should I be worried?" flags for the UI
  - query_suggestion_cards() — formats AI query suggestions
  - risk_badge() — visual risk level badge
  - cv_score_meaning() — cross-validation score in plain English
  - exploratory_summary() — summary for EDA mode
"""


def business_health_score(quality_score: float, best_model_score: float, insights_count: int) -> dict:
    quality_norm  = quality_score / 100.0
    model_value   = best_model_score or 0
    model_norm    = model_value / 100.0 if model_value > 1 else model_value
    model_norm    = max(0.0, min(1.0, model_norm))
    insights_norm = min(1.0, insights_count / 10.0)
    raw_score     = (quality_norm * 0.4 + model_norm * 0.4 + insights_norm * 0.2) * 100
    health_score  = round(raw_score, 1)
    if health_score >= 80:
        level, color, advice = "Excellent", "#10B981", "Your data is reliable and actionable. You can confidently use these predictions."
    elif health_score >= 60:
        level, color, advice = "Good", "#F59E0B", "Data is usable. Consider improving quality or adding more historical records."
    else:
        level, color, advice = "Needs Attention", "#EF4444", "Data quality or predictive power is low. Clean missing values or collect more data before relying on predictions."
    return {"score": health_score, "level": level, "color": color, "advice": advice}


def translate_metric(technical_name: str, value: float, task_type: str = "regression") -> str:
    value = value / 100.0 if value and value > 1 else value
    if technical_name == "accuracy":
        pct = value * 100
        if pct >= 90:
            return f"Highly reliable — correctly predicts {pct:.1f}% of outcomes"
        elif pct >= 70:
            return f"Moderately reliable — correct {pct:.1f}% of the time"
        else:
            return f"Low reliability ({pct:.1f}%) — use predictions with caution"
    elif technical_name == "r2":
        pct = value * 100
        if value >= 0.8:
            return f"Excellent — the model explains {pct:.1f}% of what drives changes"
        elif value >= 0.5:
            return f"Moderate — explains {pct:.1f}% of the variation"
        else:
            return f"Weak — only explains {pct:.1f}% of changes; more data may help"
    elif technical_name == "f1":
        pct = value * 100
        return f"Balance score: {pct:.1f}% — {'good' if pct >= 70 else 'needs improvement'}"
    elif technical_name == "mae":
        return f"On average, predictions are off by {value:.2f} units"
    elif technical_name == "rmse":
        return f"Typical prediction error: ±{value:.2f} units"
    else:
        return f"{technical_name} = {value:.3f}"


def cv_score_meaning(cv_score: float, cv_std: float, task_type: str = "regression") -> str:
    """Translate cross-validation score to plain English."""
    if cv_score <= 0:
        return "Cross-validation score not available."
    if task_type == "classification":
        pct = cv_score * 100
        std_pct = cv_std * 100
        return (
            f"Tested on {3} different data splits, the model averaged {pct:.1f}% accuracy "
            f"(±{std_pct:.1f}%). {'This is consistent and reliable.' if cv_std < 0.05 else 'Some variability — collect more data for stability.'}"
        )
    else:
        if cv_score >= 0.8:
            return f"Cross-validation R²={cv_score:.3f} — strong and consistent across data splits."
        elif cv_score >= 0.5:
            return f"Cross-validation R²={cv_score:.3f} — moderate fit; predictions are directionally correct."
        else:
            return f"Cross-validation R²={cv_score:.3f} — weak fit. More data or better features may help."


def warning_card(flag: dict) -> dict:
    """Format a warning flag for the UI 'Should I be worried?' panel."""
    severity = flag.get("severity", "info")
    color_map = {"critical": "#EF4444", "warning": "#F59E0B", "info": "#3B82F6"}
    icon_map  = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
    return {
        "message":  flag.get("message", ""),
        "severity": severity,
        "column":   flag.get("column", ""),
        "color":    color_map.get(severity, "#3B82F6"),
        "icon":     icon_map.get(severity, "ℹ️"),
        "title":    "Critical Issue" if severity == "critical" else ("Warning" if severity == "warning" else "Note"),
    }


def query_suggestion_cards(suggestions: list) -> list:
    """Format query suggestions for the UI suggestion chips."""
    icons = ["🔍", "📊", "💡", "📈", "🎯", "❓"]
    return [
        {"text": s, "icon": icons[i % len(icons)]}
        for i, s in enumerate(suggestions[:6])
    ]


def risk_badge(risk_score: int) -> dict:
    """Return a UI-ready risk badge."""
    if risk_score < 30:
        return {"score": risk_score, "label": "Low Risk", "color": "#10B981", "emoji": "✅"}
    elif risk_score < 60:
        return {"score": risk_score, "label": "Moderate Risk", "color": "#F59E0B", "emoji": "⚠️"}
    else:
        return {"score": risk_score, "label": "High Risk", "color": "#EF4444", "emoji": "🚨"}


def format_future_prediction(pred: dict) -> dict:
    if not pred or not pred.get("available") or not pred.get("future_y"):
        return {"available": False, "message": "Not enough data for trend prediction."}
    steps           = len(pred["future_y"])
    last_historical = (pred.get("historical_y") or [0])[-1]
    first_future    = pred["future_y"][0]
    last_future     = pred["future_y"][-1]
    trend           = "increasing" if last_future > last_historical else "decreasing"
    change_pct      = ((last_future - last_historical) / max(abs(last_historical), 1e-6)) * 100
    direction       = pred.get("trend_direction", trend)
    return {
        "available":      True,
        "trend":          trend,
        "change_percent": round(change_pct, 1),
        "next_value":     round(first_future, 2),
        "final_value":    round(last_future, 2),
        "direction":      direction,
        "message": (
            f"Over the next {steps} periods, values are {trend} by {abs(change_pct):.1f}%. "
            f"Next period expected: {round(first_future, 2)}."
        ),
    }


def exploratory_summary(det: dict, ana: dict) -> dict:
    """Build a summary dict for exploratory mode."""
    top_corr  = det.get("top_correlations", [])
    insights  = ana.get("key_insights", [])
    domain    = ana.get("domain", "other")
    return {
        "mode": "exploratory",
        "top_correlations": top_corr[:5],
        "insight_count": len(insights),
        "domain": domain,
        "message": (
            "No prediction target was detected. DataMind is running in Exploratory Analysis Mode — "
            "revealing patterns, correlations, and trends in your data without training a prediction model."
        ),
    }


def domain_emoji(domain: str) -> str:
    mapping = {
        "retail":      "🛍️",
        "finance":     "💰",
        "healthcare":  "🏥",
        "marketing":   "📣",
        "operations":  "⚙️",
        "hr":          "👥",
        "logistics":   "🚚",
        "real_estate": "🏠",
        "education":   "📚",
        "other":       "📊",
    }
    return mapping.get(domain.lower(), "📊")


def score_badge(score: float, task_type: str = "regression") -> dict:
    if task_type == "exploratory":
        return {"display": "EDA", "label": "Exploratory Mode", "color": "#3B82F6", "grade": "—"}
    if task_type == "classification":
        pct = round(score * 100, 1) if score <= 1 else round(score, 1)
        display = f"{pct}%"
        if pct >= 85:
            return {"display": display, "label": "Highly Accurate", "color": "#10B981", "grade": "A"}
        elif pct >= 70:
            return {"display": display, "label": "Good", "color": "#F59E0B", "grade": "B"}
        else:
            return {"display": display, "label": "Needs Improvement", "color": "#EF4444", "grade": "C"}
    else:
        r2 = score if score <= 1 else score / 100
        display = f"R²={round(r2, 3)}"
        if r2 >= 0.8:
            return {"display": display, "label": "Strong Predictor", "color": "#10B981", "grade": "A"}
        elif r2 >= 0.5:
            return {"display": display, "label": "Moderate", "color": "#F59E0B", "grade": "B"}
        else:
            return {"display": display, "label": "Weak Predictor", "color": "#EF4444", "grade": "C"}