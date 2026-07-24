"""
Agent 4: Explainability Engine — SHAP-based model interpretation.

Produces two visuals as base64-encoded PNGs (so they drop straight into
HTML dashboards or get embedded in the PDF report with no extra file
handling):
  - Global summary plot: which features matter most, across all predictions
  - Local waterfall plot: how one specific prediction was built up feature by feature

Both come with a plain-English caption so a non-technical reader gets the
point without needing to read the chart.
"""
import io
import base64
import numpy as np
import warnings

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend — required on servers with no display
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _caption_for_summary(top_features: list, task_type: str) -> str:
    if not top_features:
        return "Feature impact could not be determined for this model."
    names = ", ".join(f["feature"] for f in top_features[:3])
    verb = "predicting the outcome" if task_type == "classification" else "driving the predicted value"
    return f"{names} have the strongest influence on {verb}. Features further down the chart matter much less."


def _caption_for_waterfall(feature_contribs: list, base_value: float, final_value: float) -> str:
    if not feature_contribs:
        return "No individual prediction breakdown is available."
    pushers_up = [f for f in feature_contribs if f["contribution"] > 0]
    pushers_down = [f for f in feature_contribs if f["contribution"] < 0]
    parts = []
    if pushers_up:
        parts.append(f"{pushers_up[0]['feature']} pushed this prediction higher")
    if pushers_down:
        parts.append(f"{pushers_down[0]['feature']} pulled it lower")
    detail = " while ".join(parts) if parts else "Each feature made a small, roughly offsetting contribution"
    return f"Starting from a baseline of {round(base_value, 3)}, {detail}, landing on a final prediction of {round(final_value, 3)}."


def generate_shap_plots(model, X_sample: np.ndarray, feature_cols: list, task_type: str,
                         instance_idx: int = 0) -> dict:
    """
    model: trained sklearn-compatible estimator
    X_sample: scaled feature matrix (numpy array), already aligned to feature_cols
    feature_cols: list of feature names matching X_sample columns
    task_type: "classification" or "regression" (clustering/forecasting skip this)
    instance_idx: which row to use for the local waterfall explanation
    """
    if not HAS_SHAP:
        return {
            "available": False,
            "reason": "shap not installed — run: pip install shap",
        }
    if not HAS_MATPLOTLIB:
        return {
            "available": False,
            "reason": "matplotlib not installed — run: pip install matplotlib",
        }
    if model is None or X_sample is None or len(X_sample) == 0:
        return {"available": False, "reason": "No trained model or sample data available"}

    try:
        sample = X_sample[: min(150, len(X_sample))]
        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample)

        vals = shap_values.values
        if vals.ndim == 3:
            # multiclass classification -> average magnitude across classes
            vals_for_summary = np.abs(vals).mean(axis=2)
        else:
            vals_for_summary = vals

        # ── Global summary plot ─────────────────────────────────────────
        mean_abs = np.abs(vals_for_summary).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_n = min(12, len(feature_cols))
        top_idx = order[:top_n]

        fig1, ax1 = plt.subplots(figsize=(7, 4.2))
        y_pos = np.arange(top_n)
        ax1.barh(y_pos, mean_abs[top_idx][::-1], color="#D98E2B")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([feature_cols[i] for i in top_idx][::-1], fontsize=9)
        ax1.set_xlabel("Mean |SHAP value| (impact on prediction)", fontsize=9)
        ax1.set_title("What drives this model's predictions", fontsize=11, fontweight="bold")
        ax1.spines[["top", "right"]].set_visible(False)
        summary_b64 = _fig_to_base64(fig1)

        top_features = [
            {"feature": feature_cols[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in top_idx
        ]

        # ── Local waterfall plot for one instance ───────────────────────
        idx = min(instance_idx, len(sample) - 1)
        if vals.ndim == 3:
            inst_vals = vals[idx, :, 0]  # first class for multiclass; still illustrative
            base_value = float(np.atleast_1d(shap_values.base_values[idx])[0])
        else:
            inst_vals = vals[idx]
            base_value = float(np.atleast_1d(shap_values.base_values[idx])[0]) if hasattr(shap_values, "base_values") else 0.0

        order2 = np.argsort(np.abs(inst_vals))[::-1][:10]
        contribs = [
            {"feature": feature_cols[i], "contribution": round(float(inst_vals[i]), 6)}
            for i in order2
        ]
        final_value = base_value + float(np.sum(inst_vals))

        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        colors = ["#2F9E6E" if c["contribution"] >= 0 else "#D6455F" for c in contribs][::-1]
        ax2.barh(np.arange(len(contribs)), [c["contribution"] for c in contribs][::-1], color=colors)
        ax2.set_yticks(np.arange(len(contribs)))
        ax2.set_yticklabels([c["feature"] for c in contribs][::-1], fontsize=9)
        ax2.axvline(0, color="#8B93A3", linewidth=0.8)
        ax2.set_xlabel("Contribution to this single prediction", fontsize=9)
        ax2.set_title(f"Why this one prediction came out as it did (base={round(base_value,3)})",
                       fontsize=10.5, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        waterfall_b64 = _fig_to_base64(fig2)

        return {
            "available": True,
            "summary_plot": summary_b64,
            "summary_caption": _caption_for_summary(top_features, task_type),
            "top_features": top_features,
            "waterfall_plot": waterfall_b64,
            "waterfall_caption": _caption_for_waterfall(contribs, base_value, final_value),
            "waterfall_instance": int(idx),
            "waterfall_base_value": round(base_value, 4),
            "waterfall_final_value": round(final_value, 4),
        }
    except Exception as e:
        return {"available": False, "reason": f"SHAP computation failed: {e}"}
