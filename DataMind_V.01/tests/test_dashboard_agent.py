import json
from agents import detective, ml_engineer, dashboard_agent


def test_dashboard_payload_is_json_serializable(classification_csv):
    det = detective.run_detective(classification_csv)
    det["filename"] = "classification.csv"
    ana = {"ml_recommendation": {"target_column": "churn", "task_type": "classification"},
           "key_insights": [], "actions": [], "warning_flags": []}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)
    rep = {"headline": "test", "report": "test body", "should_i_worry": "no", "risk_score": 10, "next_steps": []}

    out = dashboard_agent.run(det, ana, ml, rep, "test query")
    json.dumps(out["payload"])  # raises if anything non-serializable leaked through
    assert "<html" in out["html"].lower()
    assert "Plotly" in out["html"]


def test_no_model_objects_leak_into_payload(classification_csv):
    """Regression test: ml_engineer's raw output includes private model/scaler
    objects (the trained estimator, the scaler, etc). dashboard_agent must
    strip every '_'-prefixed key before it reaches the JSON payload."""
    det = detective.run_detective(classification_csv)
    ana = {"ml_recommendation": {"target_column": "churn", "task_type": "classification"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)
    assert "_model_obj" in ml  # sanity check the raw output really does carry it

    out = dashboard_agent.run(det, ana, ml, {}, "q")
    assert "_model_obj" not in out["payload"]["models"]
    flat = json.dumps(out["payload"])
    assert "_model_obj" not in flat


def test_clustering_dashboard_includes_pca_projection(clustering_csv):
    import pandas as pd
    from agents import clustering
    df = pd.read_csv(clustering_csv)
    det = {"numeric_cols": ["f1", "f2", "f3"], "quality_dimensions": {}, "original_shape": [len(df), 3],
           "distribution_data": [], "correlation": {}, "categorical_cols": [], "column_stats": {}}
    ml = clustering.run_clustering(df, det)
    out = dashboard_agent.run(det, {}, ml, {}, "q")
    assert out["payload"]["clustering"] is not None
    assert len(out["payload"]["clustering"]["pca_projection"]) > 0


def test_categorical_chart_data_reaches_the_payload(messy_csv):
    """Regression test: dashboard_agent computed top_values for categorical
    columns into the payload but never rendered them — the chart card was
    silently empty. This checks the data a chart would need actually exists
    in the payload that ships to the browser."""
    det = detective.run_detective(messy_csv)
    det["filename"] = "messy.csv"
    ana = {"ml_recommendation": {}, "key_insights": [], "actions": [], "warning_flags": []}
    ml = {"task_type": "", "models": [], "feature_importances": [], "shap_plots": {"available": False}}
    out = dashboard_agent.run(det, ana, ml, {}, "q")
    cat = out["payload"]["eda"]["categorical"]
    assert len(cat) >= 1
    assert "top_values" in cat[0]
    assert len(cat[0]["top_values"]) >= 1
    assert "value" in cat[0]["top_values"][0] and "count" in cat[0]["top_values"][0]
