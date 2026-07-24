from agents import detective, ml_engineer


def test_classification_picks_a_best_model(classification_csv):
    det = detective.run_detective(classification_csv)
    ana = {"ml_recommendation": {"target_column": "churn", "task_type": "classification"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)

    assert ml["task_type"] == "classification"
    assert ml["best_model"] is not None
    assert any(m["is_best"] for m in ml["models"])
    assert 0 <= ml["best_score"] <= 1


def test_regression_picks_a_best_model(regression_csv):
    det = detective.run_detective(regression_csv)
    ana = {"ml_recommendation": {"target_column": "revenue", "task_type": "regression"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)

    assert ml["task_type"] == "regression"
    assert ml["best_model"] is not None
    # A clean linear relationship (revenue = 2.5*sales + noise) should fit well
    assert ml["best_score"] > 0.8


def test_classification_includes_shap_explanation(classification_csv):
    det = detective.run_detective(classification_csv)
    ana = {"ml_recommendation": {"target_column": "churn", "task_type": "classification"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)

    shap_plots = ml["shap_plots"]
    # Either SHAP ran successfully, or it failed with a clear reason — never silently missing
    assert "available" in shap_plots
    if shap_plots["available"]:
        assert shap_plots["summary_plot"].startswith("data:image/png;base64,")
        assert len(shap_plots["summary_caption"]) > 0


def test_mode_override_routes_to_clustering(clustering_csv):
    det = detective.run_detective(clustering_csv)
    ml = ml_engineer.run_ml_engineer(det["df"], det, {}, mode_override="clustering")
    assert ml["task_type"] == "clustering"
    assert ml["n_clusters"] >= 2


def test_mode_override_routes_to_forecasting(timeseries_csv):
    det = detective.run_detective(timeseries_csv)
    ana = {"ml_recommendation": {"target_column": "sales"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana, mode_override="forecasting")
    assert ml["task_type"] == "forecasting"


def test_models_list_never_crashes_app_layer(classification_csv):
    """Regression test for a real bug: app.py's leaderboard builder used to assume
    every model dict had a 'score' key, which crashed for forecasting/clustering
    results that use 'mae' instead. This checks the contract models must satisfy."""
    det = detective.run_detective(classification_csv)
    ana = {"ml_recommendation": {"target_column": "churn", "task_type": "classification"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)
    for m in ml["models"]:
        assert "name" in m
        assert "is_best" in m
        assert ("score" in m) or ("mae" in m) or ("error" in m)
