import numpy as np
from sklearn.ensemble import RandomForestClassifier
from agents import explainability


def test_shap_identifies_the_informative_features():
    rng = np.random.default_rng(0)
    X = rng.random((200, 5))
    # f1 (index 1) dominates the label, f0 contributes a bit, f2-f4 are noise
    y = (X[:, 1] * 2 + X[:, 0] > 1.5).astype(int)
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

    out = explainability.generate_shap_plots(model, X, ["f0", "f1", "f2", "f3", "f4"], "classification")

    assert out["available"] is True
    assert out["summary_plot"].startswith("data:image/png;base64,")
    assert out["waterfall_plot"].startswith("data:image/png;base64,")
    top_feature_names = [f["feature"] for f in out["top_features"][:2]]
    assert "f1" in top_feature_names


def test_handles_missing_model_gracefully():
    out = explainability.generate_shap_plots(None, None, [], "classification")
    assert out["available"] is False
    assert "reason" in out
