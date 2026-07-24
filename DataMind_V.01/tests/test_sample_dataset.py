import os
import pandas as pd
from agents import detective

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_data", "employee_attrition_sample.csv",
)


def test_sample_dataset_exists_and_loads():
    assert os.path.exists(SAMPLE_PATH), "sample_data/employee_attrition_sample.csv is missing"
    df = pd.read_csv(SAMPLE_PATH)
    assert len(df) > 100
    assert "attrition" in df.columns


def test_sample_dataset_has_a_realistic_attrition_rate():
    df = pd.read_csv(SAMPLE_PATH)
    rate = df["attrition"].mean()
    # Real-world attrition is rarely below 5% or above 40% — guards against
    # someone regenerating the sample with a broken signal/probability
    assert 0.05 < rate < 0.40


def test_sample_dataset_demonstrates_quality_center_features():
    """The sample is deliberately a little messy (inconsistent casing, one
    duplicate row, a few missing values) so the Data Quality Center has
    something real to show off in a demo — this guards against someone
    'cleaning it up' and accidentally making the demo boring."""
    det = detective.run_detective(SAMPLE_PATH)
    qd = det["quality_dimensions"]
    assert qd["consistency_issues"], "expected at least one consistency issue for the demo"
    assert qd["duplicate_rows"] >= 1
    assert 70 <= qd["composite_score"] < 100


def test_sample_dataset_produces_a_learnable_model():
    from agents import ml_engineer
    det = detective.run_detective(SAMPLE_PATH)
    ana = {"ml_recommendation": {"target_column": "attrition", "task_type": "classification"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)
    # Should beat a coin flip by a clear margin — confirms the generative signal
    # (job_satisfaction, overtime, tenure, income) is still present and learnable
    assert ml["best_score"] > 0.65


def test_api_sample_route_serves_the_file(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    import importlib
    import app as A
    importlib.reload(A)

    client = A.app.test_client()
    r = client.post("/api/guest")
    token = r.get_json()["token"]
    r2 = client.get("/api/sample", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    assert r2.content_type.startswith("text/csv")
    assert len(r2.data) > 1000
