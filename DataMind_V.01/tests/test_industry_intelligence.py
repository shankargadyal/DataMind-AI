import pandas as pd
from agents import detective, industry_intelligence


def test_hr_kpis_matched_by_column_name(tmp_path, rng):
    df = pd.DataFrame({
        "tenure_years": rng.uniform(0, 10, 100),
        "attrition": rng.integers(0, 2, 100),
        "performance_rating": rng.integers(1, 5, 100),
    })
    path = tmp_path / "hr.csv"
    df.to_csv(path, index=False)
    det = detective.run_detective(str(path))

    out = industry_intelligence.get_industry_insights(
        "hr", det, {"ml_recommendation": {"target_column": "attrition"}}, {"future_prediction": {}}
    )
    assert out["available"] is True
    matched_cols = [k["matched_column"] for k in out["matched_kpis"]]
    assert "tenure_years" in matched_cols
    assert "attrition" in matched_cols
    assert len(out["recommendations"]) >= 1


def test_unknown_industry_returns_unavailable(classification_csv):
    det = detective.run_detective(classification_csv)
    out = industry_intelligence.get_industry_insights("not_a_real_industry", det, {}, {})
    assert out["available"] is False


def test_no_matching_columns_still_returns_a_recommendation(classification_csv):
    det = detective.run_detective(classification_csv)
    out = industry_intelligence.get_industry_insights("manufacturing", det, {}, {})
    assert out["available"] is True
    assert len(out["recommendations"]) >= 1
