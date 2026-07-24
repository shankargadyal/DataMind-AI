import pandas as pd
from agents import forecasting


def test_detects_upward_trend(timeseries_csv):
    df = pd.read_csv(timeseries_csv)
    det = {"date_cols": ["date"], "numeric_cols": ["sales"]}
    out = forecasting.run_forecasting(df, det, target_col="sales", date_col="date", steps=10)

    assert out["task_type"] == "forecasting"
    assert len(out["future_y"]) == 10
    # The series trends up ~100 -> 160 over 60 days; the forecast should continue upward
    assert out["future_y"][-1] > out["historical_y"][-1]


def test_confidence_interval_brackets_the_forecast(timeseries_csv):
    df = pd.read_csv(timeseries_csv)
    det = {"date_cols": ["date"], "numeric_cols": ["sales"]}
    out = forecasting.run_forecasting(df, det, target_col="sales", date_col="date", steps=10)
    for lo, mid, hi in zip(out["future_lower"], out["future_y"], out["future_upper"]):
        assert lo <= mid <= hi


def test_leaderboard_has_no_silent_failures(timeseries_csv):
    """Every model entry should explain itself — either a numeric error metric
    or an explicit 'error' string. No model should disappear silently."""
    df = pd.read_csv(timeseries_csv)
    det = {"date_cols": ["date"], "numeric_cols": ["sales"]}
    out = forecasting.run_forecasting(df, det, target_col="sales", date_col="date", steps=5)
    for m in out["models"]:
        assert ("mae" in m) or ("error" in m)


def test_too_short_series_errors_gracefully():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "y": [1, 2, 3, 4, 5],
    })
    out = forecasting.run_forecasting(df, {"date_cols": ["date"]}, target_col="y", date_col="date")
    assert "error" in out
