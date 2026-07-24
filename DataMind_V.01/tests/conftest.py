"""
Shared fixtures for the DataMind test suite.

These build small, deterministic synthetic datasets on disk (tmp_path)
rather than depending on any real uploaded CSV, so the suite runs the
same way on a contributor's laptop or in CI with zero external data.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def classification_csv(tmp_path, rng):
    """A clean, linearly-separable-ish binary classification dataset."""
    n = 300
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "income": rng.normal(50000, 15000, n),
        "tenure_years": rng.uniform(0, 10, n),
    })
    df["churn"] = (df["income"] < 45000).astype(int)
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def regression_csv(tmp_path, rng):
    n = 300
    df = pd.DataFrame({
        "inventory": rng.uniform(10, 500, n),
        "sales": rng.uniform(100, 1000, n),
    })
    df["revenue"] = df["sales"] * 2.5 + rng.normal(0, 50, n)
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def clustering_csv(tmp_path, rng):
    """Three well-separated Gaussian blobs — clustering algorithms should
    have no trouble finding exactly 3 groups, which is what the assertions
    in test_clustering.py check for."""
    a = rng.normal(0, 1, (60, 3))
    b = rng.normal(8, 1, (60, 3)) + [0, 0, 0]
    c = rng.normal(0, 1, (60, 3)) + [10, 10, 0]
    df = pd.DataFrame(np.vstack([a, b, c]), columns=["f1", "f2", "f3"])
    path = tmp_path / "clustering.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def timeseries_csv(tmp_path, rng):
    """A clear upward trend so forecasting models have something
    unambiguous to pick up on."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    trend = np.linspace(100, 160, 60)
    noise = rng.normal(0, 2, 60)
    df = pd.DataFrame({"date": dates, "sales": trend + noise})
    path = tmp_path / "timeseries.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def messy_csv(tmp_path, rng):
    """Deliberately dirty data for Data Quality Center tests: negative ages,
    inconsistent casing, a duplicate row, and some missing values."""
    ages = list(rng.integers(18, 70, 95)) + [-5, -3, 200, 250, 300]
    countries = (["USA", "usa", " USA ", "India", "india"] * 20)[:100]
    df = pd.DataFrame({
        "age": ages,
        "country": countries,
        "income": rng.normal(50000, 15000, 100),
    })
    df.loc[0:5, "income"] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # exact duplicate row
    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)
    return str(path)
