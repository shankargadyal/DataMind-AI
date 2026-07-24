from agents import detective


def test_basic_shape_and_columns(classification_csv):
    det = detective.run_detective(classification_csv)
    assert det["original_shape"][0] == 300
    assert set(["age", "income", "tenure_years", "churn"]).issubset(det["columns"])


def test_numeric_and_categorical_split(classification_csv):
    det = detective.run_detective(classification_csv)
    assert "age" in det["numeric_cols"]
    assert "income" in det["numeric_cols"]


def test_date_column_detected(timeseries_csv):
    det = detective.run_detective(timeseries_csv)
    assert "date" in det["date_cols"]
    assert det["is_timeseries"] is True


def test_quality_dimensions_present_and_bounded(classification_csv):
    det = detective.run_detective(classification_csv)
    qd = det["quality_dimensions"]
    assert 0 <= qd["composite_score"] <= 100
    for dim in ("completeness", "accuracy", "consistency", "validity", "uniqueness"):
        assert 0 <= qd["dimensions"][dim] <= 100
    assert qd["grade"] in ("Excellent", "Good", "Fair", "Poor")


def test_clean_data_scores_high(classification_csv):
    det = detective.run_detective(classification_csv)
    assert det["quality_dimensions"]["composite_score"] >= 90


def test_messy_data_catches_real_issues(messy_csv):
    det = detective.run_detective(messy_csv)
    qd = det["quality_dimensions"]

    # Negative ages should be flagged as a validity problem
    assert "age" in qd["validity_issues"]
    assert qd["validity_issues"]["age"]["count"] >= 2

    # Mixed-case "USA"/"usa"/" USA " should be flagged as a consistency problem
    assert "country" in qd["consistency_issues"]

    # The duplicated row should be caught
    assert qd["duplicate_rows"] >= 1

    # Composite score should be measurably lower than a clean dataset's
    assert qd["composite_score"] < 95

    # At least one recommendation should be generated
    assert len(qd["recommendations"]) >= 1


def test_missing_values_get_filled(messy_csv):
    det = detective.run_detective(messy_csv)
    assert det["missing_after"] == 0
    assert det["missing_before"] > 0
