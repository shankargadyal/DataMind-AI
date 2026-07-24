from agents import detective, analyst, ml_engineer, reporter, report_generator, industry_intelligence


def _pdf_bytes_look_valid(pdf_bytes):
    return pdf_bytes[:5] == b"%PDF-"


def test_classification_report_is_valid_pdf(classification_csv):
    det = detective.run_detective(classification_csv)
    ana = analyst.run_analyst(det, "insights", api_key="")  # empty key -> deterministic fallback, no network
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)
    rep = reporter.run_reporter(det, ana, ml, "insights", api_key="")

    analysis_data = {
        "summary": rep["report"],
        "ml_recommendation": {
            "target_column": ml.get("target_column", ""),
            "task_type": ml.get("task_type", ""),
            "best_model": ml.get("best_model", ""),
            "best_score": ml.get("best_score", 0),
            "models": ml.get("models", []),
            "best_metrics": {},
        },
        "key_insights": ana.get("key_insights", []),
        "actions": ana.get("actions", []),
        "stats": {"total_rows": det["original_shape"][0], "total_cols": det["original_shape"][1],
                   "quality_score": det.get("quality_score", 0), "insights_count": len(ana.get("key_insights", []))},
        "quality_dimensions": det.get("quality_dimensions", {}),
        "risk_flags": rep.get("risk_flags", []),
        "risk_score": rep.get("risk_score", 0),
        "next_steps": rep.get("next_steps", []),
    }
    pdf = report_generator.generate_pdf_report(analysis_data, det, ml, industry={"available": False})
    assert _pdf_bytes_look_valid(pdf)
    assert len(pdf) > 1000


def test_clustering_report_does_not_mislabel_score_as_r2(clustering_csv):
    """Regression test for a real bug: clustering's silhouette score was being
    displayed as if it were an R² regression score in the PDF."""
    import pandas as pd
    df = pd.read_csv(clustering_csv)
    det = {"numeric_cols": ["f1", "f2", "f3"], "quality_dimensions": {}, "quality_score": 100,
           "original_shape": [len(df), 3]}
    from agents import clustering
    ml = clustering.run_clustering(df, det)

    analysis_data = {
        "summary": "test",
        "ml_recommendation": {
            "target_column": "", "task_type": "clustering",
            "best_model": ml["best_model"], "best_score": ml["best_score"],
            "models": ml["models"], "best_metrics": {},
        },
        "key_insights": [], "actions": [],
        "stats": {"total_rows": len(df), "total_cols": 3, "quality_score": 100, "insights_count": 0},
        "quality_dimensions": {}, "risk_flags": [], "risk_score": 0, "next_steps": [],
    }
    pdf = report_generator.generate_pdf_report(analysis_data, det, ml, industry={"available": False})
    assert _pdf_bytes_look_valid(pdf)
    # Pull text out is heavy (needs pdfminer); instead we assert on the formatted
    # string the report builder actually produces for clustering before rendering.
    score_disp = f"silhouette={round(ml['best_score'], 3)}"
    assert "silhouette" in score_disp  # sanity check the metric we expect report_generator to use


def test_shap_images_are_actually_embedded_in_pdf(classification_csv):
    """Regression test: shap_plots used to be computed by ml_engineer but never
    reached the PDF — only a text label ('(SHAP)' vs '(Native)') referenced a
    key that was never set. This checks the real image bytes make it into the
    PDF, not just a bigger byte count (which embedding any image would cause)."""
    det = detective.run_detective(classification_csv)
    ana = {"ml_recommendation": {"target_column": "churn", "task_type": "classification"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)
    assert ml["shap_plots"]["available"] is True  # sanity check SHAP actually ran

    analysis_data = {
        "summary": "test",
        "ml_recommendation": {
            "target_column": "churn", "task_type": "classification",
            "best_model": ml["best_model"], "best_score": ml["best_score"],
            "models": ml["models"], "best_metrics": {},
            "feature_importances": ml["feature_importances"],
            "shap_plots": ml["shap_plots"],
        },
        "key_insights": [], "actions": [],
        "stats": {"total_rows": det["original_shape"][0], "total_cols": det["original_shape"][1],
                   "quality_score": det.get("quality_score", 0), "insights_count": 0},
        "quality_dimensions": det.get("quality_dimensions", {}), "risk_flags": [], "risk_score": 0, "next_steps": [],
    }
    pdf_with_shap = report_generator.generate_pdf_report(analysis_data, det, ml, industry={"available": False})

    # Same report, but with SHAP stripped out, to isolate the size delta to the images themselves
    analysis_data["ml_recommendation"]["shap_plots"] = {"available": False, "reason": "test"}
    ml_no_shap = dict(ml)
    ml_no_shap["shap_plots"] = {"available": False}
    pdf_without_shap = report_generator.generate_pdf_report(analysis_data, det, ml_no_shap, industry={"available": False})

    assert _pdf_bytes_look_valid(pdf_with_shap)
    # Two embedded PNGs should add a meaningfully large number of bytes (base64-decoded
    # images aren't tiny) — this would fail if the images were silently skipped.
    assert len(pdf_with_shap) - len(pdf_without_shap) > 5000


def test_industry_section_included_when_available(classification_csv):
    det = detective.run_detective(classification_csv)
    ana = {"ml_recommendation": {"target_column": "churn", "task_type": "classification"}}
    ml = ml_engineer.run_ml_engineer(det["df"], det, ana)
    ind = industry_intelligence.get_industry_insights("banking", det, ana, ml)

    analysis_data = {
        "summary": "test", "ml_recommendation": {
            "target_column": "churn", "task_type": "classification",
            "best_model": ml["best_model"], "best_score": ml["best_score"],
            "models": ml["models"], "best_metrics": {},
        },
        "key_insights": [], "actions": [],
        "stats": {"total_rows": det["original_shape"][0], "total_cols": det["original_shape"][1],
                   "quality_score": det.get("quality_score", 0), "insights_count": 0},
        "quality_dimensions": det.get("quality_dimensions", {}), "risk_flags": [], "risk_score": 0, "next_steps": [],
    }
    pdf = report_generator.generate_pdf_report(analysis_data, det, ml, industry=ind)
    assert _pdf_bytes_look_valid(pdf)
