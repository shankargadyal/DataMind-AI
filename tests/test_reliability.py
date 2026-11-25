import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from agents import analysis_evidence, detective, evaluator, ml_engineer, reporter
from agents.explainability import generate_shap_plots
from agents.rag_agent import run_rag_indexing


@pytest.fixture
def bank_frame():
    counts = {"age": [40], "job": ["management"], "marital": ["married"], "education": ["secondary"],
              "default": ["no"], "balance": [100], "housing": ["yes"], "loan": ["no"], "contact": ["cellular"],
              "day": [2], "month": ["may"], "duration": [120], "campaign": [1], "pdays": [-1],
              "previous": [0], "poutcome": ["unknown"], "y": ["no"] * 4000 + ["yes"] * 521}
    return pd.DataFrame({key: value * 4521 if len(value) == 1 else value for key, value in counts.items()})


@pytest.fixture
def bank_metrics():
    actual = [0] * 801 + [1] * 104
    predicted = [0] * 766 + [1] * 35 + [0] * 62 + [1] * 42
    return ml_engineer.classification_metrics(actual, predicted, ["no", "yes"], 1)


@pytest.fixture
def bank_analysis(bank_frame, bank_metrics):
    quality = {"composite_score": 96.4, "grade": "Excellent", "dimensions": {"completeness": 100},
               "recommendations": ["Review outliers in balance."]}
    det = {"filename": "bank.csv", "original_shape": [4521, 17], "quality_score": 96.4, "quality_dimensions": quality}
    model = {"name": "XGBoost", "score": bank_metrics["accuracy"], "is_best": True, "metrics": bank_metrics}
    ml = {"target_column": "y", "task_type": "classification", "best_model": "XGBoost",
          "best_score": bank_metrics["accuracy"], "models": [model], "evaluation_protocol": {"positive_class": "yes"}}
    return det, ml, analysis_evidence.build_evidence(bank_frame, det, ml, "y", "fixture", {})


def test_bank_target_counts_and_percentages(bank_analysis):
    _, _, evidence = bank_analysis
    resolve = lambda text, previous="": analysis_evidence.resolve_question(text, evidence, previous)[0]
    assert "subscribed to a term deposit" in resolve("What is the prediction target?")
    assert "4,521 records" in resolve("How many records and columns are in the dataset?")
    assert "17 columns" in resolve("How many records and columns are in the dataset?")
    assert "521" in resolve("How many customers subscribed?")
    assert "4,000" in resolve("How many customers did not subscribe?")
    assert "11.52%" in resolve("What percentage subscribed?")
    assert "11.52%" in resolve("And the percentage?", "How many customers subscribed?")
    assert "88.48%" in resolve("What is the dataset majority-class baseline?")
    assert "0 missing cells" in resolve("How many missing values are there?")
    assert "no class distribution for the requested target" in resolve("How many housing=yes?")


def test_bank_confusion_matrix_metrics_and_rag(bank_analysis, bank_metrics):
    det, ml, evidence = bank_analysis
    assert bank_metrics["binary_confusion"] == {"tp": 42, "fp": 35, "fn": 62, "tn": 766}
    assert bank_metrics["accuracy"] == pytest.approx(0.893, abs=0.0005)
    assert bank_metrics["macro_f1"] == pytest.approx(0.702, abs=0.0005)
    assert bank_metrics["weighted_f1"] == pytest.approx(0.886, abs=0.0005)
    assert bank_metrics["positive_recall"] == pytest.approx(42 / 104)
    assert bank_metrics["positive_precision"] == pytest.approx(42 / 77)
    assert bank_metrics["positive_f1"] == pytest.approx(84 / 181)
    reply, _ = analysis_evidence.resolve_question("XGBoost macro F1, weighted F1 and subscriber recall", evidence)
    assert all(value in reply for value in ("70.2%", "88.6%", "40.4%", "[models/XGBoost]"))
    index = run_rag_indexing({}, ml, det, evidence=evidence)
    assert index.evidence["model"]["models"][0]["metrics"] == bank_metrics
    assert any(item["source"] == "models/XGBoost" for item in index.retrieve("XGBoost precision macro F1"))


@pytest.mark.parametrize(("delta", "outcome"), [
    (-0.015, "below"), (0.0005, "approximately_equal"), (0.008, "slightly_above"), (0.03, "substantially_above")
])
def test_baseline_guardrail_outcomes(delta, outcome):
    comparison = evaluator.compare_baseline({"accuracy": 0.885 + delta, "majority_baseline_accuracy": 0.885})
    assert comparison["outcome"] == outcome
    assert comparison["difference_percentage_points"] == pytest.approx(delta * 100)


def test_bank_guardrail_warns_about_recall_without_denying_improvement(bank_analysis):
    det, ml, _ = bank_analysis
    result = evaluator.run_evaluator(reporter.run_reporter(det, {}, ml, "", ""), det, ml)
    assert result["status"] == "FLAGGED"
    assert result["baseline_comparison"]["outcome"] == "slightly_above"
    assert "exceeds" in result["baseline_comparison"]["message"]
    assert any(f["check"] == "positive_recall" for f in result["flags"])


def test_pdf_quality_and_fallback_disclosure(bank_analysis):
    from pypdf import PdfReader
    from agents.report_generator import generate_pdf_report

    det, ml, evidence = bank_analysis
    report = reporter.run_reporter(det, {}, ml, "", "")
    metrics = ml["models"][0]["metrics"]
    analysis = {"filename": "bank.csv", "target": evidence["target"], "summary": report["report"],
                "quality_dimensions": det["quality_dimensions"], "degraded_mode": True,
                "ai_sources": {"reporter": "fallback"},
                "ml_recommendation": {**ml, "best_metrics": metrics}}
    payload = generate_pdf_report(analysis, det, ml, review={"decision": "approve"})
    text = " ".join(" ".join(page.extract_text().split()) for page in PdfReader(io.BytesIO(payload)).pages)
    assert "Composite data-quality score: 96.4/100" in text
    assert "Data completeness: 100%" in text
    assert "Prediction target: y" in text
    assert "fallback" in text.lower()
    assert "Human review approved" in text


@pytest.fixture
def fast_models(monkeypatch):
    monkeypatch.setattr(ml_engineer, "_get_models", lambda task: [
        ("Logistic Regression", LogisticRegression(random_state=42)) if task == "classification"
        else ("Ridge Regression", Ridge())
    ])


def test_pipeline_preserves_explicit_target_and_trains_preprocessing_by_fold(fast_models, monkeypatch):
    import agents.explainability as explainability
    monkeypatch.setattr(explainability, "generate_shap_plots", lambda *a, **k: {"available": False})
    rng = np.random.RandomState(42)
    frame = pd.DataFrame({"amount": rng.normal(size=96), "channel": ["phone", "email"] * 48,
                          "other": ["yes", "no", "no"] * 32, "outcome": ["no", "yes"] * 48})
    frame.loc[4, "amount"] = np.nan
    frame.loc[5, "channel"] = None
    medians = []
    original = ml_engineer.SimpleImputer.fit

    def record(imputer, X, y=None):
        result = original(imputer, X, y)
        if imputer.strategy == "median":
            medians.append((len(X), result.statistics_.tolist()))
        return result

    monkeypatch.setattr(ml_engineer.SimpleImputer, "fit", record)
    model = ml_engineer.run_ml_engineer(frame, {"df_raw": frame, "exploratory_mode": True},
                                       {"ml_recommendation": {"target_column": "other", "task_type": "exploratory"}},
                                       selected_target="outcome",
                                       availability={"channel": {"reason": "Collected later."}})
    assert model["target_column"] == "outcome"
    assert model["best_model"] == "Logistic Regression"
    assert model["availability_comparison"]["excluded"] == ["channel"]
    n_train = model["evaluation_protocol"]["n_train"]
    assert len([size for size, _ in medians if size < n_train]) == 6
    assert max(size for size, _ in medians) == n_train
    assert len({tuple(values) for _, values in medians}) > 1
    rejected = ml_engineer.run_ml_engineer(frame, {}, {}, selected_target="bad_column")
    assert rejected["target_column"] == "bad_column" and not rejected["models"]


def test_shap_explained_probability_matches_model():
    pytest.importorskip("shap")
    X = np.array([[-1, 0], [1, 0], [-2, 1], [2, 1], [0.5, 1], [-0.5, 0]])
    model = LogisticRegression(random_state=42).fit(X, [0, 1, 0, 1, 1, 0])
    result = generate_shap_plots(model, X, ["amount", "channel"], "classification", class_index=1,
                                 class_labels=["no", "yes"], target="y")
    assert result["available"], result.get("reason")
    assert result["explained_class"] == "yes"
    assert result["output_scale"] == "probability"
    assert result["target"] == "y"
    assert result["waterfall_final_value"] == pytest.approx(model.predict_proba(X[:1])[0, 1], abs=0.0001)


def test_regression_uses_r2_mae_rmse(fast_models):
    frame = pd.DataFrame({"quantity": np.arange(80), "target": np.arange(80) * 2 + 5})
    outcome = ml_engineer.run_ml_engineer(frame, {"df_raw": frame}, {}, mode_override="regression", selected_target="target")
    assert outcome["best_model"] == "Ridge Regression"
    assert outcome["evaluation_protocol"]["selection_rule"].endswith("R²")
    assert outcome["models"][0]["metrics"]["r2"] > .99
    assert outcome["models"][0]["metrics"]["mae"] >= 0


def test_session_integration_review_prediction_switching_and_restart(tmp_path, fast_models, monkeypatch):
    import app as datamind
    from agents import explainability
    monkeypatch.setattr(explainability, "generate_shap_plots", lambda *a, **k: {"available": False})
    monkeypatch.setattr(datamind, "GROQ_API_KEY", "")
    monkeypatch.setitem(datamind.app.config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(datamind, "HAS_DASHBOARD", False)
    client = datamind.app.test_client()
    other_client = datamind.app.test_client()
    for test_client in (client, other_client):
        assert test_client.post("/api/guest").status_code == 200

    frame = pd.DataFrame({"age": np.arange(80), "role": ["manager", "engineer"] * 40,
                          "y": ["yes"] * 12 + ["no"] * 68, "housing": ["no", "yes"] * 40})

    def submit(target):
        response = client.post("/api/analyze", data={
            "file": (io.BytesIO(frame.to_csv(index=False).encode()), "fixture.csv"),
            "target_column": target,
        })
        assert response.status_code == 200, response.json
        job = response.json["job_id"]
        for _ in range(300):
            status = client.get(f"/api/status/{job}").json
            if status["status"] != "running":
                break
            time.sleep(.05)
        assert status["status"] == "needs_review", status
        assert status["selected_target"] == target
        return job

    def ask(job, question):
        return client.post("/api/chat", json={"job_id": job, "message": question}).json["reply"]

    first = submit("y")
    route = f"/api/predict_info/{first}"
    assert client.get(route).json["status"] == "pending_review"
    assert "12" in ask(first, "How many positive cases?")
    assert other_client.post("/api/chat", json={"job_id": first, "message": "What is the prediction target?"}).status_code == 403
    review = f"/api/jobs/{first}/review"
    assert client.post(review, json={"decision": "approve"}).json["status"] == "done"
    assert client.post(review, json={"decision": "approve"}).status_code == 200
    assert client.post(review, json={"decision": "reject"}).status_code == 409
    ready = client.get(route).json
    assert ready["status"] == "ready"
    assert ready["target"] == "y"
    inputs = {feature["name"]: feature["default"] for feature in ready["features"]}
    assert client.post(f"/api/predict/{first}", json={"inputs": inputs}).json["target"] == "y"
    assert client.post(f"/api/whatif/{first}", json={"scenarios": [inputs]}).status_code == 200
    assert client.get(f"/api/download_report/{first}").data.startswith(b"%PDF")
    run = next(row for row in client.get("/api/history").json["runs"] if row["job_id"] == first)
    assert run["evaluation"]["selected_target"] == "y"
    assert len(run["evaluation"]["dataset_sha256"]) == 64
    second = submit("housing")
    assert "`housing`" in ask(second, "What is the prediction target?")
    assert "40" in ask(second, "How many positive cases?")
    del datamind.jobs[second]
    assert "`housing`" in ask(second, "What is the prediction target?")
    assert other_client.post("/api/chat", json={"job_id": second, "message": "What is the prediction target?"}).status_code == 403


def test_qualitative_claim_filter_rejects_unverified_numbers_and_causation():
    for text in ("The result is 100%.", "Five rows changed.", "This caused subscriptions.", "This guarantees retention."):
        assert not analysis_evidence.qualitative_text_is_safe(text)
    assert analysis_evidence.qualitative_text_is_safe("The model learned an association.")


def test_container_contract_and_frontend_asset():
    import app as datamind

    assert "${PORT:-5000}" in Path("Dockerfile").read_text(encoding="utf-8")
    with datamind.app.test_client() as client:
        body = client.get("/").get_data(as_text=True)
        import re
        asset = re.search(r'src="(/assets/[^"]+\.js)"', body).group(1)
        assert client.get(asset).status_code == 200


def test_chat_greets_with_or_without_an_analysis(monkeypatch):
    import app as datamind

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("A greeting should not look up analysis evidence")

    monkeypatch.setattr(datamind.data_store, "load_meta", unexpected_lookup)
    with datamind.app.test_client() as client:
        assert client.post("/api/guest").status_code == 200
        for message, job_id in (("hi", ""), ("  Hi!  ", "job_" + "a" * 32), ("Hello there!", "")):
            result = client.post("/api/chat", json={"message": message, "job_id": job_id})
            assert result.status_code == 200
            assert result.json["source"] == "greeting"
            assert result.json["reply"].startswith("Hi!")

        question = client.post("/api/chat", json={"message": "Hi, how many rows are there?"})
        assert question.json["source"] == "limitation"
