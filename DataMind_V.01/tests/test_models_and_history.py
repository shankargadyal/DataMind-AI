import os
import io
import time
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    """A fresh app + in-memory SQLite DB per test, so tests never share state
    or depend on run order."""
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    import importlib
    import app as A
    importlib.reload(A)
    return A


def test_register_creates_a_user_in_the_database(app_module):
    client = app_module.app.test_client()
    r = client.post("/api/register", json={"name": "Ada", "email": "ada@test.com", "password": "secret123"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["email"] == "ada@test.com"
    assert "token" in body

    with app_module.app.app_context():
        from models import get_user_by_email
        user = get_user_by_email("ada@test.com")
        assert user is not None
        assert user.name == "Ada"
        # Password must never be stored in plaintext
        assert user.password_hash != "secret123"


def test_duplicate_email_is_rejected(app_module):
    client = app_module.app.test_client()
    client.post("/api/register", json={"name": "Ada", "email": "ada@test.com", "password": "secret123"})
    r = client.post("/api/register", json={"name": "Someone Else", "email": "ada@test.com", "password": "other123"})
    assert r.status_code == 409


def test_login_requires_correct_password(app_module):
    client = app_module.app.test_client()
    client.post("/api/register", json={"name": "Ada", "email": "ada@test.com", "password": "secret123"})
    bad = client.post("/api/login", json={"email": "ada@test.com", "password": "wrong"})
    assert bad.status_code == 401
    good = client.post("/api/login", json={"email": "ada@test.com", "password": "secret123"})
    assert good.status_code == 200


def test_login_persists_across_requests(app_module):
    """Regression guard for the users.json -> SQLAlchemy migration: a user
    registered in one request must still be found in a later, separate
    request — this is exactly the bug class flat-file-without-locking has
    under concurrent access."""
    client = app_module.app.test_client()
    client.post("/api/register", json={"name": "Ada", "email": "ada@test.com", "password": "secret123"})
    r1 = client.post("/api/login", json={"email": "ada@test.com", "password": "secret123"})
    r2 = client.post("/api/login", json={"email": "ada@test.com", "password": "secret123"})
    assert r1.status_code == 200 and r2.status_code == 200


def test_history_empty_for_new_user(app_module):
    client = app_module.app.test_client()
    r = client.post("/api/register", json={"name": "Ada", "email": "ada@test.com", "password": "secret123"})
    token = r.get_json()["token"]
    r2 = client.get("/api/history", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    assert r2.get_json()["runs"] == []


def test_completed_run_appears_in_history(app_module, rng):
    client = app_module.app.test_client()
    r = client.post("/api/guest")
    token = r.get_json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    n = 120
    df = pd.DataFrame({"age": rng.integers(18, 70, n), "income": rng.normal(50000, 15000, n)})
    df["target"] = (df["income"] > 50000).astype(int)
    csv_bytes = df.to_csv(index=False).encode()

    data = {"file": (io.BytesIO(csv_bytes), "test.csv"), "query": "insights",
            "target_column": "target", "mode": "auto"}
    r2 = client.post("/api/analyze", data=data, content_type="multipart/form-data", headers=headers)
    job_id = r2.get_json()["job_id"]

    for _ in range(60):
        rs = client.get(f"/api/status/{job_id}", headers=headers)
        if rs.get_json()["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert rs.get_json()["status"] == "done"

    r3 = client.get("/api/history", headers=headers)
    runs = r3.get_json()["runs"]
    assert len(runs) == 1
    assert runs[0]["filename"] == "test.csv"
    assert runs[0]["task_type"] == "classification"
    assert 0 <= runs[0]["best_score"] <= 1
    assert runs[0]["scoring_metric"] == "accuracy"


def test_history_isolated_per_user(app_module, rng):
    """One user's run history must never leak into another user's /api/history."""
    client = app_module.app.test_client()
    r1 = client.post("/api/register", json={"name": "Alice", "email": "alice@test.com", "password": "secret123"})
    token_alice = r1.get_json()["token"]
    r2 = client.post("/api/register", json={"name": "Bob", "email": "bob@test.com", "password": "secret123"})
    token_bob = r2.get_json()["token"]

    n = 80
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)})
    csv_bytes = df.to_csv(index=False).encode()
    data = {"file": (io.BytesIO(csv_bytes), "alice_data.csv"), "query": "segment", "mode": "clustering"}
    r3 = client.post("/api/analyze", data=data, content_type="multipart/form-data",
                      headers={"Authorization": f"Bearer {token_alice}"})
    job_id = r3.get_json()["job_id"]
    for _ in range(60):
        rs = client.get(f"/api/status/{job_id}", headers={"Authorization": f"Bearer {token_alice}"})
        if rs.get_json()["status"] in ("done", "error"):
            break
        time.sleep(0.2)

    alice_history = client.get("/api/history", headers={"Authorization": f"Bearer {token_alice}"}).get_json()
    bob_history = client.get("/api/history", headers={"Authorization": f"Bearer {token_bob}"}).get_json()
    assert len(alice_history["runs"]) == 1
    assert len(bob_history["runs"]) == 0
