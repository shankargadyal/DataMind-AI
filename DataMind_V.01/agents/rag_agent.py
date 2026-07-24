"""
Agent: RAG (Retrieval-Augmented Generation)
────────────────────────────────────────────
Turns the chat endpoint from fixed context-stuffing (first 4 insights,
first 3 actions, first 800 chars of summary) into real retrieval: every
piece of analysis output — insights, actions, SHAP captions, quality
report, industry KPIs, column stats — gets chunked and indexed, and the
chat agent pulls back only the chunks relevant to whatever the user
actually asked.

Uses TF-IDF + cosine similarity (scikit-learn, already a dependency)
rather than sentence-transformers/FAISS — no new heavy install, no
model download at runtime, consistent with how Prophet is kept optional
elsewhere in this project for the same reason (build weight vs. hosting
constraints on Render/Railway free tiers).
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _safe_str(x, limit=500):
    return str(x)[:limit] if x is not None else ""


def build_chunks(analysis_data: dict, ml_result: dict = None, det_result: dict = None) -> list[dict]:
    """
    Flatten every piece of analysis output into retrievable chunks:
    {"text": ..., "source": ...} — source is shown to the user so answers
    stay traceable back to which agent produced the underlying fact.
    """
    chunks = []
    ad = analysis_data or {}
    ml = ml_result or {}
    det = det_result or {}

    # All insights (not just the first 4 the old chat prompt used)
    for i, ins in enumerate(ad.get("key_insights", [])):
        if isinstance(ins, dict):
            text = f"{ins.get('title','')}: {ins.get('detail', ins.get('description',''))}"
        else:
            text = str(ins)
        if text.strip():
            chunks.append({"text": text, "source": f"insight_{i}"})

    # All recommended actions (not just the first 3)
    for i, a in enumerate(ad.get("actions", [])):
        if isinstance(a, dict):
            text = f"Recommended action: {a.get('action', a.get('title',''))}. Reason: {a.get('reason','')}"
        else:
            text = str(a)
        if text.strip():
            chunks.append({"text": text, "source": f"action_{i}"})

    # Data Quality Center (Detective agent)
    if det.get("quality_score") is not None:
        chunks.append({
            "text": f"Data quality score: {det.get('quality_score')}/100. "
                    f"Dimensions: {_safe_str(det.get('quality_dimensions', {}))}",
            "source": "quality_center",
        })
    for i, rec in enumerate(det.get("recommendations", []) or []):
        chunks.append({"text": f"Data quality recommendation: {rec}", "source": f"quality_rec_{i}"})

    # ML Engineer leaderboard + best model
    ml_rec = ad.get("ml_recommendation", {}) or ml.get("ml_recommendation", {})
    if ml_rec:
        chunks.append({
            "text": f"Best model: {ml_rec.get('best_model','')}, task: {ml_rec.get('task_type','')}, "
                    f"target column: {ml_rec.get('target_column','')}, score: {ml_rec.get('score','')}",
            "source": "ml_leaderboard",
        })
    for i, fi in enumerate((ml_rec.get("feature_importances") or [])[:20]):
        chunks.append({"text": f"Feature importance: {_safe_str(fi)}", "source": f"feature_importance_{i}"})

    # SHAP explainability captions (global + local), if present
    for i, cap in enumerate(ad.get("shap_global_captions", []) or []):
        chunks.append({"text": f"Model explanation (global): {cap}", "source": f"shap_global_{i}"})
    for i, cap in enumerate(ad.get("shap_local_captions", []) or []):
        chunks.append({"text": f"Model explanation (local): {cap}", "source": f"shap_local_{i}"})

    # Industry KPI matches
    for i, kpi in enumerate(ad.get("industry_kpis", []) or []):
        chunks.append({"text": f"Industry KPI: {_safe_str(kpi)}", "source": f"industry_kpi_{i}"})

    # Full summary (not truncated to 800 chars like the old prompt)
    if ad.get("summary"):
        # split into sentence-ish pieces so retrieval can pull just the relevant part
        for i, part in enumerate(str(ad["summary"]).split(". ")):
            if part.strip():
                chunks.append({"text": part.strip(), "source": f"summary_{i}"})

    return chunks


class JobRAGIndex:
    """One of these is built per completed job and cached in the job dict."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks] or [""]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 6) -> list[dict]:
        if not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [
            {**self.chunks[i], "score": round(float(sims[i]), 4)}
            for i in top_idx if sims[i] > 0
        ]


def run_rag_indexing(analysis_data: dict, ml_result: dict = None, det_result: dict = None) -> JobRAGIndex:
    """Entry point called once from run_pipeline() after Reporter finishes."""
    chunks = build_chunks(analysis_data, ml_result, det_result)
    return JobRAGIndex(chunks)
