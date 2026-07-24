"""
Agent 3b: Clustering Engine — unsupervised learning for datasets with no
clear prediction target.

Runs KMeans (auto k-selection via silhouette score), DBSCAN, and
Agglomerative clustering, scores each with silhouette / Calinski-Harabasz,
and returns the best segmentation along with a 2D PCA projection for
visualization and per-cluster feature profiles.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

warnings.filterwarnings("ignore")

MAX_ROWS_FOR_CLUSTERING = 5000  # sample large datasets for speed


def _best_kmeans_k(X: np.ndarray, k_min=2, k_max=8):
    """Try a range of k for KMeans, pick the one with the best silhouette score."""
    best_k, best_score, best_model = k_min, -1.0, None
    k_max = min(k_max, max(k_min, len(X) - 1))
    for k in range(k_min, k_max + 1):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k, best_model = score, k, model
        except Exception:
            continue
    return best_k, best_score, best_model


def _cluster_profiles(df_numeric: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> list:
    """Mean feature values per cluster, used to describe what makes each segment distinct."""
    profiles = []
    df_tmp = df_numeric.copy()
    df_tmp["_cluster"] = labels
    overall_mean = df_numeric.mean()
    for c in sorted(set(labels)):
        if c == -1:
            continue  # DBSCAN noise points
        sub = df_tmp[df_tmp["_cluster"] == c]
        means = sub[feature_cols].mean()
        # Which features are most distinctive for this cluster (largest deviation from overall mean)
        deviation = ((means - overall_mean) / (overall_mean.abs() + 1e-9)).abs().sort_values(ascending=False)
        top_features = [
            {"feature": f, "cluster_mean": round(float(means[f]), 4), "overall_mean": round(float(overall_mean[f]), 4)}
            for f in deviation.index[:5]
        ]
        profiles.append({
            "cluster": int(c),
            "size": int(len(sub)),
            "pct": round(len(sub) / max(len(df_tmp), 1) * 100, 1),
            "distinctive_features": top_features,
        })
    return profiles


def run_clustering(df: pd.DataFrame, det: dict) -> dict:
    """
    Main entry point. Returns a dict shaped consistently with run_ml_engineer's
    output so app.py / dashboard_agent can consume either without branching logic
    everywhere — task_type is "clustering" instead of classification/regression.
    """
    numeric_cols = [c for c in det.get("numeric_cols", []) if c in df.columns]
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for clustering", "models": [], "task_type": "clustering"}

    df_num = df[numeric_cols].dropna()
    if len(df_num) < 10:
        return {"error": "Not enough complete rows for clustering (need >= 10)", "models": [], "task_type": "clustering"}

    sampled = False
    if len(df_num) > MAX_ROWS_FOR_CLUSTERING:
        df_num = df_num.sample(MAX_ROWS_FOR_CLUSTERING, random_state=42)
        sampled = True

    scaler = StandardScaler()
    X = scaler.fit_transform(df_num.values)

    results = []

    # ── KMeans (auto k) ──────────────────────────────────────────────────
    try:
        k, sil, km_model = _best_kmeans_k(X)
        if km_model is not None:
            labels_km = km_model.labels_
            ch = calinski_harabasz_score(X, labels_km)
            results.append({
                "name": f"KMeans (k={k})", "algorithm": "kmeans",
                "score": round(float(sil), 4), "silhouette": round(float(sil), 4),
                "calinski_harabasz": round(float(ch), 2),
                "n_clusters": int(k), "labels": labels_km, "model": km_model,
                "is_best": False,
            })
    except Exception as e:
        results.append({"name": "KMeans", "algorithm": "kmeans", "score": -1, "error": str(e), "is_best": False})

    # ── DBSCAN ───────────────────────────────────────────────────────────
    try:
        # Heuristic eps based on feature scale (data is standardized, so ~0.5-1.5 is reasonable)
        best_db, best_db_score = None, -1.0
        for eps in (0.5, 0.8, 1.2):
            db = DBSCAN(eps=eps, min_samples=max(5, len(X) // 200))
            labels_db = db.fit_predict(X)
            n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
            if n_clusters < 2:
                continue
            try:
                score = silhouette_score(X, labels_db)
            except Exception:
                continue
            if score > best_db_score:
                best_db_score, best_db = score, (db, labels_db, n_clusters, eps)
        if best_db is not None:
            db, labels_db, n_clusters, eps = best_db
            noise_pct = round(float((labels_db == -1).sum()) / len(labels_db) * 100, 1)
            results.append({
                "name": f"DBSCAN (eps={eps})", "algorithm": "dbscan",
                "score": round(float(best_db_score), 4), "silhouette": round(float(best_db_score), 4),
                "n_clusters": int(n_clusters), "noise_pct": noise_pct,
                "labels": labels_db, "model": db, "is_best": False,
            })
    except Exception as e:
        results.append({"name": "DBSCAN", "algorithm": "dbscan", "score": -1, "error": str(e), "is_best": False})

    # ── Agglomerative / Hierarchical ────────────────────────────────────
    try:
        k_agg = next((r["n_clusters"] for r in results if r.get("algorithm") == "kmeans"), 3)
        agg = AgglomerativeClustering(n_clusters=k_agg)
        labels_agg = agg.fit_predict(X)
        sil_agg = silhouette_score(X, labels_agg)
        results.append({
            "name": f"Hierarchical (k={k_agg})", "algorithm": "agglomerative",
            "score": round(float(sil_agg), 4), "silhouette": round(float(sil_agg), 4),
            "n_clusters": int(k_agg), "labels": labels_agg, "model": agg, "is_best": False,
        })
    except Exception as e:
        results.append({"name": "Hierarchical", "algorithm": "agglomerative", "score": -1, "error": str(e), "is_best": False})

    valid_results = [r for r in results if r.get("score", -1) >= 0 and "labels" in r]
    if not valid_results:
        return {"error": "All clustering algorithms failed", "models": [], "task_type": "clustering"}

    best = max(valid_results, key=lambda r: r["score"])
    best["is_best"] = True
    best_labels = best["labels"]

    # ── PCA projection for 2D scatter visualization ─────────────────────
    pca_coords = []
    try:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        pca_coords = [
            {"x": round(float(c[0]), 4), "y": round(float(c[1]), 4), "cluster": int(lbl)}
            for c, lbl in zip(coords[:2000], best_labels[:2000])
        ]
        explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    except Exception:
        explained_var = []

    profiles = _cluster_profiles(df_num, best_labels, numeric_cols)

    clean_models = []
    for r in results:
        clean = {k: v for k, v in r.items() if k not in ("labels", "model")}
        clean_models.append(clean)

    return {
        "task_type": "clustering",
        "models": clean_models,
        "best_model": best["name"],
        "best_score": best["score"],
        "n_clusters": best.get("n_clusters", 0),
        "cluster_profiles": profiles,
        "pca_projection": pca_coords,
        "pca_explained_variance": explained_var,
        "feature_count": len(numeric_cols),
        "sampled": sampled,
        "sample_size": len(df_num),
        "feature_importances": [],
        "engineered_features": [],
        "future_prediction": {"available": False},
        "class_imbalance": {},
        "exploratory_mode": False,
        "target_column": "",
        # Private objects (kept for interface parity with ml_engineer output)
        "_model_obj": best.get("model"),
        "_scaler_obj": scaler,
        "_feature_cols": numeric_cols,
        "_le_target": None,
        "_numeric_cols": numeric_cols,
        "_cluster_labels": best_labels,
    }
