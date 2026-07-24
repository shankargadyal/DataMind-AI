from agents import clustering


def test_finds_three_well_separated_blobs(clustering_csv):
    import pandas as pd
    df = pd.read_csv(clustering_csv)
    det = {"numeric_cols": ["f1", "f2", "f3"]}
    out = clustering.run_clustering(df, det)

    assert out["task_type"] == "clustering"
    assert out["n_clusters"] == 3
    # Well-separated blobs should score a strong silhouette
    assert out["best_score"] > 0.5


def test_cluster_profiles_sum_to_total_rows(clustering_csv):
    import pandas as pd
    df = pd.read_csv(clustering_csv)
    det = {"numeric_cols": ["f1", "f2", "f3"]}
    out = clustering.run_clustering(df, det)
    total = sum(p["size"] for p in out["cluster_profiles"])
    # Allow for DBSCAN noise points not being assigned to a profile
    assert total <= len(df)
    assert total > 0


def test_too_few_numeric_columns_errors_gracefully():
    import pandas as pd
    df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})
    out = clustering.run_clustering(df, {"numeric_cols": ["only_col"]})
    assert "error" in out
    assert out["models"] == []
