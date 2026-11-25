"""Run with python -m experiments.duration BANK_CSV from the repository root."""
import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from agents.analysis_evidence import file_identity
from agents.ml_engineer import _get_models, _preprocessor, classification_metrics


def run(path, output):
    frame = pd.read_csv(path, sep=None, engine="python")
    if set(frame["y"].unique()) != {"no", "yes"} or "duration" not in frame:
        raise ValueError("Expected a Bank Marketing dataset with duration and a complete yes/no target y.")
    y = frame.pop("y").eq("yes").astype(int)
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    model = dict(_get_models("classification"))["XGBoost"].set_params(n_jobs=1)
    rows = []
    for include in (True, False):
        X = frame if include else frame.drop(columns="duration")
        pipe = make_pipeline(_preprocessor(X), clone(model))
        pipe.fit(X.iloc[train], y.iloc[train])
        pred = pipe.predict(X.iloc[test])
        metrics = classification_metrics(y.iloc[test], pred, ["no", "yes"], 1, int(y.iloc[train].mode()[0]))
        metrics["roc_auc"] = float(roc_auc_score(y.iloc[test], pipe.predict_proba(X.iloc[test])[:, 1]))
        rows.append({"include_duration": include, "features": list(X), **metrics})
    params = {k: v for k, v in model.get_params().items() if v is not None}
    result = {"dataset_sha256": file_identity(path), "target": "y", "seed": 42, "test_fraction": 0.2,
              "stratified": True, "train_rows": len(train), "test_rows": len(test),
              "train_index_sha256": hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
              "test_index_sha256": hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
              "versions": {k: version(k) for k in ("xgboost", "numpy", "pandas", "scikit-learn")},
              "xgboost_parameters": {k: ("NaN" if isinstance(v, float) and np.isnan(v) else v) for k, v in params.items()},
              "arms": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "duration.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "duration.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    run(args.bank_csv, args.output)
