"""
Agent 3: ML Engineer v3.0 — Full ML pipeline.
New in v3:
  - Cross-validation scores alongside test scores
  - Exploratory mode (returns EDA stats instead of model)
  - Better future prediction with trend-based adjustment
  - Class imbalance detection and warning
  - Prediction confidence interval (classification)
  - Safe fallback for every sub-step
"""
import os
import platform
from importlib.metadata import PackageNotFoundError, version
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, confusion_matrix,
)
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _load_optional_models():
    """Import optional ML libraries only when model training is requested."""
    optional = {}

    try:
        from xgboost import XGBClassifier, XGBRegressor
        optional["xgboost"] = (XGBClassifier, XGBRegressor)
    except Exception as exc:
        print(f"[INFO] XGBoost unavailable: {type(exc).__name__}")

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        optional["lightgbm"] = (LGBMClassifier, LGBMRegressor)
    except Exception as exc:
        print(f"[INFO] LightGBM unavailable: {type(exc).__name__}")

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        optional["catboost"] = (CatBoostClassifier, CatBoostRegressor)
    except Exception as exc:
        print(f"[INFO] CatBoost unavailable: {type(exc).__name__}")

    return optional


from agents import clustering as _clustering_agent
from agents import forecasting as _forecasting_agent


# ── Task detection ──────────────────────────────────────────────────────────

def _detect_task(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    n_unique = series.nunique()
    n_total = max(len(series), 1)
    if n_unique <= 10 or (n_unique / n_total < 0.05 and n_unique <= 20):
        return "classification"
    return "regression"


def _select_target(df: pd.DataFrame, det: dict, requested: str) -> str | None:
    if requested and requested in df.columns:
        return requested
    suggested = det.get("suggested_target")
    if suggested in df.columns:
        return suggested
    priority = ["target", "label", "outcome", "diagnosis", "class", "status", "close", "price", "sales", "revenue"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in priority:
        if name in lower_map:
            return lower_map[name]
    numeric_cols = det.get("numeric_cols", [])
    if det.get("is_timeseries"):
        for name in ["close", "price", "sales", "revenue"]:
            if name in lower_map:
                return lower_map[name]
    return numeric_cols[-1] if numeric_cols else (df.columns[-1] if len(df.columns) else None)


# ── Feature engineering ─────────────────────────────────────────────────────

def _engineer_features(frame, date_cols, chronological):
    """Create row-local date parts and strictly past-only exogenous features."""
    features = frame.copy()
    engineered = []
    for col in date_cols:
        if col not in features:
            continue
        values = pd.to_datetime(features.pop(col), errors="coerce", format="mixed")
        for suffix, part in (("year", values.dt.year), ("month", values.dt.month),
                             ("day", values.dt.day), ("dow", values.dt.dayofweek)):
            name = f"{col}_{suffix}"
            if name not in features:
                features[name] = part
        engineered.append(f"date_{col}")

    if chronological:
        for col in frame.select_dtypes(include="number").columns[:3]:
            past = frame[col].shift(1)
            for suffix, values in (("roll3", past.rolling(3, min_periods=1).mean()),
                                   ("lag1", past), ("lag3", frame[col].shift(3))):
                name = f"{col}_{suffix}"
                if name not in features:
                    features[name] = values
            engineered.append(f"rolling_{col}")
    return features, engineered


def _preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in X if col not in numeric]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
    ], verbose_feature_names_out=False)


def _get_models(task_type: str):
    optional = _load_optional_models()

    if task_type == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ]

        if "xgboost" in optional:
            XGBClassifier, _ = optional["xgboost"]
            models.append(("XGBoost", XGBClassifier(
                n_estimators=100, random_state=42, verbosity=0,
                use_label_encoder=False, eval_metric="logloss"
            )))

        if "lightgbm" in optional:
            LGBMClassifier, _ = optional["lightgbm"]
            models.append(("LightGBM", LGBMClassifier(
                n_estimators=100, random_state=42, verbosity=-1
            )))

        if "catboost" in optional:
            CatBoostClassifier, _ = optional["catboost"]
            models.append(("CatBoost", CatBoostClassifier(
                iterations=100, random_state=42, verbose=False,
                allow_writing_files=False
            )))
    else:
        models = [
            ("Ridge Regression", Ridge(alpha=1.0)),
            ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]

        if "xgboost" in optional:
            _, XGBRegressor = optional["xgboost"]
            models.append(("XGBoost Regressor", XGBRegressor(
                n_estimators=100, random_state=42, verbosity=0
            )))

        if "lightgbm" in optional:
            _, LGBMRegressor = optional["lightgbm"]
            models.append(("LightGBM Regressor", LGBMRegressor(
                n_estimators=100, random_state=42, verbosity=-1
            )))

        if "catboost" in optional:
            _, CatBoostRegressor = optional["catboost"]
            models.append(("CatBoost Regressor", CatBoostRegressor(
                iterations=100, random_state=42, verbose=False,
                allow_writing_files=False
            )))

    return models



def _safe_cv(model, X_train, y_train, scoring: str, task_type: str, is_timeseries: bool = False):
    """Cross-validated score on the TRAINING split only. Returns (mean, std, folds, note).
    Returns (None, None, 0, reason) when CV is not statistically valid — never a fake 0.0.

    For time-series regression, plain KFold(shuffle=False) still trains 2 of its 3 folds on rows that
    are NEWER than the fold being scored (shuffle=False only stops rows being reordered WITHIN a fold,
    it does not stop a later fold's rows appearing in an earlier fold's training set). This silently lets
    model SELECTION see the future even though shuffle=False looks safe at a glance. TimeSeriesSplit
    instead guarantees every test fold is strictly later than every row used to train it.
    """
    from sklearn.base import clone
    if task_type == "classification":
        _, counts = np.unique(y_train, return_counts=True)
        folds = int(min(3, counts.min())) if len(counts) else 0
        if folds < 2:
            return None, None, 0, "cross-validation skipped: smallest class in the training split has < 2 members"
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        folds = 3 if len(y_train) >= 12 else 0
        if folds < 2:
            return None, None, 0, "cross-validation skipped: too few training rows"
        if is_timeseries:
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=folds)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(clone(model), X_train, y_train, cv=cv, scoring=scoring, error_score='raise')
        return float(scores.mean()), float(scores.std()), folds, ""
    except Exception as e:
        return None, None, 0, f"cross-validation failed: {type(e).__name__}"


def _safe_cv_classification(model, X_train, y_train):
    """One cross-validation pass that returns BOTH accuracy and macro-F1 (same folds as _safe_cv, no extra fits).
    Returns (acc_mean, acc_std, f1_macro_mean, folds, note); Nones when CV is not statistically valid."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_validate
    _, counts = np.unique(y_train, return_counts=True)
    folds = int(min(3, counts.min())) if len(counts) else 0
    if folds < 2:
        return None, None, None, 0, "cross-validation skipped: smallest class in the training split has < 2 members"
    try:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        cv = cross_validate(clone(model), X_train, y_train, cv=splitter, error_score="raise",
                            scoring={"acc": "accuracy", "f1m": "f1_macro"})
        return (float(cv["test_acc"].mean()), float(cv["test_acc"].std()),
                float(cv["test_f1m"].mean()), folds, "")
    except Exception as e:
        return None, None, None, 0, f"cross-validation failed: {type(e).__name__}"


def classification_metrics(y_test, y_pred, labels, positive_index=None, baseline_class=None):
    classes = list(range(len(labels)))
    matrix = confusion_matrix(y_test, y_pred, labels=classes)
    support = matrix.sum(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": list(labels),
        "holdout_support": {label: int(n) for label, n in zip(labels, support)},
        "majority_baseline_accuracy": float(support.max() / support.sum()),
    }
    majority = int(support.argmax())
    baseline = np.full(len(y_test), majority)
    metrics["baseline_macro_f1"] = float(f1_score(y_test, baseline, labels=classes, average="macro", zero_division=0))
    metrics["majority_baseline_label"] = labels[majority]
    if baseline_class is not None:
        metrics["train_majority_holdout_accuracy"] = float(support[baseline_class] / support.sum())
    metrics["f1"] = metrics["weighted_f1"]
    if positive_index is not None:
        negative_index = 1 - positive_index
        tp, fp = (int(matrix[i, positive_index]) for i in (positive_index, negative_index))
        fn, tn = (int(matrix[i, negative_index]) for i in (positive_index, negative_index))
        metrics.update({
            "positive_class": labels[positive_index],
            "positive_recall": tp / (tp + fn) if tp + fn else None,
            "positive_precision": tp / (tp + fp) if tp + fp else None,
            "positive_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "binary_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        })
    return metrics


def _eval_metrics(model, X_train, X_test, y_train, y_test, task_type, labels, positive_index, chronological):
    predicted = model.predict(X_test)
    if task_type == "classification":
        acc, std, macro, folds, note = _safe_cv_classification(model, X_train, y_train)
        metrics = classification_metrics(y_test, predicted, labels, positive_index, int(np.bincount(y_train).argmax()))
        if positive_index is not None and hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            probability = model.predict_proba(X_test)[:, list(model.classes_).index(positive_index)]
            metrics["roc_auc"] = float(roc_auc_score(np.asarray(y_test) == positive_index, probability))
        metrics.update(cv_score=acc, cv_std=std, cv_f1_macro=macro, cv_folds=folds, cv_note=note,
                       n_test=len(y_test), f1_macro=metrics["macro_f1"],
                       averaging="Precision, recall and weighted F1 use class support. Macro F1 weights classes equally.")
        return metrics
    mean, std, folds, note = _safe_cv(model, X_train, y_train, "r2", task_type, chronological)
    return {"r2": float(r2_score(y_test, predicted)), "mae": float(mean_absolute_error(y_test, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "cv_score": mean, "cv_std": std, "cv_folds": folds, "cv_note": note, "n_test": len(y_test)}


# ── SHAP / feature importance ───────────────────────────────────────────────

def _shap_unsafe(model) -> bool:
    """CatBoost multiclass + shap.Explainer corrupts the heap (native abort that try/except cannot catch and
    that would kill the whole worker). Use the model's own feature_importances_ for it instead."""
    return type(model).__name__.startswith("CatBoost") and len(getattr(model, "classes_", [])) > 2


MACRO_SELECTION_RATIO = 0.7        # choose models by macro-F1 when smallest/largest class ratio is below this
PERMUTATION_MAX_FEATURES = 80      # work cap: n_features x repeats x predict(rows) must stay cheap for slow predictors (KNN)


def _permutation_importance_fallback(model, X_eval, y_eval, feature_cols, scoring) -> list:
    """Model-agnostic importance for models with no gini/coef/SHAP (e.g. KNN): how much the HOLD-OUT score drops
    when one feature's values are shuffled. Uses a small fixed sample and seed (reproducible, bounded cost)."""
    if X_eval is None or y_eval is None or len(feature_cols) == 0:
        return []
    X_eval = np.asarray(X_eval)
    if X_eval.ndim != 2 or X_eval.shape[1] != len(feature_cols) or X_eval.shape[1] > PERMUTATION_MAX_FEATURES:
        return []
    try:
        from sklearn.inspection import permutation_importance
        n = min(300, len(y_eval))
        res = permutation_importance(model, X_eval[:n], np.asarray(y_eval)[:n], scoring=scoring,
                                     n_repeats=3, random_state=42, n_jobs=1)
        out = [{"feature": c, "importance": round(float(max(m, 0.0)), 6), "method": "permutation"}
               for c, m in zip(feature_cols, res.importances_mean)]
        out.sort(key=lambda x: x["importance"], reverse=True)
        return out[:15] if any(o["importance"] > 0 for o in out) else []
    except Exception:
        return []


def _data_relationships(df, target_col) -> dict:
    """Copies / bins / lookups / ratios between columns and target leakage, inferred from the data alone."""
    try:
        from agents.leakage_check import analyze_relationships
        return analyze_relationships(df, target=target_col)
    except Exception:
        return {"id_columns": [], "near_duplicates": [], "derived_columns": [], "target_leakage": [], "summary": []}

def _get_feature_importance(
    model, X_sample: np.ndarray, feature_cols: list,
    X_eval=None, y_eval=None, scoring=None
) -> list:
    importances = []

    # Load SHAP only when feature importance is requested.
    try:
        if not _shap_unsafe(model):
            import shap

            explainer = shap.Explainer(model, X_sample[:100])
            shap_values = explainer(X_sample[:100])
            vals = np.abs(shap_values.values)

            if vals.ndim == 3:
                vals = vals.mean(axis=2)

            mean_shap = vals.mean(axis=0)
            for col, imp in zip(feature_cols, mean_shap):
                importances.append({
                    "feature": col,
                    "importance": round(float(imp), 6),
                    "method": "shap",
                })

            importances.sort(
                key=lambda x: x["importance"], reverse=True
            )
            return importances[:15]
    except Exception:
        pass

    # Fall back to built-in tree feature importance.
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        for col, imp in list(zip(feature_cols, fi))[:len(feature_cols)]:
            importances.append({
                "feature": col,
                "importance": round(float(imp), 6),
                "method": "gini",
            })
        importances.sort(
            key=lambda x: x["importance"], reverse=True
        )
        return importances[:15]

    # Fall back to linear-model coefficients.
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).flatten()[:len(feature_cols)]
        for col, imp in zip(feature_cols, coef):
            importances.append({
                "feature": col,
                "importance": round(float(imp), 6),
                "method": "coefficient",
            })
        importances.sort(
            key=lambda x: x["importance"], reverse=True
        )
        return importances[:15]

    # Final fallback for models such as KNN.
    perm = _permutation_importance_fallback(
        model, X_eval, y_eval, feature_cols, scoring
    )
    if perm:
        return perm

    return []  # no honest importance available for this model type — never fabricate scores


# ── Future prediction ───────────────────────────────────────────────────────

def _future_prediction(pipeline, feature_cols, df_feat, target_col, steps=10):
    try:
        historical_y = df_feat[target_col].values.tolist()
        recent = historical_y[-min(10, len(historical_y)):]
        trend_per_step = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        recent_std = float(np.std(recent)) if len(recent) > 1 else 0.0

        # Fit quality of the LINEAR TREND against the actual recent history (R^2 of recent vs. the straight
        # line used for trend_per_step), and its standard error. These were previously never computed at
        # all: the report printed `fut.get('trend_r2', 0)` / `fut.get('std_error', 0)` defaults for keys
        # that simply did not exist, which looked like a genuine "R2 = 0" result rather than a missing value.
        n_recent = len(recent)
        if n_recent >= 3:
            xs = np.arange(n_recent, dtype=float)
            line = recent[0] + trend_per_step * xs
            resid = np.asarray(recent) - line
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((np.asarray(recent) - np.mean(recent)) ** 2))
            trend_r2 = round(1 - ss_res / ss_tot, 4) if ss_tot > 1e-12 else None
            std_error = round(float(np.std(resid, ddof=1)), 4) if n_recent > 2 else None
        else:
            trend_r2, std_error = None, None

        base_row = df_feat[feature_cols].iloc[[-1]]
        future_y, conf_upper, conf_lower = [], [], []
        for i in range(steps):
            pred = float(pipeline.predict(base_row)[0])
            trend_adjusted = pred + trend_per_step * min(i, 3) * 0.25
            future_y.append(round(trend_adjusted, 4))
            # Band widens with the forecast horizon (uncertainty compounds further out) instead of being a
            # flat +/-5% of the point prediction at every step, which previously produced an identical band
            # width for step 0 and step 9 and is not a genuine (let alone 95%) prediction interval.
            horizon_factor = (1 + i) ** 0.5
            se = std_error if std_error is not None else recent_std
            band = max(abs(trend_adjusted) * 0.02, se * 1.0) * horizon_factor
            conf_upper.append(round(trend_adjusted + band, 4))
            conf_lower.append(round(trend_adjusted - band, 4))

        return {
            "available": True,
            "target":          target_col,
            "future_y":        future_y,
            "future_upper":    conf_upper,
            "future_lower":    conf_lower,
            "confidence_upper": conf_upper,
            "confidence_lower": conf_lower,
            "historical_y":    [round(float(v), 4) for v in historical_y[-30:]],
            "future_steps":    steps,
            "trend_per_step":  round(trend_per_step, 4),
            "trend_direction": "up" if trend_per_step > 0 else "down",
            "trend_r2":        trend_r2,
            "std_error":       std_error,
            # Explicitly documents that this is NOT a model-fit interval; it widens with horizon but is
            # still a heuristic band around a point forecast, not a calibrated prediction interval.
            "interval_note":   "Approximate band (grows with forecast horizon), not a calibrated prediction interval.",
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ── Class imbalance check ───────────────────────────────────────────────────

def _class_imbalance(y) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return {"imbalanced": False}
    ratio = counts.min() / counts.max()
    return {
        "imbalanced": ratio < 0.3,
        "ratio": round(float(ratio), 3),
        "class_counts": {str(u): int(c) for u, c in zip(unique, counts)},
        "warning": "Class imbalance detected — model may be biased toward majority class" if ratio < 0.3 else "",
    }


# ── EDA-only output ────────────────────────────────────────────────────────

def _exploratory_output(df: pd.DataFrame, det: dict) -> dict:
    numeric_cols = det.get("numeric_cols", [])
    stats = det.get("column_stats", {})
    top_corr = det.get("top_correlations", [])
    return {
        "models": [],
        "task_type": "exploratory",
        "target_column": "",
        "best_model": "N/A — Exploratory Mode",
        "best_score": 0.0,
        "feature_importances": [],
        "engineered_features": [],
        "feature_count": len(numeric_cols),
        "future_prediction": {"available": False},
        "class_imbalance": {},
        "exploratory_mode": True,
        "eda_summary": {
            "top_correlations": top_corr[:5],
            "column_stats": {c: stats[c] for c in list(stats.keys())[:10]},
        },
        "_model_obj": None,
        "_scaler_obj": None,
        "_feature_cols": numeric_cols,
        "_le_target": None,
        "_numeric_cols": numeric_cols,
    }


# ── Main entry ─────────────────────────────────────────────────────────────

def run_ml_engineer(df, det, ana, mode_override="", selected_target="", availability=None):
    mode = (mode_override or "").strip().lower()
    if mode == "clustering":
        return _clustering_agent.run_clustering(df, det)

    recommendation = ana.get("ml_recommendation", {})
    if not selected_target and not mode and (det.get("exploratory_mode") or recommendation.get("task_type") == "exploratory"):
        return _exploratory_output(df, det)
    target = selected_target or _select_target(df, det, recommendation.get("target_column", ""))
    if target not in df:
        return {"error": "No valid target column", "models": [], "target_column": selected_target}
    dates = det.get("date_cols", [])
    if mode == "forecasting":
        return _forecasting_agent.run_forecasting(df, det, target_col=target, date_col=dates[0] if dates else "")

    raw = det.get("df_raw")
    if raw is None:
        raw = df
    detected = _detect_task(raw[target].dropna())
    task = mode if mode in ("classification", "regression") else (
        "classification" if detected == "classification" or recommendation.get("task_type") == "classification" else "regression"
    )
    eval_warnings = []
    relationships = _data_relationships(df, target)

    def unavailable(message, models=None):
        return {"error": message, "models": models or [], "target_column": target, "task_type": task,
                "evaluation_warnings": eval_warnings, "data_relationships": relationships}

    y_raw = raw[target] if task == "classification" else pd.to_numeric(raw[target], errors="coerce")
    valid = y_raw.notna()
    if task == "regression":
        valid &= np.isfinite(y_raw)
    if valid.sum() < 20:
        return unavailable("Not enough labeled data rows (need at least 20)")
    if (~valid).any():
        eval_warnings.append(f"{int((~valid).sum())} rows without a usable target were excluded.")

    chronological = bool(det.get("is_timeseries") and task == "regression")
    frame = raw.drop(columns=[target])
    if chronological and dates and dates[0] in frame:
        order = pd.to_datetime(frame[dates[0]], errors="coerce", format="mixed").sort_values(kind="stable").index
        frame = frame.loc[order]
    X, engineered = _engineer_features(frame, dates, chronological)
    X = X.loc[valid]
    for col in X:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            X[col] = X[col].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    y_raw = y_raw.loc[X.index]
    y = y_raw.astype(str).to_numpy() if task == "classification" else y_raw.to_numpy()
    counts = pd.Series(y).value_counts() if task == "classification" else pd.Series(dtype=int)
    fraction = 0.2 if len(X) > 100 else 0.15
    n_test = int(np.ceil(len(X) * fraction))
    stratified = (task == "classification" and len(counts) >= 2 and counts.min() >= 2
                  and min(n_test, len(X) - n_test) >= len(counts))
    if task == "classification" and not stratified:
        return unavailable("Class support is insufficient for a representative hold-out")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=fraction, random_state=42, shuffle=not chronological, stratify=y if stratified else None,
    )
    before_cap = len(y_train)
    cap = max(1000, int(os.environ.get("DATAMIND_MAX_TRAIN_ROWS", "20000")))
    if before_cap > cap:
        if chronological:
            X_train, y_train = X_train.iloc[-cap:], y_train[-cap:]
        else:
            try:
                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, train_size=cap, random_state=42, stratify=y_train if stratified else None
                )
            except ValueError:
                return unavailable("The training sample cap cannot represent all target classes")
        eval_warnings.append(f"Used {cap:,} of {before_cap:,} training rows. The hold-out split was kept intact.")

    # Declared identifier names are a fixed schema rule. Data-dependent
    # transformations are left to the pipeline fitted in each fold.
    excluded = [col for col in X_train if col.lower() in
                {"id", "index", "row_id", "record_id", "customer_id", "employee_id", "uuid"}]
    if excluded:
        eval_warnings.append("Identifier fields excluded by schema: " + ", ".join(excluded) + ".")
        X_train, X_test = X_train.drop(columns=excluded), X_test.drop(columns=excluded)
    if not len(X_train.columns):
        return unavailable("No usable features after removing identifier columns")

    encoder, labels, positive = None, [], None
    imbalance = {}
    if task == "classification":
        encoder = LabelEncoder().fit(y_train)
        labels = encoder.classes_.tolist()
        if len(labels) != len(counts) or not set(y_test) <= set(labels):
            return unavailable("The hold-out split cannot represent every class in training")
        y_train, y_test = encoder.transform(y_train), encoder.transform(y_test)
        if np.bincount(y_train).min() < 2 or len(np.unique(y_test)) != len(labels):
            return unavailable("Class support is insufficient for stratified validation")
        normalized = [item.lower() for item in labels]
        if len(labels) == 2:
            key = next((item for item in ("yes", "true", "1", "1.0", "positive", "subscribed") if item in normalized), None)
            positive = normalized.index(key) if key is not None else None
        imbalance = _class_imbalance(y_train)
        imbalance["class_counts"] = {label: int(n) for label, n in zip(labels, np.bincount(y_train))}

    primary = "accuracy" if stratified else "r2"
    macro_selection = stratified and np.bincount(y_train).min() / np.bincount(y_train).max() < MACRO_SELECTION_RATIO
    selection_key = "cv_f1_macro" if macro_selection else "cv_score"
    selection_name = "macro F1" if macro_selection else ("accuracy" if stratified else "R²")

    def train(features):
        train_X, test_X = X_train[features], X_test[features]
        results, selected, fitted = [], None, None
        for name, estimator in _get_models(task):
            pipeline = Pipeline([("preprocess", _preprocessor(train_X)), ("scale", StandardScaler()), ("model", estimator)])
            try:
                pipeline.fit(train_X, y_train)
                metrics = _eval_metrics(pipeline, train_X, test_X, y_train, y_test, task, labels, positive, chronological)
                score = metrics[selection_key]
                if score is None or not np.isfinite(score):
                    raise ValueError(metrics["cv_note"] or "Cross-validation produced no usable score")
                model = {"name": name, "score": metrics[primary], "score_metric": primary, "cv_score": metrics["cv_score"],
                         "selection_score": score, "metrics": metrics, "is_best": False}
                if selected is None or score > selected["selection_score"]:
                    selected, fitted = model, pipeline
                results.append(model)
            except Exception as exc:
                results.append({"name": name, "score": 0, "metrics": {}, "is_best": False, "error": str(exc)})
        if selected:
            selected["is_best"] = True
        return results, selected, fitted

    features = list(X_train.columns)
    models, winner, pipeline = train(features)
    if not winner:
        return unavailable("No model completed training and cross-validation", models)
    preprocess, scaler, estimator = (pipeline.named_steps[key] for key in ("preprocess", "scale", "model"))
    names = preprocess.get_feature_names_out().tolist()
    transformed = pipeline[:-1].transform(X_test.iloc[:200])
    numeric = next(cols for name, _, cols in preprocess.transformers_ if name == "numeric")
    categorical = next(cols for name, _, cols in preprocess.transformers_ if name == "categorical")
    mappings, defaults = {}, {}
    if numeric:
        defaults.update(zip(numeric, preprocess.named_transformers_["numeric"].statistics_.tolist()))
    if categorical:
        cat = preprocess.named_transformers_["categorical"]
        mappings.update({col: values.tolist() for col, values in zip(categorical, cat.named_steps["encode"].categories_)})
        defaults.update(zip(categorical, cat.named_steps["impute"].statistics_.tolist()))

    try:
        from agents.explainability import generate_shap_plots
        shap = generate_shap_plots(estimator, transformed[:50], names, task, class_index=positive,
                                   class_labels=labels, target=target)
    except Exception as exc:
        shap = {"available": False, "reason": str(exc)}
    if shap.get("available"):
        importance = [{"feature": item["feature"], "importance": item["mean_abs_shap"], "method": "shap"}
                      for item in shap["top_features"]]
    else:
        importance = _get_feature_importance(estimator, transformed, names, X_eval=transformed,
                                             y_eval=y_test[:len(transformed)], scoring="f1_macro" if stratified else "r2")

    features_with_timing = {col: entry for col, entry in (availability or {}).items() if col in features}
    excluded_for_timing = [col for col in features if col in features_with_timing or any(
        col == f"{source}_{suffix}" for source in (availability or {})
        for suffix in ("year", "month", "day", "dow", "roll3", "lag1", "lag3")
    )]
    restricted = [col for col in features if col not in excluded_for_timing]
    comparison = None
    if excluded_for_timing and restricted:
        alternatives, alternate, _ = train(restricted)
        comparison = {"excluded": excluded_for_timing, "models": alternatives,
                      "best_model": (alternate or {}).get("name"), "best_metrics": (alternate or {}).get("metrics", {})}
    future = _future_prediction(pipeline, features, X.assign(**{target: y_raw}), target) if chronological else {}
    package_versions = {}
    for package in ("numpy", "pandas", "xgboost", "lightgbm", "catboost", "shap"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    protocol = {
        "target": target, "split": "chronological hold-out" if chronological else "hold-out test set",
        "n_train": len(y_train), "n_test": len(y_test), "test_fraction": fraction,
        "n_train_before_cap": before_cap, "train_sample_cap": cap if before_cap > cap else None,
        "stratified": stratified, "random_state": 42,
        "reported_metric": primary, "selection_rule": f"Highest training cross-validation {selection_name}",
        "cv_type": "StratifiedKFold" if stratified else ("TimeSeriesSplit" if chronological else "KFold"),
        "cv_folds": winner["metrics"]["cv_folds"],
        "positive_class": labels[positive] if positive is not None else None,
        "excluded_unlabeled_rows": int((~valid).sum()), "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(), "package_versions": package_versions,
    }
    return {
        "models": models, "task_type": task, "target_column": target, "best_model": winner["name"],
        "best_score": winner["score"], "feature_importances": importance, "shap_plots": shap,
        "engineered_features": engineered, "feature_count": len(features), "future_prediction": future,
        "class_imbalance": imbalance, "data_relationships": relationships, "exploratory_mode": False,
        "score_metric": primary, "evaluation_protocol": protocol, "evaluation_warnings": eval_warnings,
        "selection_metric": selection_key, "feature_availability": availability or {},
        "availability_comparison": comparison,
        "_model_obj": estimator, "_scaler_obj": scaler, "_preprocessor_obj": preprocess,
        "_feature_cols": features, "_le_target": encoder, "_numeric_cols": numeric,
        "_categorical_mappings": mappings, "_feature_defaults": defaults,
    }
