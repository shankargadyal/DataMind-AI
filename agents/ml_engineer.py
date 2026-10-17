"""
Agent 3: ML Engineer v3.0 — Full ML pipeline.
New in v3:
  - Polynomial feature engineering (degree-2 for top features)
  - Cross-validation scores alongside test scores
  - Exploratory mode (returns EDA stats instead of model)
  - Better future prediction with trend-based adjustment
  - Class imbalance detection and warning
  - Prediction confidence interval (classification)
  - Safe fallback for every sub-step
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error, confusion_matrix,
)
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from agents import clustering as _clustering_agent
from agents import forecasting as _forecasting_agent


# ── Task detection ──────────────────────────────────────────────────────────

def _detect_task(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    n_unique = series.nunique()
    n_total = len(series)
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

def _engineer_features(df: pd.DataFrame, det: dict) -> tuple:
    df = df.copy()
    date_cols = det.get("date_cols", [])
    numeric_cols = det.get("numeric_cols", [])
    categorical_cols = det.get("categorical_cols", [])
    engineered = {}

    # Date extraction
    for col in date_cols:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_year"]  = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"]   = df[col].dt.day
            df[f"{col}_dow"]   = df[col].dt.dayofweek
            engineered[f"date_{col}"] = ["year", "month", "day", "dayofweek"]

    # Label-encode categoricals
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Drop raw date columns
    for col in date_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df.drop(columns=[col], inplace=True)

    # Rolling/lag for time-series
    if det.get("is_timeseries") and numeric_cols:
        for col in numeric_cols[:3]:
            if col in df.columns:
                df[f"{col}_roll3"] = df[col].rolling(3, min_periods=1).mean()
                df[f"{col}_lag1"]  = df[col].shift(1).fillna(method="bfill")
                df[f"{col}_lag3"]  = df[col].shift(3).fillna(method="bfill")
                engineered[f"rolling_{col}"] = ["roll3", "lag1", "lag3"]

    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df, feature_cols, engineered


def _add_polynomial_features(X_train, X_test, feature_cols, top_n=5):
    """Add degree-2 polynomial interactions for the top N features."""
    try:
        if len(feature_cols) < 2:
            return X_train, X_test, feature_cols
        top = min(top_n, len(feature_cols))
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        # Only on top features to avoid explosion
        X_tr_top = X_train[:, :top]
        X_te_top = X_test[:, :top]
        X_tr_poly = poly.fit_transform(X_tr_top)
        X_te_poly = poly.transform(X_te_top)
        X_train_out = np.hstack([X_train, X_tr_poly[:, top:]])
        X_test_out  = np.hstack([X_test,  X_te_poly[:, top:]])
        poly_names = [f"poly_{i}" for i in range(X_tr_poly.shape[1] - top)]
        return X_train_out, X_test_out, feature_cols + poly_names
    except Exception:
        return X_train, X_test, feature_cols


# ── Model catalogue ─────────────────────────────────────────────────────────

def _get_models(task_type: str):
    if task_type == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
            ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
            ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ]
        if HAS_XGB:
            models.append(("XGBoost", XGBClassifier(
                n_estimators=100, random_state=42, verbosity=0,
                use_label_encoder=False, eval_metric="logloss"
            )))
        if HAS_LGBM:
            models.append(("LightGBM", LGBMClassifier(
                n_estimators=100, random_state=42, verbosity=-1
            )))
        if HAS_CATBOOST:
            models.append(("CatBoost", CatBoostClassifier(
                iterations=100, random_state=42, verbose=False, allow_writing_files=False
            )))
    else:
        models = [
            ("Ridge Regression",  Ridge(alpha=1.0)),
            ("Random Forest",     RandomForestRegressor(n_estimators=100, random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]
        if HAS_XGB:
            models.append(("XGBoost Regressor", XGBRegressor(
                n_estimators=100, random_state=42, verbosity=0
            )))
        if HAS_LGBM:
            models.append(("LightGBM Regressor", LGBMRegressor(
                n_estimators=100, random_state=42, verbosity=-1
            )))
        if HAS_CATBOOST:
            models.append(("CatBoost Regressor", CatBoostRegressor(
                iterations=100, random_state=42, verbose=False, allow_writing_files=False
            )))
    return models


def _safe_cv(model, X_train, y_train, scoring: str, task_type: str):
    """Cross-validated score on the TRAINING split only. Returns (mean, std, folds, note).
    Returns (None, None, 0, reason) when CV is not statistically valid — never a fake 0.0."""
    from sklearn.base import clone
    if task_type == "classification":
        _, counts = np.unique(y_train, return_counts=True)
        folds = int(min(3, counts.min())) if len(counts) else 0
        if folds < 2:
            return None, None, 0, "cross-validation skipped: smallest class in the training split has < 2 members"
    else:
        folds = 3 if len(y_train) >= 30 else 0
        if folds < 2:
            return None, None, 0, "cross-validation skipped: too few training rows"
    try:
        cv = cross_val_score(clone(model), X_train, y_train, cv=folds, scoring=scoring)
        return round(float(cv.mean()), 4), round(float(cv.std()), 4), folds, ""
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
        cv = cross_validate(clone(model), X_train, y_train, cv=folds, scoring={"acc": "accuracy", "f1m": "f1_macro"})
        return (round(float(cv["test_acc"].mean()), 4), round(float(cv["test_acc"].std()), 4),
                round(float(cv["test_f1m"].mean()), 4), folds, "")
    except Exception as e:
        return None, None, None, 0, f"cross-validation failed: {type(e).__name__}"


def _eval_metrics(model, X_train, X_test, y_train, y_test, task_type: str, n_classes: int = 0) -> dict:
    y_pred = model.predict(X_test)
    if task_type == "classification":
        avg = "weighted"
        labels = list(range(n_classes)) if n_classes else None
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
        cv_mean, cv_std, cv_f1_macro, folds, cv_note = _safe_cv_classification(model, X_train, y_train)
        return {
            "accuracy":          round(float(accuracy_score(y_test, y_pred)), 4),
            "precision":         round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "recall":            round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "f1":                round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "f1_macro":          round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "confusion_matrix":  cm,
            "cv_score":          cv_mean,
            "cv_std":            cv_std,
            "cv_f1_macro":       cv_f1_macro,
            "cv_folds":          folds,
            "cv_note":           cv_note,
            "n_test":            int(len(y_test)),
            # precision/recall/f1 above are weighted by class size; f1_macro counts every class equally
            "averaging":         "precision/recall/f1 = weighted by class size; f1_macro = unweighted mean over classes",
        }
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    cv_mean, cv_std, folds, cv_note = _safe_cv(model, X_train, y_train, "r2", task_type)
    return {
        "r2":       round(float(r2_score(y_test, y_pred)), 4),
        "mae":      round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse":     round(rmse, 4),
        "cv_score": cv_mean,
        "cv_std":   cv_std,
        "cv_folds": folds,
        "cv_note":  cv_note,
        "n_test":   int(len(y_test)),
    }


def _plan_split(y, task_type: str, test_size: float) -> dict:
    """Decide whether a stratified split is valid BEFORE calling train_test_split."""
    plan = {"stratify": False, "warnings": [], "class_counts": {}}
    if task_type != "classification":
        return plan
    classes, counts = np.unique(y, return_counts=True)
    plan["class_counts"] = {int(c): int(n) for c, n in zip(classes, counts)}
    n_test = int(np.ceil(len(y) * test_size))
    if counts.min() >= 2 and n_test >= len(classes) and (len(y) - n_test) >= len(classes):
        plan["stratify"] = True
    else:
        rare = [int(c) for c, n in zip(classes, counts) if n < 2]
        if rare:
            plan["warnings"].append(
                f"{len(rare)} target class(es) have only 1 member; a stratified split is impossible, "
                "so a plain random split was used. Rare classes were kept (not removed); "
                "metrics for them are not reliable."
            )
        elif n_test < len(classes):
            plan["warnings"].append("Test split is smaller than the number of classes; a plain random split was used.")
    few = [int(c) for c, n in zip(classes, counts) if 2 <= n < 5]
    if few:
        plan["warnings"].append(f"{len(few)} class(es) have fewer than 5 members — per-class metrics are unreliable.")
    return plan


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


def _get_feature_importance(model, X_sample: np.ndarray, feature_cols: list,
                            X_eval=None, y_eval=None, scoring=None) -> list:
    importances = []

    if HAS_SHAP and not _shap_unsafe(model):
        try:
            explainer   = shap.Explainer(model, X_sample[:100])
            shap_values = explainer(X_sample[:100])
            vals = np.abs(shap_values.values)
            if vals.ndim == 3:
                vals = vals.mean(axis=2)
            mean_shap = vals.mean(axis=0)
            for col, imp in zip(feature_cols, mean_shap):
                importances.append({"feature": col, "importance": round(float(imp), 6), "method": "shap"})
            importances.sort(key=lambda x: x["importance"], reverse=True)
            return importances[:15]
        except Exception:
            pass

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        # align to min length in case poly added cols
        pairs = list(zip(feature_cols, fi))[:len(feature_cols)]
        for col, imp in pairs:
            importances.append({"feature": col, "importance": round(float(imp), 6), "method": "gini"})
        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances[:15]

    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).flatten()[:len(feature_cols)]
        for col, imp in zip(feature_cols, coef):
            importances.append({"feature": col, "importance": round(float(imp), 6), "method": "coefficient"})
        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances[:15]

    perm = _permutation_importance_fallback(model, X_eval, y_eval, feature_cols, scoring)
    if perm:
        return perm
    return []   # no honest importance available for this model type — never fabricate scores


# ── Future prediction ───────────────────────────────────────────────────────

def _future_prediction(model, scaler, feature_cols, df_feat, target_col, steps=10):
    try:
        historical_y = df_feat[target_col].values.tolist()
        recent = historical_y[-min(10, len(historical_y)):]
        trend_per_step = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        recent_std = float(np.std(recent)) if len(recent) > 1 else 0.0
        lag1 = f"{target_col}_lag1"
        lag3 = f"{target_col}_lag3"
        roll3 = f"{target_col}_roll3"
        base_row = df_feat[feature_cols].iloc[-1].copy()
        history = historical_y[:]
        future_y, conf_upper, conf_lower = [], [], []
        for i in range(steps):
            row = base_row.copy()
            if lag1 in row.index:
                row[lag1] = history[-1]
            if lag3 in row.index and len(history) >= 3:
                row[lag3] = history[-3]
            if roll3 in row.index:
                row[roll3] = float(np.mean(history[-3:]))
            row_scaled = scaler.transform([row.values])
            pred = float(model.predict(row_scaled)[0])
            trend_adjusted = pred + trend_per_step * min(i, 3) * 0.25
            future_y.append(round(trend_adjusted, 4))
            band = max(abs(trend_adjusted) * 0.05, recent_std * 0.5)
            conf_upper.append(round(trend_adjusted + band, 4))
            conf_lower.append(round(trend_adjusted - band, 4))
            history.append(trend_adjusted)

        return {
            "available": True,
            "future_y":        future_y,
            "future_upper":    conf_upper,
            "future_lower":    conf_lower,
            "confidence_upper": conf_upper,
            "confidence_lower": conf_lower,
            "historical_y":    [round(float(v), 4) for v in historical_y[-30:]],
            "future_steps":    steps,
            "trend_per_step":  round(trend_per_step, 4),
            "trend_direction": "up" if trend_per_step > 0 else "down",
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

def run_ml_engineer(df: pd.DataFrame, det: dict, ana: dict, mode_override: str = "") -> dict:
    mode_override = (mode_override or "").strip().lower()

    # Explicit routing to clustering or forecasting, requested by the user or
    # auto-selected by the analyze pipeline (see app.py "mode" form field).
    if mode_override == "clustering":
        return _clustering_agent.run_clustering(df, det)

    if mode_override == "forecasting":
        ml_rec = ana.get("ml_recommendation", {})
        target_col = _select_target(df, det, ml_rec.get("target_column", ""))
        date_cols = det.get("date_cols", [])
        return _forecasting_agent.run_forecasting(
            df, det, target_col=target_col or "", date_col=(date_cols[0] if date_cols else "")
        )

    # Exploratory mode — skip training unless clustering was explicitly requested above
    if det.get("exploratory_mode") or ana.get("ml_recommendation", {}).get("task_type") == "exploratory":
        return _exploratory_output(df, det)

    ml_rec      = ana.get("ml_recommendation", {})
    target_col  = _select_target(df, det, ml_rec.get("target_column", ""))
    if not target_col or target_col not in df.columns:
        return {"error": "No valid target column", "models": []}

    # Feature engineering
    df_feat, feature_cols, engineered = _engineer_features(df, det)
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    if not feature_cols:
        return {"error": "No features available after engineering", "models": []}

    # Drop columns that cannot carry generalisable signal: identifiers & constants.
    # (Label-encoded IDs/names behave like noise or leak row identity.)
    skip = set(det.get("identifier_columns", []) or []) | set(det.get("constant_columns", []) or [])
    dropped_features = [c for c in feature_cols if c in skip and c != target_col]
    feature_cols = [c for c in feature_cols if c not in skip]
    if not feature_cols:
        return {"error": "No usable features after removing identifier/constant columns", "models": []}

    # Rows whose target was missing in the ORIGINAL file must not be trained on with an imputed label.
    df_raw = det.get("df_raw")
    target_missing_dropped = 0
    row_ok = np.ones(len(df_feat), dtype=bool)
    if isinstance(df_raw, pd.DataFrame) and target_col in df_raw.columns and len(df_raw) == len(df_feat):
        row_ok = df_raw[target_col].notna().values
        target_missing_dropped = int((~row_ok).sum())
    X = df_feat.loc[row_ok, feature_cols].values
    y_raw = (df_feat[target_col] if target_col in df_feat.columns else df[target_col]).values[row_ok]

    detected_task = _detect_task(pd.Series(y_raw))
    analyst_task = (ml_rec.get("task_type") or "").strip().lower()
    if analyst_task == "exploratory":
        task_type = "exploratory"
    elif detected_task == "classification":
        task_type = "classification"
    elif analyst_task in ("classification", "regression"):
        task_type = analyst_task
    else:
        task_type = detected_task

    le_target = None
    if task_type == "classification":
        le_target = LabelEncoder()
        y = le_target.fit_transform(pd.Series(y_raw).astype(str).values)
        # Feature engineering label-encodes text targets, so le_target only knows "0","1",... here.
        # Restore the ORIGINAL class names for display (codes/y are untouched), so confusion matrices
        # and per-class output say "Churned"/"Retained" instead of "0"/"1".
        try:
            if target_col in df.columns and len(df) == len(df_feat) and not pd.api.types.is_numeric_dtype(df[target_col]):
                pair = pd.DataFrame({"code": pd.Series(y_raw).astype(str).values,
                                     "name": pd.Series(df[target_col].values[row_ok]).astype(str).values}).drop_duplicates()
                if pair["code"].is_unique and pair["name"].is_unique:
                    names = dict(zip(pair["code"], pair["name"]))
                    le_target.classes_ = np.array([names[c] for c in le_target.classes_], dtype=object)
        except Exception:
            pass   # display-only nicety; never break training
    else:
        y = y_raw.astype(float)

    mask = ~np.isnan(y.astype(float)) if task_type == "regression" else np.ones(len(y), dtype=bool)
    target_missing_dropped += int((~mask).sum())
    X, y = X[mask], y[mask]
    if len(X) < 20:
        return {"error": "Not enough data rows (need ≥ 20)", "models": []}

    eval_warnings = []
    if target_missing_dropped:
        eval_warnings.append(f"{target_missing_dropped} row(s) with a missing target were excluded from model training (not imputed).")
    if dropped_features:
        eval_warnings.append(f"Identifier/constant columns excluded from features: {', '.join(dropped_features[:8])}.")

    n_classes = 0
    if task_type == "classification":
        n_classes = int(len(np.unique(y)))
        if n_classes < 2:
            return {
                "error": "Classification evaluation impossible: the target has only one class.",
                "models": [], "task_type": "classification", "target_column": target_col,
                "classification_skipped": True, "evaluation_warnings": eval_warnings,
                "class_imbalance": _class_imbalance(y),
            }

    imbalance = _class_imbalance(y) if task_type == "classification" else {}

    test_size = 0.2 if len(X) > 100 else 0.15
    plan = _plan_split(y, task_type, test_size)
    eval_warnings += plan["warnings"]
    split_kwargs = {"test_size": test_size, "random_state": 42}
    if det.get("is_timeseries") and task_type == "regression":
        split_kwargs["shuffle"] = False
    elif plan["stratify"]:
        split_kwargs["stratify"] = y
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, **split_kwargs)
    except ValueError as e:
        split_kwargs.pop("stratify", None)
        eval_warnings.append(f"Stratified split failed ({e}); a plain random split was used.")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, **split_kwargs)

    # Speed cap: RandomForest / GradientBoosting (and their k-fold CV) scale super-linearly, so on very large
    # data (100k rows took >4.5 min) model *selection* runs on a sample of the TRAINING split only. The hold-out
    # test set is untouched, and all rows are still used for profiling, analysis and chat calculations.
    max_train = max(1000, int(os.environ.get("DATAMIND_MAX_TRAIN_ROWS", "20000")))
    train_rows_before_cap = int(len(y_train))
    train_cap_note = None
    if train_rows_before_cap > max_train:
        if det.get("is_timeseries") and task_type == "regression":
            X_train_raw, y_train = X_train_raw[-max_train:], y_train[-max_train:]      # keep the most recent rows
            how = "the most recent"
        else:
            sub = {"train_size": max_train, "random_state": 42}
            if split_kwargs.get("stratify") is not None:
                sub["stratify"] = y_train
            try:
                X_train_raw, _, y_train, _ = train_test_split(X_train_raw, y_train, **sub)
                how = "a stratified random" if "stratify" in sub else "a random"
            except ValueError:
                sub.pop("stratify", None)
                X_train_raw, _, y_train, _ = train_test_split(X_train_raw, y_train, **sub)
                how = "a random"
        train_cap_note = (f"Large dataset: models were fitted and cross-validated on {how} sample of {max_train:,} of the "
                          f"{train_rows_before_cap:,} training rows to keep runtime reasonable. The hold-out test set "
                          f"({len(y_test):,} rows) is untouched; profiling, analysis and chat calculations use all rows.")
        eval_warnings.append(train_cap_note)

    # Fit the scaler on the TRAINING split only (no test-set leakage).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_scaled = X_train   # sample used below for importance/SHAP: real trained-model feature space
    poly_feature_cols = feature_cols[:]

    primary_metric = "accuracy" if task_type == "classification" else "r2"
    # On imbalanced classes, "highest accuracy" is won by a model that only predicts the majority class.
    # Select by cross-validated MACRO-F1 there (every class counts equally); the reported score stays accuracy.
    # "Clearly not balanced" = smallest class < 70% the size of the largest (churn at 27/73 qualifies; the
    # separate `imbalanced` warning flag keeps its stricter < 30% threshold).
    _ratio = imbalance.get("ratio", 1.0) if task_type == "classification" else 1.0
    use_macro_selection = task_type == "classification" and _ratio < MACRO_SELECTION_RATIO
    if use_macro_selection:
        eval_warnings.append(f"Classes are unevenly sized (smallest/largest = {_ratio:.2f}): the best model was chosen by "
                             f"cross-validated macro F1 (every class counts equally) instead of accuracy, so predicting only "
                             f"the majority class cannot win by default.")
    evaluation_protocol = {
        "split": "hold-out test set",
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_train_before_cap": train_rows_before_cap, "train_sample_cap": max_train if train_cap_note else None,
        "stratified": bool(split_kwargs.get("stratify") is not None),
        "random_state": 42,
        "reported_metric": primary_metric,
        "selection_rule": ("highest cross-validated macro F1 on the training split (imbalanced classes; falls back to hold-out macro F1 if CV is not valid)"
                           if use_macro_selection else
                           "highest cross-validated score on the training split (falls back to hold-out score if CV is not valid)"),
    }

    model_results, best_model_obj, best_sel, best_name = [], None, -np.inf, ""
    for name, clf in _get_models(task_type):
        try:
            clf.fit(X_train, y_train)
            metrics = _eval_metrics(clf, X_train, X_test, y_train, y_test, task_type, n_classes)
            holdout = metrics.get(primary_metric)
            cvm = metrics.get("cv_score")
            if use_macro_selection:
                cvf = metrics.get("cv_f1_macro")
                sel = cvf if cvf is not None else metrics.get("f1_macro")
            else:
                sel = cvm if cvm is not None else holdout
            # "score" is ALWAYS the hold-out metric so comparison table == detailed metrics.
            model_results.append({
                "name": name, "score": round(float(holdout), 4), "score_metric": primary_metric,
                "cv_score": cvm, "selection_score": round(float(sel), 4),
                "metrics": metrics, "is_best": False, "_obj": clf,
            })
            if sel > best_sel:
                best_sel, best_model_obj, best_name = sel, clf, name
        except Exception as e:
            model_results.append({"name": name, "score": 0, "metrics": {}, "is_best": False, "error": str(e)})

    for m in model_results:
        m["is_best"] = (m["name"] == best_name)
    best_entry = next((m for m in model_results if m["is_best"]), None)
    best_score = float(best_entry["score"]) if best_entry else 0.0

    # Weighted metrics hide poor minority-class results; say so when the gap is large (informational, not a hold).
    if best_entry and task_type == "classification":
        bm_ = best_entry.get("metrics") or {}
        if bm_.get("f1") is not None and bm_.get("f1_macro") is not None and bm_["f1"] - bm_["f1_macro"] >= 0.2:
            eval_warnings.append(f"Weighted F1 ({bm_['f1']:.0%}) is much higher than macro F1 ({bm_['f1_macro']:.0%}): "
                                 f"the model predicts the larger classes well but the smaller classes poorly.")

    feature_importances = []
    if best_model_obj is not None:
        _imp_scoring = ("f1_macro" if use_macro_selection else "accuracy") if task_type == "classification" else "r2"
        feature_importances = _get_feature_importance(best_model_obj, X_scaled[:200], poly_feature_cols,
                                                      X_eval=X_test, y_eval=y_test, scoring=_imp_scoring)

    shap_plots = {"available": False, "reason": "not computed"}
    if best_model_obj is not None:
        try:
            from agents.explainability import generate_shap_plots
            shap_plots = generate_shap_plots(best_model_obj, X_scaled[:200], poly_feature_cols, task_type)
        except Exception as e:
            shap_plots = {"available": False, "reason": str(e)}

    future_pred = {}
    if det.get("is_timeseries") and task_type == "regression" and best_model_obj is not None:
        future_pred = _future_prediction(best_model_obj, scaler, feature_cols, df_feat, target_col)

    clean_models = [{k: v for k, v in m.items() if k != "_obj"} for m in model_results]

    return {
        "models":               clean_models,
        "task_type":            task_type,
        "target_column":        target_col,
        "best_model":           best_name,
        "best_score":           round(float(best_score), 4),
        "feature_importances":  feature_importances,
        "shap_plots":           shap_plots,
        "engineered_features":  list(engineered.keys()),
        "feature_count":        len(feature_cols),
        "future_prediction":    future_pred,
        "class_imbalance":      imbalance,
        "exploratory_mode":     False,
        "score_metric":         primary_metric,
        "evaluation_protocol":  evaluation_protocol,
        "evaluation_warnings":  eval_warnings,
        "selection_metric":     "cv_f1_macro" if use_macro_selection else "cv_score",
        # Private objects
        "_model_obj":    best_model_obj,
        "_scaler_obj":   scaler,
        "_feature_cols": feature_cols,
        "_le_target":    le_target,
        "_numeric_cols": det.get("numeric_cols", []),
    }
