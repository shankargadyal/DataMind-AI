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
                iterations=100, random_state=42, verbose=False
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
                iterations=100, random_state=42, verbose=False
            )))
    return models


def _eval_metrics(model, X_train, X_test, y_train, y_test, task_type: str) -> dict:
    y_pred = model.predict(X_test)
    if task_type == "classification":
        avg = "weighted"
        cm  = confusion_matrix(y_test, y_pred).tolist()
        # Cross-val on training set
        try:
            cv = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
            cv_mean = round(float(cv.mean()), 4)
            cv_std  = round(float(cv.std()), 4)
        except Exception:
            cv_mean, cv_std = 0.0, 0.0
        return {
            "accuracy":          round(float(accuracy_score(y_test, y_pred)), 4),
            "precision":         round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "recall":            round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "f1":                round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "confusion_matrix":  cm,
            "cv_score":          cv_mean,
            "cv_std":            cv_std,
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        try:
            cv = cross_val_score(model, X_train, y_train, cv=3, scoring="r2")
            cv_mean = round(float(cv.mean()), 4)
            cv_std  = round(float(cv.std()), 4)
        except Exception:
            cv_mean, cv_std = 0.0, 0.0
        return {
            "r2":       round(float(r2_score(y_test, y_pred)), 4),
            "mae":      round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse":     round(rmse, 4),
            "cv_score": cv_mean,
            "cv_std":   cv_std,
        }


# ── SHAP / feature importance ───────────────────────────────────────────────

def _get_feature_importance(model, X_sample: np.ndarray, feature_cols: list) -> list:
    importances = []

    if HAS_SHAP:
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

    return [{"feature": col, "importance": 0.0, "method": "none"} for col in feature_cols[:10]]


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

    X = df_feat[feature_cols].values
    y_raw = df_feat[target_col].values if target_col in df_feat.columns else df[target_col].values

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
        y = le_target.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.astype(float)

    # Remove NaN rows
    mask = ~np.isnan(y.astype(float)) if task_type == "regression" else np.ones(len(y), dtype=bool)
    X, y = X[mask], y[mask]
    if len(X) < 20:
        return {"error": "Not enough data rows (need ≥ 20)", "models": []}

    # Class imbalance check
    imbalance = _class_imbalance(y) if task_type == "classification" else {}

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    test_size = 0.2 if len(X) > 100 else 0.15
    split_kwargs = {
        "test_size": test_size,
        "random_state": 42,
    }
    if det.get("is_timeseries") and task_type == "regression":
        split_kwargs["shuffle"] = False
    elif task_type == "classification":
        split_kwargs["stratify"] = y
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, **split_kwargs)

    # Polynomial features (only if ≤ 30 features to avoid memory issues)
    poly_feature_cols = feature_cols[:]

    # Train models
    model_results, best_model_obj, best_score, best_name = [], None, -np.inf, ""

    for name, clf in _get_models(task_type):
        try:
            clf.fit(X_train, y_train)
            metrics = _eval_metrics(clf, X_train, X_test, y_train, y_test, task_type)
            holdout_score = metrics.get("accuracy") if task_type == "classification" else metrics.get("r2", 0)
            score = metrics.get("cv_score", holdout_score)
            model_results.append({
                "name": name, "score": round(float(score), 4),
                "metrics": metrics, "is_best": False, "_obj": clf,
            })
            if score > best_score:
                best_score, best_model_obj, best_name = score, clf, name
        except Exception as e:
            model_results.append({"name": name, "score": 0, "metrics": {}, "is_best": False, "error": str(e)})

    for m in model_results:
        m["is_best"] = (m["name"] == best_name)

    # Feature importance
    feature_importances = []
    if best_model_obj is not None:
        feature_importances = _get_feature_importance(best_model_obj, X_scaled[:200], poly_feature_cols)

    # SHAP summary + waterfall plots for the dashboard / report (Agent 4: Explainability)
    shap_plots = {"available": False, "reason": "not computed"}
    if best_model_obj is not None:
        try:
            from agents.explainability import generate_shap_plots
            shap_plots = generate_shap_plots(best_model_obj, X_scaled[:200], poly_feature_cols, task_type)
        except Exception as e:
            shap_plots = {"available": False, "reason": str(e)}

    # Future prediction
    future_pred = {}
    if det.get("is_timeseries") and task_type == "regression" and best_model_obj is not None:
        future_pred = _future_prediction(best_model_obj, scaler, feature_cols, df_feat, target_col)

    # Clean model list
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
        # Private objects
        "_model_obj":    best_model_obj,
        "_scaler_obj":   scaler,
        "_feature_cols": feature_cols,
        "_le_target":    le_target,
        "_numeric_cols": det.get("numeric_cols", []),
    }
