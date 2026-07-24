"""
Agent 3c: Forecasting Engine — univariate time-series forecasting.

Fits ARIMA, SARIMA, and (optionally) Prophet on a date-indexed numeric
series, backtests each on a held-out tail to compute MAE/RMSE, and
returns the best-performing model's forward forecast with confidence
intervals plus a leaderboard so the comparison is transparent.

Statsmodels (ARIMA/SARIMA) is treated as a required-ish soft dependency;
Prophet is optional since it's a heavier install that some free hosting
tiers (Render free, Railway hobby) can struggle to build.
"""
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def _infer_seasonal_period(n_points: int) -> int:
    """Rough heuristic: weekly data -> 7, monthly-ish -> 12, else no strong seasonality assumed."""
    if n_points >= 24:
        return 12
    if n_points >= 14:
        return 7
    return 0


def _mae_rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return round(mae, 4), round(rmse, 4)


def _backtest_arima(series: pd.Series, order, holdout: int):
    train, test = series[:-holdout], series[-holdout:]
    model = ARIMA(train, order=order).fit()
    pred = model.forecast(steps=holdout)
    mae, rmse = _mae_rmse(test.values, pred.values)
    return mae, rmse


def _backtest_sarima(series: pd.Series, order, seasonal_order, holdout: int):
    train, test = series[:-holdout], series[-holdout:]
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred = model.forecast(steps=holdout)
    mae, rmse = _mae_rmse(test.values, pred.values)
    return mae, rmse


def _backtest_prophet(df_prophet: pd.DataFrame, holdout: int):
    train, test = df_prophet.iloc[:-holdout], df_prophet.iloc[-holdout:]
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(train)
    future = m.make_future_dataframe(periods=holdout, freq="D")
    fc = m.predict(future)
    pred = fc["yhat"].iloc[-holdout:].values
    mae, rmse = _mae_rmse(test["y"].values, pred)
    return mae, rmse, m


def run_forecasting(df: pd.DataFrame, det: dict, target_col: str, date_col: str = "", steps: int = 12) -> dict:
    """
    Main entry. Returns a dict shaped to match run_ml_engineer's output keys
    where reasonable, with task_type="forecasting" so downstream code can
    branch on it.
    """
    if not date_col:
        date_cols = det.get("date_cols", [])
        date_col = date_cols[0] if date_cols else ""
    if not date_col or date_col not in df.columns or target_col not in df.columns:
        return {"error": "Need a date column and a numeric target column for forecasting",
                "models": [], "task_type": "forecasting"}

    work = df[[date_col, target_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, target_col]).sort_values(date_col)
    work = work.groupby(date_col, as_index=False)[target_col].mean()  # collapse duplicate timestamps

    if len(work) < 15:
        return {"error": "Need at least 15 time-ordered observations for forecasting",
                "models": [], "task_type": "forecasting"}

    series = pd.Series(work[target_col].values, index=pd.RangeIndex(len(work)))
    holdout = max(3, min(10, len(series) // 5))
    seasonal_period = _infer_seasonal_period(len(series))

    leaderboard = []

    # ── ARIMA ────────────────────────────────────────────────────────────
    if HAS_STATSMODELS:
        try:
            mae, rmse = _backtest_arima(series, order=(2, 1, 2), holdout=holdout)
            leaderboard.append({"name": "ARIMA(2,1,2)", "algorithm": "arima", "mae": mae, "rmse": rmse, "is_best": False})
        except Exception as e:
            leaderboard.append({"name": "ARIMA(2,1,2)", "algorithm": "arima", "error": str(e), "is_best": False})

        # ── SARIMA (only if there's enough data for a seasonal cycle) ───
        if seasonal_period and len(series) > seasonal_period * 2:
            try:
                mae, rmse = _backtest_sarima(
                    series, order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, seasonal_period), holdout=holdout,
                )
                leaderboard.append({
                    "name": f"SARIMA(1,1,1)x(1,1,1,{seasonal_period})", "algorithm": "sarima",
                    "mae": mae, "rmse": rmse, "is_best": False,
                })
            except Exception as e:
                leaderboard.append({"name": "SARIMA", "algorithm": "sarima", "error": str(e), "is_best": False})
    else:
        leaderboard.append({
            "name": "ARIMA/SARIMA", "algorithm": "arima",
            "error": "statsmodels not installed — run: pip install statsmodels", "is_best": False,
        })

    # ── Prophet (optional) ───────────────────────────────────────────────
    prophet_model = None
    if HAS_PROPHET:
        try:
            df_p = pd.DataFrame({"ds": work[date_col].values, "y": work[target_col].values})
            mae, rmse, prophet_model = _backtest_prophet(df_p, holdout)
            leaderboard.append({"name": "Prophet", "algorithm": "prophet", "mae": mae, "rmse": rmse, "is_best": False})
        except Exception as e:
            leaderboard.append({"name": "Prophet", "algorithm": "prophet", "error": str(e), "is_best": False})
    else:
        leaderboard.append({
            "name": "Prophet", "algorithm": "prophet",
            "error": "prophet not installed — run: pip install prophet", "is_best": False,
        })

    valid = [m for m in leaderboard if "mae" in m]
    if not valid:
        return {"error": "All forecasting models failed or are unavailable. Install statsmodels at minimum.",
                "models": leaderboard, "task_type": "forecasting"}

    best = min(valid, key=lambda m: m["mae"])  # lower error = better
    best["is_best"] = True

    # ── Refit the best model on the FULL series and forecast forward ────
    future_y, lower, upper = [], [], []
    try:
        if best["algorithm"] == "arima":
            full_model = ARIMA(series, order=(2, 1, 2)).fit()
            fc = full_model.get_forecast(steps=steps)
            future_y = [round(float(v), 4) for v in fc.predicted_mean.values]
            ci = fc.conf_int(alpha=0.05)
            lower = [round(float(v), 4) for v in ci.iloc[:, 0].values]
            upper = [round(float(v), 4) for v in ci.iloc[:, 1].values]
        elif best["algorithm"] == "sarima":
            full_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc = full_model.get_forecast(steps=steps)
            future_y = [round(float(v), 4) for v in fc.predicted_mean.values]
            ci = fc.conf_int(alpha=0.05)
            lower = [round(float(v), 4) for v in ci.iloc[:, 0].values]
            upper = [round(float(v), 4) for v in ci.iloc[:, 1].values]
        elif best["algorithm"] == "prophet" and prophet_model is not None:
            df_p_full = pd.DataFrame({"ds": work[date_col].values, "y": work[target_col].values})
            m_full = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m_full.fit(df_p_full)
            future_df = m_full.make_future_dataframe(periods=steps, freq="D")
            fc = m_full.predict(future_df)
            tail = fc.iloc[-steps:]
            future_y = [round(float(v), 4) for v in tail["yhat"].values]
            lower = [round(float(v), 4) for v in tail["yhat_lower"].values]
            upper = [round(float(v), 4) for v in tail["yhat_upper"].values]
    except Exception as e:
        return {"error": f"Best model ({best['name']}) failed to refit: {e}",
                "models": leaderboard, "task_type": "forecasting"}

    last_date = work[date_col].iloc[-1]
    try:
        freq = pd.infer_freq(work[date_col]) or "D"
    except Exception:
        freq = "D"
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]

    return {
        "task_type": "forecasting",
        "models": leaderboard,
        "best_model": best["name"],
        "best_score": best.get("mae", 0),
        "scoring_metric": "MAE (lower is better)",
        "target_column": target_col,
        "date_column": date_col,
        "historical_dates": [d.strftime("%Y-%m-%d") for d in work[date_col]],
        "historical_y": [round(float(v), 4) for v in work[target_col].values],
        "future_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "future_y": future_y,
        "future_lower": lower,
        "future_upper": upper,
        "future_steps": steps,
        "feature_importances": [],
        "engineered_features": [],
        "future_prediction": {
            "available": bool(future_y),
            "future_y": future_y, "future_lower": lower, "future_upper": upper,
            "future_steps": steps, "target": target_col,
        },
        "class_imbalance": {},
        "exploratory_mode": False,
        "feature_count": 1,
        "_model_obj": None,
        "_scaler_obj": None,
        "_feature_cols": [target_col],
        "_le_target": None,
        "_numeric_cols": det.get("numeric_cols", []),
    }
