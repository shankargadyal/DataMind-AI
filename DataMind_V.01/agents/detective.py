"""
Agent 1: Data Detective v3.0 — CSV/Excel cleaning, profiling, outlier detection.
New in v3:
  - Excel (.xlsx/.xls) support
  - IQR-based outlier detection
  - Smarter target column suggestion
  - Exploratory Analysis fallback mode (no target → EDA only)
  - Dataset type hint (timeseries / classification / regression)
  - Column-level completeness scores
"""
import pandas as pd
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_date_col(series: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().head(50).astype(str)
        hits = sum(1 for v in sample if _try_parse_date(v))
        return hits / max(len(sample), 1) > 0.7
    return False


def _try_parse_date(v: str) -> bool:
    try:
        pd.to_datetime(v)
        return True
    except Exception:
        return False


def _detect_outliers_iqr(series: pd.Series) -> dict:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (series < lower) | (series > upper)
    count = int(outlier_mask.sum())
    return {
        "count": count,
        "pct": round(count / max(len(series), 1) * 100, 2),
        "lower_fence": round(float(lower), 4),
        "upper_fence": round(float(upper), 4),
    }


def _suggest_target(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> str | None:
    """Heuristically pick the best target column."""
    target_keywords = ["target", "label", "outcome", "churn", "sales", "revenue",
                       "price", "profit", "score", "result", "class", "status",
                       "default", "fraud", "survived", "death", "diagnosis"]
    all_cols_lower = {c.lower(): c for c in df.columns}
    for kw in target_keywords:
        if kw in all_cols_lower:
            return all_cols_lower[kw]
    # Fall back to last numeric column
    if numeric_cols:
        return numeric_cols[-1]
    return None


def _detect_mixed_type_cols(df: pd.DataFrame, categorical_cols: list) -> dict:
    """Columns where values look inconsistently formatted (mixed casing, stray whitespace)
    are a Consistency problem — same real-world value represented multiple ways."""
    issues = {}
    for col in categorical_cols:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        normalized = s.str.strip().str.lower()
        raw_unique = s.nunique()
        norm_unique = normalized.nunique()
        if raw_unique > norm_unique:
            issues[col] = {
                "raw_unique": int(raw_unique),
                "normalized_unique": int(norm_unique),
                "duplicate_variants": int(raw_unique - norm_unique),
            }
    return issues


def _validity_checks(df: pd.DataFrame, numeric_cols: list, date_cols: list) -> dict:
    """Validity = do values respect the kind of constraints you'd expect for that column
    (no negative ages/counts, dates that actually parsed, etc.)."""
    total_checked, total_invalid = 0, 0
    details = {}
    negative_suspect_kw = ("age", "count", "quantity", "qty", "price", "amount", "salary",
                            "income", "revenue", "cost", "duration", "years", "population")
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        if any(kw in col.lower() for kw in negative_suspect_kw):
            invalid = int((s < 0).sum())
            total_checked += len(s)
            total_invalid += invalid
            if invalid:
                details[col] = {"issue": "negative values where none expected", "count": invalid}
    for col in date_cols:
        s = df[col]
        invalid = int(s.isna().sum())
        total_checked += len(s)
        total_invalid += invalid
        if invalid:
            details[col] = {"issue": "unparseable date values", "count": invalid}
    pct_valid = 100.0 if total_checked == 0 else round((1 - total_invalid / total_checked) * 100, 1)
    return {"score": pct_valid, "checked": total_checked, "invalid": total_invalid, "details": details}


def _compute_quality_dimensions(df: pd.DataFrame, original_shape: tuple, missing_before: int,
                                 numeric_cols: list, categorical_cols: list, date_cols: list,
                                 outlier_summary: dict) -> dict:
    """
    Five-dimension Data Quality Center score (each 0-100, weighted into one composite /100):
      Completeness — how much data isn't missing
      Accuracy     — how few extreme/implausible outliers there are (proxy: inverse outlier rate)
      Consistency  — how uniformly categorical values are formatted (no "USA"/"usa"/" USA " splits)
      Validity     — values respect basic real-world constraints (no negative ages, parseable dates)
      Uniqueness   — how few exact duplicate rows exist
    """
    total_cells = max(original_shape[0] * original_shape[1], 1)
    completeness = round((1 - missing_before / total_cells) * 100, 1)

    if outlier_summary:
        avg_outlier_pct = sum(v.get("pct", 0) for v in outlier_summary.values()) / len(outlier_summary)
    else:
        avg_outlier_pct = 0.0
    accuracy = round(max(0.0, 100 - avg_outlier_pct * 2), 1)  # outliers weighted 2x since they compound

    consistency_issues = _detect_mixed_type_cols(df, categorical_cols)
    if categorical_cols:
        flagged_cols = len(consistency_issues)
        consistency = round(max(0.0, 100 - (flagged_cols / len(categorical_cols)) * 100), 1)
    else:
        consistency = 100.0

    validity_result = _validity_checks(df, numeric_cols, date_cols)
    validity = validity_result["score"]

    dup_rows = int(df.duplicated().sum())
    uniqueness = round(max(0.0, (1 - dup_rows / max(original_shape[0], 1)) * 100), 1)

    weights = {"completeness": 0.30, "accuracy": 0.20, "consistency": 0.15, "validity": 0.20, "uniqueness": 0.15}
    composite = round(
        completeness * weights["completeness"] + accuracy * weights["accuracy"] +
        consistency * weights["consistency"] + validity * weights["validity"] +
        uniqueness * weights["uniqueness"], 1
    )
    grade = (
        "Excellent" if composite >= 90 else
        "Good" if composite >= 75 else
        "Fair" if composite >= 60 else
        "Poor"
    )

    recommendations = []
    if completeness < 90:
        recommendations.append(f"Completeness is {completeness}% — investigate why values are missing before trusting downstream models.")
    if accuracy < 85:
        recommendations.append(f"Outlier rate is elevated (accuracy score {accuracy}%) — review extreme values for data entry errors.")
    if consistency < 90 and consistency_issues:
        worst = max(consistency_issues.items(), key=lambda kv: kv[1]["duplicate_variants"])
        recommendations.append(f"'{worst[0]}' has inconsistent formatting (e.g. mixed casing/whitespace) — normalize before grouping or joining.")
    if validity < 95 and validity_result["details"]:
        col, info = next(iter(validity_result["details"].items()))
        recommendations.append(f"'{col}' has {info['count']} values that fail a basic sanity check ({info['issue']}).")
    if uniqueness < 98:
        recommendations.append(f"{dup_rows} exact duplicate rows found — consider deduplicating before analysis.")
    if not recommendations:
        recommendations.append("Data quality is strong across all five dimensions — no urgent cleanup needed.")

    return {
        "composite_score": composite,
        "grade": grade,
        "dimensions": {
            "completeness": completeness,
            "accuracy": accuracy,
            "consistency": consistency,
            "validity": validity,
            "uniqueness": uniqueness,
        },
        "radar_chart": {
            "labels": ["Completeness", "Accuracy", "Consistency", "Validity", "Uniqueness"],
            "values": [completeness, accuracy, consistency, validity, uniqueness],
        },
        "duplicate_rows": dup_rows,
        "consistency_issues": consistency_issues,
        "validity_issues": validity_result["details"],
        "recommendations": recommendations[:5],
    }


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_detective(filepath: str) -> dict:
    # ── Load file ────────────────────────────────────────────────
    try:
        if filepath.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        return {"error": str(e)}

    original_shape = df.shape
    missing_before = int(df.isnull().sum().sum())

    # ── Detect & parse date columns ──────────────────────────────
    date_cols = []
    for col in df.columns:
        if _is_date_col(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                date_cols.append(col)
            except Exception:
                pass

    # ── Separate numeric / categorical ───────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in date_cols
    ]

    # ── Fill missing values ──────────────────────────────────────
    missing_values_fixed = {}
    for col in numeric_cols:
        n = int(df[col].isnull().sum())
        if n:
            df[col] = df[col].fillna(df[col].median())
            missing_values_fixed[col] = n
    for col in categorical_cols:
        n = int(df[col].isnull().sum())
        if n:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
            missing_values_fixed[col] = n

    missing_after = int(df.isnull().sum().sum())

    # ── Quality score ────────────────────────────────────────────
    completeness = 1 - (missing_before / max(original_shape[0] * original_shape[1], 1))
    quality_score = round(completeness * 100, 1)

    # ── Column stats (with outliers for numeric) ─────────────────
    column_stats = {}
    outlier_summary = {}
    for col in numeric_cols:
        s = df[col]
        outs = _detect_outliers_iqr(s.dropna())
        outlier_summary[col] = outs
        column_stats[col] = {
            "type": "numeric",
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "missing": missing_values_fixed.get(col, 0),
            "unique": int(s.nunique()),
            "outlier_count": outs["count"],
            "outlier_pct": outs["pct"],
        }
    for col in categorical_cols:
        s = df[col]
        vc = s.value_counts().head(8)
        column_stats[col] = {
            "type": "categorical",
            "unique": int(s.nunique()),
            "top": str(s.mode()[0]) if not s.mode().empty else "",
            "missing": missing_values_fixed.get(col, 0),
            "top_values": [{"value": str(k), "count": int(v)} for k, v in vc.items()],
        }

    # ── Distribution data (for charts) ──────────────────────────
    distribution_data = []
    for col in numeric_cols[:8]:
        try:
            hist, edges = np.histogram(df[col].dropna(), bins=20)
            distribution_data.append({
                "column": col,
                "histogram": hist.tolist(),
                "edges": [round(float(e), 4) for e in edges.tolist()],
            })
        except Exception:
            pass

    # ── Correlation matrix ───────────────────────────────────────
    correlation = {}
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr().round(3)
            correlation = {col: corr[col].to_dict() for col in corr.columns}
        except Exception:
            pass

    # ── Time-series & mode detection ────────────────────────────
    is_timeseries = len(date_cols) > 0
    row_count = original_shape[0]
    size_hint = "large" if row_count > 50000 else ("medium" if row_count > 5000 else "small")

    # ── Suggested target column ──────────────────────────────────
    suggested_target = _suggest_target(df, numeric_cols, categorical_cols)

    # ── Exploratory mode flag (no clear target) ──────────────────
    exploratory_mode = suggested_target is None

    # ── Column completeness (per-column quality) ─────────────────
    total_rows = original_shape[0]
    column_completeness = {
        col: round((1 - df[col].isnull().sum() / max(total_rows, 1)) * 100, 1)
        for col in df.columns
    }

    # ── Top correlations (pairs) ─────────────────────────────────
    top_correlations = []
    if len(numeric_cols) >= 2:
        try:
            corr_df = df[numeric_cols].corr().abs()
            pairs = []
            for i in range(len(corr_df.columns)):
                for j in range(i + 1, len(corr_df.columns)):
                    c1, c2 = corr_df.columns[i], corr_df.columns[j]
                    pairs.append({"col1": c1, "col2": c2, "r": round(float(corr_df.iloc[i, j]), 3)})
            pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
            top_correlations = pairs[:10]
        except Exception:
            pass

    # ── Data Quality Center: five-dimension composite score ───────
    quality_dimensions = _compute_quality_dimensions(
        df, original_shape, missing_before, numeric_cols, categorical_cols, date_cols, outlier_summary
    )

    return {
        "df": df,
        "original_shape": list(original_shape),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
        "column_stats": column_stats,
        "distribution_data": distribution_data,
        "correlation": correlation,
        "top_correlations": top_correlations,
        "missing_values_fixed": missing_values_fixed,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "quality_score": quality_score,
        "quality_dimensions": quality_dimensions,
        "is_timeseries": is_timeseries,
        "size_hint": size_hint,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "suggested_target": suggested_target,
        "exploratory_mode": exploratory_mode,
        "outlier_summary": outlier_summary,
        "column_completeness": column_completeness,
    }