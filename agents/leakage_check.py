"""
Data-only relationship checks: which columns are copies / bins / lookups / ratios of other columns,
and which of those make a model look better than it is (target leakage).

No outside knowledge is needed (the app never sees how a CSV was produced); everything is inferred from
the values, with fixed seeds and a row cap so it is deterministic and cheap.

Findings (all plain dicts, JSON-safe):
  id_columns          columns that are row identifiers (excluded from every check)
  near_duplicates     numeric pairs with |Spearman| >= 0.95 ("monotonic_copy" at >= 0.995)
  derived_columns     a column that is (almost) a function of ONE or TWO other columns
                      kind: monotonic_copy | binned_version | lookup | pair_lookup
  target_leakage      features that make the target (almost) a formula of other columns, or that are
                      near-copies of the target:  kind: target_function | log_ratio | near_copy
  summary             short human-readable sentences
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

MAX_ROWS = 5000
NEAR_DUP_RHO = 0.95
COPY_RHO = 0.995
TARGET_COPY_RHO = 0.998   # a feature must RANK rows like the target; a strong predictor (e.g. 0.99) is not a leak
COEF_RANGE = (0.5, 1.5)    # ratio/product identities have both log-exponents of substantial size (~ +/-1)
CAT_ACC = 0.99            # accuracy at which a categorical column counts as "reproducible"
LOG_R2 = 0.95             # log-space R^2 at which target ~ ratio/product of two columns
MAX_CAT_LEVELS = 30
MAX_PAIR_LEVELS = 12


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _id_columns(df: pd.DataFrame) -> list[str]:
    ids = []
    n = len(df)
    for c in df.columns:
        s = df[c].dropna()
        if len(s) < 20 or s.nunique() < 0.99 * n:
            continue
        if _is_numeric(df[c]):
            if pd.api.types.is_integer_dtype(df[c]) or (s % 1 == 0).all():
                name_hit = any(k in str(c).lower() for k in ("id", "key", "code", "index", "no", "number"))
                srt = np.sort(s.values)
                steps = np.diff(srt)
                regular = len(steps) > 0 and (np.median(steps) >= 1) and (np.mean(steps == np.median(steps)) > 0.2 or name_hit)
                if name_hit or regular:
                    ids.append(c)
        else:
            ids.append(c)          # a text column that is unique per row is an identifier, not a feature
    return ids


def _encode(s: pd.Series) -> np.ndarray:
    if _is_numeric(s):
        return s.astype(float).fillna(s.median()).values
    return s.astype("category").cat.codes.values.astype(float)


def _cv_acc(X: np.ndarray, y: np.ndarray, depth: int) -> float:
    if len(np.unique(y)) < 2:
        return 1.0
    counts = np.unique(y, return_counts=True)[1]
    folds = int(min(3, counts.min()))
    if folds < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=0), X, y, cv=folds,
                                     scoring="accuracy").mean())


def analyze_relationships(df: pd.DataFrame, target: str | None = None, seed: int = 42) -> dict:
    out = {"id_columns": [], "near_duplicates": [], "derived_columns": [], "target_leakage": [], "summary": [],
           "rows_used": 0}
    try:
        if df is None or len(df) < 30 or df.shape[1] < 2:
            return out
        d = df.sample(MAX_ROWS, random_state=seed) if len(df) > MAX_ROWS else df
        d = d.reset_index(drop=True)
        out["rows_used"] = int(len(d))
        ids = _id_columns(d)
        out["id_columns"] = ids
        cols = [c for c in d.columns if c not in ids and d[c].nunique(dropna=True) > 1]
        num = [c for c in cols if _is_numeric(d[c])]
        cat = [c for c in cols if c not in num and d[c].nunique() <= MAX_CAT_LEVELS]
        # low-cardinality numerics (scores, ranks) can also be the OUTPUT of a lookup/binning
        lowcard_num = [c for c in num if d[c].nunique() <= MAX_PAIR_LEVELS]

        # ── 1. numeric ↔ numeric: monotonic copies / near-duplicates (Spearman) ──
        seen_pairs: set = set()
        if len(num) >= 2:
            rho = d[num].corr(method="spearman")
            for a, b in itertools.combinations(num, 2):
                r = rho.loc[a, b]
                if pd.notna(r) and abs(r) >= NEAR_DUP_RHO:
                    kind = "monotonic_copy" if abs(r) >= COPY_RHO else "near_duplicate"
                    out["near_duplicates"].append({"a": a, "b": b, "spearman": round(float(r), 4), "kind": kind})
                    seen_pairs.add(frozenset((a, b)))

        # ── 2. categorical / low-cardinality output reproducible from one numeric (binned) or one/two columns ──
        outputs = list(dict.fromkeys(cat + lowcard_num))
        enc = {c: _encode(d[c]) for c in cols}
        for o in outputs:
            y = d[o].astype("category").cat.codes.values
            best = None
            for f in cols:
                if f == o or frozenset((f, o)) in seen_pairs:
                    continue
                if _is_numeric(d[f]) and d[f].nunique() > MAX_CAT_LEVELS:
                    acc, depth, kind = _cv_acc(enc[f].reshape(-1, 1), y, 6), 6, "binned_version"
                else:
                    acc, depth, kind = _cv_acc(enc[f].reshape(-1, 1), y, 12), 12, "lookup"
                if acc >= CAT_ACC:
                    if best is None or acc > best["score"] + 1e-9:
                        best = {"column": o, "explained_by": [f], "kind": kind, "score": round(acc, 4), "metric": "accuracy", "alternatives": []}
                    elif abs(acc - best["score"]) <= 1e-9:
                        best["alternatives"].append(f)           # equally good source (e.g. a monotonic copy of the first)
            if best is None:
                small = [c for c in cols if c != o and d[c].nunique() <= MAX_PAIR_LEVELS]
                for f1, f2 in itertools.combinations(small, 2):
                    acc = _cv_acc(np.c_[enc[f1], enc[f2]], y, 12)
                    if acc >= CAT_ACC and (best is None or acc > best["score"]):
                        best = {"column": o, "explained_by": [f1, f2], "kind": "pair_lookup", "score": round(acc, 4), "metric": "accuracy"}
            if best is not None:
                out["derived_columns"].append(best)

        # an "equivalent source" that is itself derived FROM this column is circular, not equivalent
        for item in out["derived_columns"]:
            item["alternatives"] = [a for a in item.get("alternatives", [])
                                    if not any(o2["column"] == a and item["column"] in o2["explained_by"] for o2 in out["derived_columns"])]

        # ── 3. target leakage ──
        if target and target in d.columns and target not in ids:
            feats = [c for c in cols if c != target]
            if _is_numeric(d[target]):
                y = d[target].astype(float)
                for f in feats:                                   # near copy of the target
                    if _is_numeric(d[f]):
                        r = d[f].corr(y, method="spearman")
                        if pd.notna(r) and abs(r) >= TARGET_COPY_RHO:
                            out["target_leakage"].append({"feature": f, "kind": "near_copy", "score": round(float(r), 4), "metric": "spearman"})
                # target ~ ratio/product of two positive columns:  log y = a*log x1 + b*log x2 + c
                pos = [f for f in feats if _is_numeric(d[f]) and (d[f] > 0).mean() > 0.98]
                if (y > 0).mean() > 0.98 and len(pos) >= 2:
                    ly = np.log(y.clip(lower=1e-9)).values
                    L = {f: np.log(d[f].clip(lower=1e-9)).values for f in pos}

                    def fit(cols_):
                        X = np.c_[tuple(L[c] for c in cols_) + (np.ones(len(ly)),)]
                        beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
                        res = ly - X @ beta
                        return 1 - (res ** 2).sum() / ((ly - ly.mean()) ** 2).sum(), beta
                    singles = {f: fit([f])[0] for f in pos}
                    lo, hi = COEF_RANGE
                    for f1, f2 in itertools.combinations(pos, 2):
                        pr, beta = fit([f1, f2])
                        both_matter = lo <= abs(beta[0]) <= hi and lo <= abs(beta[1]) <= hi
                        if pr >= LOG_R2 and pr - max(singles[f1], singles[f2]) >= 0.10 and both_matter:
                            out["target_leakage"].append({"feature": f1, "partner": f2, "kind": "log_ratio",
                                                          "score": round(float(pr), 4), "metric": "log_r2",
                                                          "single_r2": [round(float(singles[f1]), 3), round(float(singles[f2]), 3)],
                                                          "exponents": [round(float(beta[0]), 2), round(float(beta[1]), 2)]})
            else:
                dep = next((x for x in out["derived_columns"] if x["column"] == target), None)
                if dep:
                    out["target_leakage"].append({"feature": dep["explained_by"][0], "partner": (dep["explained_by"] + [None])[1],
                                                  "kind": "target_function", "score": dep["score"], "metric": "accuracy"})
            for nd in out["near_duplicates"]:                    # a feature that is a copy of a copy of the target
                if target in (nd["a"], nd["b"]) and nd["kind"] == "monotonic_copy":
                    other = nd["b"] if nd["a"] == target else nd["a"]
                    if not any(t["feature"] == other for t in out["target_leakage"]):
                        out["target_leakage"].append({"feature": other, "kind": "near_copy", "score": nd["spearman"], "metric": "spearman"})

        out["summary"] = _summaries(out, target)
    except Exception as e:   # diagnostics must never break the pipeline
        out["summary"] = [f"Relationship check skipped: {type(e).__name__}"]
    return out


def _summaries(r: dict, target: str | None) -> list[str]:
    s = []
    for t in r["target_leakage"]:
        if t["kind"] == "log_ratio":
            s.append(f"Target '{target}' is almost a formula of '{t['feature']}' and '{t['partner']}' "
                     f"(log-scale R²={t['score']:.2f}; either alone only {max(t['single_r2']):.2f}). "
                     f"A model using both will look far better than it can really predict.")
        elif t["kind"] == "near_copy":
            s.append(f"Feature '{t['feature']}' is a near-copy of the target '{target}' ({t['metric']}={t['score']:.3f}).")
        else:
            s.append(f"Target '{target}' can be computed from '{t['feature']}'"
                     + (f" and '{t['partner']}'" if t.get("partner") else "") + f" ({t['score']:.0%} accuracy): the model would just rediscover that rule.")
    for d in r["derived_columns"]:
        by = " + ".join(f"'{c}'" for c in d["explained_by"])
        if d.get("alternatives"):
            by += " (equivalently " + ", ".join(f"'{c}'" for c in d["alternatives"]) + ")"
        verb = {"binned_version": "is a binned version of", "lookup": "is a lookup of", "pair_lookup": "is a lookup of"}[d["kind"]]
        s.append(f"'{d['column']}' {verb} {by} ({d['score']:.0%} reproducible); it adds no independent information.")
    for n in r["near_duplicates"]:
        word = "a monotonic copy of" if n["kind"] == "monotonic_copy" else "nearly a duplicate of"
        s.append(f"'{n['a']}' is {word} '{n['b']}' (Spearman {n['spearman']:.3f}); their importances cannot be told apart.")
    return s
