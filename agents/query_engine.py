"""
Agent: Safe Data-Query Engine
─────────────────────────────
Answers natural-language questions about the ORIGINAL uploaded DataFrame with
deterministic pandas operations. No LLM-generated code, no eval()/exec().

Pipeline:
  1. Resolve column names mentioned in the question against the real schema.
  2. Parse filters (explicit "col op value", ranges, missing checks) and
     implicit filters ("female passengers", "survivors", "diabetic patients")
     by matching words against real column names / category values.
  3. Validate columns, operators, dtypes and categories.
  4. Pick an allowlisted operation and run it with pandas.
  5. Return a structured result (operation, filters, values, limitations).

If the question cannot be interpreted safely the engine returns
status="unsupported" so the caller can fall back to RAG/LLM chat.
If it is ambiguous it returns status="clarify" with a question to ask.
Nothing is ever fabricated: unknown columns/categories produce an explicit
explanation instead of a number.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

import numpy as np
import pandas as pd

MAX_ROWS_OUT = 25          # bounded table size returned to client / LLM
MAX_NUMERIC_COLS = 12      # bounded number of columns in wide summaries
MAX_VALUE_CARD = 200       # max unique values for implicit category matching
MAX_GROUPS = 30

# Words that carry operation meaning and must never be treated as category values
_KEYWORDS = {
    "average", "mean", "median", "min", "minimum", "max", "maximum", "std", "sum", "total",
    "count", "number", "percentage", "percent", "compare", "comparison", "versus", "vs",
    "analyze", "analyse", "show", "calculate", "what", "which", "how", "many", "much",
    "each", "every", "only", "where", "with", "without", "among", "records", "rows",
    "columns", "column", "missing", "null", "unique", "distinct", "duplicate", "duplicates",
    "the", "and", "for", "are", "there", "present", "all", "any", "top", "most", "common",
    "values", "value", "data", "dataset", "table", "group", "grouped", "per", "between",
    "than", "from", "that", "this", "have", "has", "had", "does", "did", "not", "non",
    "list", "give", "tell", "get", "find", "display", "describe", "summary", "statistics",
    "distribution", "crosstab", "cross", "tabulation", "highest", "lowest", "greater",
    "less", "more", "equals", "equal", "above", "below", "over", "under", "least", "most",
}
_NEGATORS = {"not", "non", "didn't", "didnt", "never", "without", "no", "isn't", "wasn't", "weren't"}

_TRUE_WORDS = {"1", "true", "yes", "y", "t", "positive"}
_FALSE_WORDS = {"0", "false", "no", "n", "f", "negative"}


# ═════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════
def _norm(text: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("non-", "non "))


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _prefix_match(a: str, b: str) -> bool:
    """Loose stem match: 'survivors' ~ 'survived', 'diabetic' ~ 'diabetes'."""
    a, b = _singular(a.lower()), _singular(b.lower())
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 6 or (n >= 4 and n >= min(len(a), len(b)) - 1 and min(len(a), len(b)) >= 5)


def _fmt(v: Any, nd: int = 4) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), nd)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v if isinstance(v, (int, bool, str)) else str(v)


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _binary_positive(s: pd.Series):
    """Return (positive_value, negative_value) for a binary column, else None."""
    vals = s.dropna().unique().tolist()
    if len(vals) != 2:
        return None
    lowered = {str(v).strip().lower(): v for v in vals}
    pos = [lowered[k] for k in lowered if k in _TRUE_WORDS and k not in {"no", "n", "f", "false", "0", "negative"}]
    neg = [lowered[k] for k in lowered if k in _FALSE_WORDS]
    if len(pos) == 1 and len(neg) == 1:
        return pos[0], neg[0]
    return None


class QueryError(Exception):
    pass


class Clarify(Exception):
    pass


# ═════════════════════════════════════════════════════════════════
# schema + parsing
# ═════════════════════════════════════════════════════════════════
class _Schema:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = [str(c) for c in df.columns]
        self.norm = {c: _norm(c) for c in self.columns}
        self.cat_values: dict[str, dict[str, Any]] = {}   # col -> {lower value -> original}
        for c in self.columns:
            s = df[c]
            if _is_num(s) and s.nunique(dropna=True) > 2:
                continue
            if pd.api.types.is_datetime64_any_dtype(s):
                continue
            try:
                n = s.nunique(dropna=True)
            except TypeError:
                continue
            if 0 < n <= MAX_VALUE_CARD:
                self.cat_values[c] = {str(v).strip().lower(): v for v in s.dropna().unique()}

    # -- column mention resolution ---------------------------------
    def find_columns(self, message: str) -> list[tuple[str, int, int]]:
        """Return [(column, start, end)] for columns mentioned in message (non-overlapping)."""
        text = _norm(message)
        hits: list[tuple[str, int, int]] = []
        used = [False] * (len(text) + 1)
        for col in sorted(self.columns, key=lambda c: -len(self.norm[c])):
            n = self.norm[col]
            if not n:
                continue
            variants = {n, n + "s", n + "es"}
            if n.endswith("y"):
                variants.add(n[:-1] + "ies")
            for v in variants:
                for m in re.finditer(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", text):
                    if any(used[m.start():m.end()]):
                        continue
                    for i in range(m.start(), m.end()):
                        used[i] = True
                    hits.append((col, m.start(), m.end()))
        # typo tolerance on single tokens (e.g. "pasenger")
        for m in re.finditer(r"[a-z0-9]{5,}", text):
            if any(used[m.start():m.end()]) or m.group() in _KEYWORDS:
                continue
            for col in self.columns:
                n = self.norm[col]
                if " " in n or len(n) < 5:
                    continue
                if difflib.SequenceMatcher(None, m.group(), n).ratio() >= 0.88 and \
                        m.group() not in {_singular(n), n}:
                    hits.append((col, m.start(), m.end()))
                    for i in range(m.start(), m.end()):
                        used[i] = True
                    break
        hits.sort(key=lambda h: h[1])
        return hits


_OP_PATTERNS = [
    (r">=|at least|no less than|greater than or equal to|not less than", ">="),
    (r"<=|at most|no more than|less than or equal to|not more than", "<="),
    (r"!=|<>|not equal to|is not|isn't|are not|does not equal|other than|not equals?", "!="),
    (r">|greater than|more than|above|over|older than|exceeds?|higher than", ">"),
    (r"<|less than|fewer than|below|under|younger than|lower than", "<"),
    (r"==|=|equals?|equal to|is|are|was|were|:", "=="),
]
_OP_RE = "|".join(f"(?P<op{i}>{p})" for i, (p, _) in enumerate(_OP_PATTERNS))
_COND_AFTER = re.compile(
    r"^\s*(?:" + _OP_RE + r")\s*(?P<val>\"[^\"]+\"|'[^']+'|-?\d+(?:\.\d+)?|[A-Za-z0-9_\-\.]+(?:\s+[A-Za-z0-9_\-\.]+){0,3})",
    re.IGNORECASE,
)
_MISSING_AFTER = re.compile(r"^\s*(?:is|are|was|were)?\s*(?:missing|null|nan|empty|blank|na)\b", re.IGNORECASE)
_BETWEEN = re.compile(r"^\s*(?:is|are)?\s*between\s+(-?\d+(?:\.\d+)?)\s+(?:and|to|-)\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


class _Cond(dict):
    """{'column','op','value','source','negated','pos'} – op in ==,!=,>,>=,<,<=,between,isna,notna"""


def _coerce_value(series: pd.Series, raw: str, col: str, schema: _Schema):
    """Validate and coerce the textual value to the column dtype. Raises QueryError."""
    raw = raw.strip().strip("\"'").strip()
    s = series
    if pd.api.types.is_bool_dtype(s):
        low = raw.lower()
        if low in _TRUE_WORDS - {"positive"}:
            return True
        if low in _FALSE_WORDS - {"negative"}:
            return False
        raise QueryError(f"'{raw}' is not a valid Boolean value for column '{col}' (use true/false).")
    if _is_num(s):
        try:
            f = float(raw)
        except ValueError:
            raise QueryError(f"Column '{col}' is numeric but '{raw}' is not a number.")
        return int(f) if float(f).is_integer() and pd.api.types.is_integer_dtype(s) else f
    # categorical / text
    lookup = schema.cat_values.get(col)
    if lookup is None:
        uniq = {str(v).strip().lower(): v for v in s.dropna().unique()[:50000]}
    else:
        uniq = lookup
    key = raw.lower()
    if key in uniq:
        return uniq[key]
    sing = _singular(key)
    for k, v in uniq.items():
        if _singular(k) == sing:
            return v
    close = difflib.get_close_matches(key, list(uniq.keys()), n=5, cutoff=0.6)
    sample = [str(uniq[c]) for c in close] or [str(v) for v in list(uniq.values())[:8]]
    raise QueryError(
        f"The category '{raw}' does not exist in column '{col}'. "
        f"{'Closest existing values' if close else 'Example values'}: {', '.join(sample)}."
    )


def _apply_cond(df: pd.DataFrame, c: _Cond) -> pd.Series:
    s = df[c["column"]]
    op, v = c["op"], c.get("value")
    if op == "isna":
        return s.isna()
    if op == "notna":
        return s.notna()
    if op == "between":
        lo, hi = v
        return s.between(lo, hi)
    if op in {">", ">=", "<", "<="}:
        if not _is_num(s) and not pd.api.types.is_datetime64_any_dtype(s):
            raise QueryError(f"Cannot use '{op}' on non-numeric column '{c['column']}'.")
        return {">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
    if op == "==":
        return s == v
    if op == "!=":
        return (s != v) & s.notna()
    raise QueryError(f"Unsupported operator '{op}'.")


def _vtxt(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _cond_text(c: _Cond) -> str:
    op = {"==": "=", "isna": "is missing", "notna": "is not missing"}.get(c["op"], c["op"])
    if c["op"] == "between":
        return f"{c['column']} between {_vtxt(c['value'][0])} and {_vtxt(c['value'][1])}"
    if c["op"] in {"isna", "notna"}:
        return f"{c['column']} {op}"
    v = _vtxt(c["value"])
    return f"{c['column']} {op} {v!r}" if isinstance(v, str) else f"{c['column']} {op} {v}"


class _Parser:
    def __init__(self, df: pd.DataFrame, message: str):
        self.df = df
        self.schema = _Schema(df)
        self.message = message
        self.low = message.lower().replace("non-", "non ")
        self.cols = self.schema.find_columns(message)
        self.conds: list[_Cond] = []
        self.consumed_cols: set[str] = set()      # columns used in a filter
        self.consumed_spans: list[tuple[int, int]] = []
        self.interpretations: list[str] = []

    # -- explicit "col op value" -----------------------------------
    def parse_explicit(self):
        raw_low = self.message
        # map normalized-text spans back is brittle; instead search each column phrase in raw text
        for col in dict.fromkeys(c for c, _, _ in self.cols):
            pat = re.compile(r"(?<![A-Za-z0-9])" + r"[\s_\-]*".join(map(re.escape, self.schema.norm[col].split())) + r"s?(?![A-Za-z0-9])", re.IGNORECASE)
            for m in pat.finditer(raw_low):
                tail = raw_low[m.end():]
                mm = _MISSING_AFTER.match(tail)
                if mm:
                    self.conds.append(_Cond(column=col, op="isna", value=None, source="explicit", negated=False, pos=m.start()))
                    self.consumed_cols.add(col); self.consumed_spans.append((m.start(), m.end() + mm.end()))
                    continue
                mb = _BETWEEN.match(tail)
                if mb:
                    lo, hi = float(mb.group(1)), float(mb.group(2))
                    if not _is_num(self.df[col]):
                        raise QueryError(f"Cannot apply a numeric range to non-numeric column '{col}'.")
                    self.conds.append(_Cond(column=col, op="between", value=(min(lo, hi), max(lo, hi)), source="explicit", negated=False, pos=m.start()))
                    self.consumed_cols.add(col); self.consumed_spans.append((m.start(), m.end() + mb.end()))
                    continue
                ma = _COND_AFTER.match(tail)
                if not ma:
                    continue
                op = None
                for i, (_, sym) in enumerate(_OP_PATTERNS):
                    if ma.group(f"op{i}"):
                        op = sym
                        break
                if op is None:
                    continue
                val = ma.group("val").strip()
                # "is" followed by aggregation/stop words is not a filter ("what is the average ...")
                first = _tokens(val)[0] if _tokens(val) else ""
                if first in _KEYWORDS and not val.startswith(("'", '"')):
                    continue
                # trim multi-word values to the longest known category prefix
                val = self._trim_value(col, val)
                if op in {"==", "!="} and val.lower() in {"missing", "null", "nan", "empty", "blank"}:
                    self.conds.append(_Cond(column=col, op="isna" if op == "==" else "notna", value=None, source="explicit", negated=False, pos=m.start()))
                else:
                    coerced = _coerce_value(self.df[col], val, col, self.schema)
                    self.conds.append(_Cond(column=col, op=op, value=coerced, source="explicit", negated=False, pos=m.start()))
                self.consumed_cols.add(col)
                self.consumed_spans.append((m.start(), m.end() + ma.end()))

    def _trim_value(self, col: str, val: str) -> str:
        if val[:1] in "\"'":
            return val
        lookup = self.schema.cat_values.get(col)
        words = val.split()
        if lookup:
            for n in range(len(words), 0, -1):
                cand = " ".join(words[:n]).lower()
                if cand in lookup or _singular(cand) in {_singular(k) for k in lookup}:
                    return " ".join(words[:n])
        if lookup:
            return words[0]
        # numeric or unknown: first token only (stop at joiners)
        out = []
        for w in words:
            if w.lower() in {"and", "or", "but", "for", "with", "of", "in", "by"}:
                break
            out.append(w)
            if len(out) == 1 and not lookup:
                break
        return " ".join(out) if out else words[0]

    # -- implicit filters: category words and binary labels -------
    def parse_implicit(self):
        toks = [(m.group(), m.start()) for m in re.finditer(r"[a-z0-9']+", self.low)]
        used_tok: set[int] = set()
        # in-span tokens (already consumed by explicit filters / column mentions) are skipped
        spans = list(self.consumed_spans)
        for _, s, e in self.cols:
            pass  # column spans are in normalized-text coordinates, handled by name below
        col_words = {w for c, _, _ in self.cols for w in self.schema.norm[c].split()}
        col_words |= {_singular(w) for w in col_words}

        def in_span(pos: int) -> bool:
            return any(a <= pos < b for a, b in spans)

        # multi-word category values first (longest n-gram)
        for n in (3, 2):
            for i in range(len(toks) - n + 1):
                if any(j in used_tok for j in range(i, i + n)):
                    continue
                phrase = " ".join(t for t, _ in toks[i:i + n])
                hits = [(c, lk[phrase]) for c, lk in self.schema.cat_values.items() if phrase in lk and not _is_num(self.df[c])]
                if hits:
                    self._add_value_hits(hits, phrase, toks[i][1], i, n, used_tok)
        for i, (tok, pos) in enumerate(toks):
            if i in used_tok or in_span(pos) or tok in _KEYWORDS or len(tok) < 3 or tok.isdigit():
                continue
            if tok in col_words:
                continue
            neg = any(toks[j][0] in _NEGATORS for j in range(max(0, i - 2), i))
            # 1) exact / singular category value match
            hits = []
            for c, lk in self.schema.cat_values.items():
                if _is_num(self.df[c]) or c in self.consumed_cols:
                    continue
                if tok in lk or _singular(tok) in lk:
                    hits.append((c, lk.get(tok, lk.get(_singular(tok)))))
            if not hits:
                # 2) stem match against category values ("diabetic" ~ "Diabetes")
                for c, lk in self.schema.cat_values.items():
                    if _is_num(self.df[c]) or c in self.consumed_cols or len(lk) > 60:
                        continue
                    for k, orig in lk.items():
                        if len(k) >= 5 and " " not in k and _prefix_match(tok, k):
                            hits.append((c, orig))
                            break
            label_hits = []
            for c in self.schema.columns:
                if c in self.consumed_cols or c in {h[0] for h in hits}:
                    continue
                words = self.schema.norm[c].split()
                if len(words) == 1 and len(words[0]) >= 5 and _prefix_match(tok, words[0]) and tok != words[0]:
                    bp = _binary_positive(self.df[c])
                    if bp:
                        label_hits.append((c, bp))
            if hits and label_hits:
                opts = [f"{c} = {v!r}" for c, v in hits] + [f"{c} (positive class)" for c, _ in label_hits]
                raise Clarify(f"'{tok}' could refer to more than one thing: {', '.join(opts)}. Which do you mean?")
            if len(hits) > 1 and len({h[0] for h in hits}) > 1:
                opts = [f"{c} = {v!r}" for c, v in hits]
                raise Clarify(f"'{tok}' matches several columns: {', '.join(opts)}. Which one do you mean?")
            if hits:
                self._add_value_hits(hits, tok, pos, i, 1, used_tok, neg=neg)
            elif len(label_hits) == 1:
                c, (pv, nv) = label_hits[0]
                v = nv if neg else pv
                self.conds.append(_Cond(column=c, op="==", value=v, source="label", negated=neg, pos=pos))
                self.interpretations.append(f"'{'non-' if neg else ''}{tok}' interpreted as {c} = {v!r}")
                used_tok.add(i)
            elif len(label_hits) > 1:
                raise Clarify(f"'{tok}' could refer to columns: {', '.join(c for c, _ in label_hits)}. Which one do you mean?")

    def raw_pos(self, col: str) -> int:
        words = self.schema.norm[col].split()
        m = re.search(r"[\s_\-]*".join(map(re.escape, words)), self.low) if words else None
        return m.start() if m else 0

    def add_binary_outcomes(self):
        """'how many passengers survived' -> the binary column named in the question is the outcome (positive class)."""
        toks = [(m.group(), m.start()) for m in re.finditer(r"[a-z0-9']+", self.low)]
        for col in dict.fromkeys(c for c, _, _ in self.cols):
            if col in self.consumed_cols or any(c["column"] == col for c in self.conds):
                continue
            bp = _binary_positive(self.df[col])
            if not bp:
                continue
            pos = self.raw_pos(col)
            idx = next((i for i, (_, tp) in enumerate(toks) if tp >= pos), len(toks))
            neg = any(toks[j][0] in _NEGATORS for j in range(max(0, idx - 2), idx))
            v = bp[1] if neg else bp[0]
            self.conds.append(_Cond(column=col, op="==", value=v, source="label", negated=neg, pos=pos))
            self.interpretations.append(f"'{col}' interpreted as {col} = {v!r} ({'negative' if neg else 'positive'} class)")

    def _add_value_hits(self, hits, phrase, pos, i, n, used_tok, neg=False):
        cols = {c for c, _ in hits}
        if len(cols) > 1:
            raise Clarify(f"'{phrase}' matches several columns: {', '.join(f'{c} = {v!r}' for c, v in hits)}. Which one do you mean?")
        c, v = hits[0]
        self.conds.append(_Cond(column=c, op="!=" if neg else "==", value=v, source="value", negated=neg, pos=pos))
        self.interpretations.append(f"'{phrase}' interpreted as {c} {'!=' if neg else '='} {v!r}")
        for j in range(i, i + n):
            used_tok.add(j)


# ═════════════════════════════════════════════════════════════════
# operations
# ═════════════════════════════════════════════════════════════════
def _mask(df: pd.DataFrame, conds: list[_Cond]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in conds:
        m &= _apply_cond(df, c)
    return m


def _numeric_cols(df: pd.DataFrame, exclude=()) -> list[str]:
    out = []
    for c in df.columns:
        if c in exclude or not _is_num(df[c]):
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        out.append(str(c))
    return out


def _numeric_summary(df: pd.DataFrame, cols: list[str], stats=("mean", "median", "min", "max", "std")) -> list[dict]:
    rows = []
    for c in cols[:MAX_NUMERIC_COLS]:
        s = df[c]
        r = {"column": c, "non_null": int(s.notna().sum()), "missing": int(s.isna().sum())}
        for st in stats:
            r[st] = _fmt(getattr(s, st)()) if s.notna().any() else None
        rows.append(r)
    return rows


def _md_table(rows: list[dict], cols: list[str] | None = None, limit: int = MAX_ROWS_OUT) -> str:
    if not rows:
        return "_(no rows)_"
    cols = cols or list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows[:limit]:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    if len(rows) > limit:
        lines.append(f"_(showing {limit} of {len(rows)} rows)_")
    return "\n".join(lines)


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 2) if d else None


def _result(status, operation, text="", filters=None, values=None, limitations=None, **extra):
    r = {
        "handled": status != "unsupported",
        "status": status,
        "operation": operation,
        "filters": [_cond_text(c) if isinstance(c, dict) else c for c in (filters or [])],
        "values": values if values is not None else {},
        "limitations": limitations or [],
        "text": text,
    }
    r.update(extra)
    return r


# -- dataset overview --------------------------------------------
def op_shape(df, p, **_):
    v = {"rows": int(len(df)), "columns": int(df.shape[1])}
    return _result("ok", "shape", f"The dataset has **{v['rows']:,} rows** and **{v['columns']} columns**.", values=v)


def op_columns(df, p, dtypes=False, **_):
    rows = [{"column": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
    text = f"The dataset has {len(rows)} columns:\n\n" + _md_table(rows, limit=100)
    return _result("ok", "dtypes" if dtypes else "column_names", text, values={"columns": rows})


def op_missing(df, p, **_):
    target_cols = list(dict.fromkeys(c for c, _, _ in p.cols)) or [str(c) for c in df.columns]
    n = len(df)
    rows = [{"column": c, "missing": int(df[c].isna().sum()), "missing_pct": _pct(int(df[c].isna().sum()), n)} for c in target_cols]
    shown = [r for r in rows if r["missing"] > 0]
    text = ("Missing values (computed on the original uploaded data, before any cleaning):\n\n"
            + (_md_table(shown or rows, limit=100)))
    if not shown:
        text += "\n\nNo missing values found in the selected column(s)."
    return _result("ok", "missing_values", text,
                   values={"per_column": rows, "total_missing": int(sum(r["missing"] for r in rows)), "rows": n})


def op_duplicates(df, p, **_):
    d = int(df.duplicated().sum())
    return _result("ok", "duplicate_rows", f"There are **{d:,}** duplicate rows ({_pct(d, len(df))}% of {len(df):,}).",
                   values={"duplicate_rows": d, "rows": int(len(df))})


def op_describe(df, p, filters_text=None, **_):
    cols = [c for c, _, _ in p.cols if _is_num(df[c])] or _numeric_cols(df)
    summary = _numeric_summary(df, cols, stats=("mean", "median", "min", "max", "std"))
    lim = []
    if len(cols) > MAX_NUMERIC_COLS:
        lim.append(f"Showing the first {MAX_NUMERIC_COLS} of {len(cols)} numeric columns.")
    text = f"Descriptive statistics (n = {len(df):,} rows; missing values excluded per column):\n\n" + _md_table(summary)
    return _result("ok", "describe", text, values={"numeric_summary": summary, "rows": int(len(df))}, limitations=lim)


def op_value_counts(df, p, col, top=None, **_):
    s = df[col]
    vc = s.value_counts(dropna=False)
    total = int(len(s))
    rows = [{"value": "(missing)" if (isinstance(k, float) and np.isnan(k)) or k is None else _fmt(k), "count": int(v), "percent": _pct(int(v), total)}
            for k, v in vc.items()]
    lim = []
    if top:
        rows = rows[:top]
    elif len(rows) > MAX_ROWS_OUT:
        lim.append(f"{len(rows)} distinct values; showing the top {MAX_ROWS_OUT}.")
    text = f"Value counts for **{col}** (n = {total:,}, {int(s.nunique())} distinct non-missing values):\n\n" + _md_table(rows)
    return _result("ok", "value_counts", text, values={"column": col, "counts": rows[:MAX_ROWS_OUT], "rows": total,
                                                       "distinct": int(s.nunique())}, limitations=lim)


def op_unique(df, p, col, **_):
    n = int(df[col].nunique(dropna=True))
    vals = [_fmt(v) for v in df[col].dropna().unique()[:20]]
    return _result("ok", "unique_values", f"**{col}** has **{n:,}** distinct non-missing values"
                   + (f": {', '.join(map(str, vals))}" if n <= 20 else f" (first 20: {', '.join(map(str, vals))})") + ".",
                   values={"column": col, "distinct": n, "sample": vals})


# -- filtering / aggregation -------------------------------------
def _filtered(df, p):
    m = _mask(df, p.conds)
    sub = df[m]
    return sub, int(m.sum())


def _no_match(df, p, op):
    return _result("ok", op, "No rows match the requested filter(s), so there is nothing to calculate.",
                   filters=p.conds, values={"matching_rows": 0, "total_rows": int(len(df))},
                   limitations=["Filter matched zero rows."])


def op_count(df, p, **_):
    sub, n = _filtered(df, p)
    total = len(df)
    text = f"**{n:,}** rows match ({_pct(n, total)}% of {total:,} total rows)."
    if p.conds:
        text = f"**{n:,}** of {total:,} rows match {' AND '.join(_cond_text(c) for c in p.conds)} ({_pct(n, total)}%)."
    return _result("ok", "count", text, filters=p.conds, values={"count": n, "total_rows": int(total), "percent": _pct(n, total)})


def op_analyze_subset(df, p, **_):
    sub, n = _filtered(df, p)
    if n == 0:
        return _no_match(df, p, "analyze_subset")
    excl = {c["column"] for c in p.conds if c["op"] == "=="}
    num = _numeric_summary(sub, _numeric_cols(sub, exclude=excl))
    cats = []
    for c in df.columns:
        if c in excl or _is_num(df[c]) or df[c].nunique(dropna=True) > 20 or not df[c].nunique(dropna=True):
            continue
        vc = sub[c].value_counts().head(5)
        cats.append({"column": str(c), "top_values": {str(k): int(v) for k, v in vc.items()}, "missing": int(sub[c].isna().sum())})
    cats = cats[:10]
    miss = {str(c): int(sub[c].isna().sum()) for c in df.columns if sub[c].isna().any()}
    text = (f"Analysis of the filtered subset ({' AND '.join(_cond_text(c) for c in p.conds)}): "
            f"**{n:,}** of {len(df):,} rows ({_pct(n, len(df))}%).\n\n**Numeric columns**\n\n" + _md_table(num))
    if cats:
        text += "\n\n**Top categories**\n\n" + "\n".join(f"- {c['column']}: " + ", ".join(f"{k} ({v})" for k, v in c['top_values'].items()) for c in cats)
    if miss:
        text += "\n\n**Missing values in subset**: " + ", ".join(f"{k}: {v}" for k, v in miss.items())
    return _result("ok", "analyze_subset", text, filters=p.conds,
                   values={"rows": n, "total_rows": int(len(df)), "numeric_summary": num, "categorical_summary": cats, "missing_in_subset": miss})


_AGG = {
    "mean": ("average", "mean", "avg"), "median": ("median",), "min": ("minimum", "min", "lowest", "smallest"),
    "max": ("maximum", "max", "highest", "largest"), "std": ("std", "stdev", "deviation", "standard"),
    "sum": ("sum", "total"),
}


def _agg_functions(low: str) -> list[str]:
    toks = set(_tokens(low))
    found = [k for k, words in _AGG.items() if any(w in toks for w in words)]
    return found


def op_aggregate(df, p, funcs, cols, group_cols=None, **_):
    sub, n = _filtered(df, p)
    if n == 0:
        return _no_match(df, p, "aggregate")
    lim = []
    if group_cols:
        g = group_cols[0]
        if sub[g].nunique(dropna=True) > MAX_GROUPS:
            lim.append(f"{sub[g].nunique()} groups; showing the first {MAX_GROUPS}.")
        rows = []
        for key, part in list(sub.groupby(g, dropna=False))[:MAX_GROUPS]:
            r = {g: "(missing)" if pd.isna(key) else _fmt(key), "count": int(len(part))}
            for c in cols:
                for f in funcs:
                    r[f"{f}({c})"] = _fmt(getattr(part[c], f)()) if part[c].notna().any() else None
            rows.append(r)
        text = f"{', '.join(funcs)} of {', '.join(cols)} grouped by **{g}**" + (f" (filter: {' AND '.join(_cond_text(c) for c in p.conds)})" if p.conds else "") + ":\n\n" + _md_table(rows)
        return _result("ok", "group_aggregate", text, filters=p.conds, values={"group_by": g, "groups": rows, "rows": n}, limitations=lim)
    out, lines = [], []
    for c in cols:
        s = sub[c]
        nn = int(s.notna().sum())
        entry = {"column": c, "non_null": nn, "missing": int(s.isna().sum())}
        for f in funcs:
            entry[f] = _fmt(getattr(s, f)()) if nn else None
        out.append(entry)
        parts = ", ".join(f"{f} = {entry[f]}" for f in funcs)
        note = f" (based on {nn:,} non-missing of {n:,} rows)" if nn != n else f" (n = {n:,})"
        lines.append(f"- **{c}**: {parts}{note}")
        if nn != n:
            lim.append(f"{c}: {n - nn:,} missing values were excluded from the calculation.")
    head = f"Computed on {n:,} rows" + (f" where {' AND '.join(_cond_text(c) for c in p.conds)}" if p.conds else " (all rows)") + ":\n\n"
    return _result("ok", "aggregate", head + "\n".join(lines), filters=p.conds, values={"rows": n, "results": out}, limitations=lim)


def _group_summary(df, g_conds: list[_Cond], label_col: str, exclude: set[str]):
    num = _numeric_cols(df, exclude=exclude)[:MAX_NUMERIC_COLS]
    rows = []
    for gc in g_conds:
        part = df[_apply_cond(df, gc)]
        r = {"group": _cond_text(gc), "count": int(len(part)), "share_%": _pct(len(part), len(df))}
        for c in num:
            r[f"mean {c}"] = _fmt(part[c].mean()) if part[c].notna().any() else None
        r["_n_missing"] = {c: int(part[c].isna().sum()) for c in num if part[c].isna().any()}
        rows.append(r)
    return rows, num


def op_compare(df, p, groups: list[_Cond], **_):
    base_conds = [c for c in p.conds if c not in groups]
    base = df[_mask(df, base_conds)] if base_conds else df
    gcol = groups[0]["column"]
    rows, num = _group_summary(base, groups, gcol, {gcol})
    if not any(r["count"] for r in rows):
        return _no_match(df, p, "compare")
    missing_notes = [f"{r['group']}: " + ", ".join(f"{k} missing {v}" for k, v in r["_n_missing"].items()) for r in rows if r["_n_missing"]]
    clean_rows = [{k: v for k, v in r.items() if k != "_n_missing"} for r in rows]
    text = f"Comparison across **{gcol}** groups (n = {len(base):,} rows considered; means exclude missing values):\n\n" + _md_table(clean_rows)
    lim = [f"Missing values excluded from means — {m}" for m in missing_notes]
    return _result("ok", "compare_groups", text, filters=base_conds + groups,
                   values={"group_column": gcol, "groups": clean_rows, "rows_considered": int(len(base))}, limitations=lim)


def op_compare_all(df, p, gcol, **_):
    base = df[_mask(df, p.conds)] if p.conds else df
    if base[gcol].nunique(dropna=True) > MAX_GROUPS:
        raise QueryError(f"Column '{gcol}' has {base[gcol].nunique()} distinct values — too many to compare. Name specific categories.")
    groups = [_Cond(column=gcol, op="==", value=v, source="auto", negated=False, pos=0) for v in base[gcol].dropna().unique()]
    pp = type("P", (), {"conds": [c for c in p.conds]})()
    pp.conds = list(p.conds)
    return op_compare(df, _wrap(p, groups_conds=p.conds), groups=groups)


def _wrap(p, groups_conds):
    class _P:  # minimal stand-in carrying the base conditions
        conds = groups_conds
        cols = p.cols
    return _P()


def op_percentage(df, p, pop_conds: list[_Cond], out_cond: _Cond | None, **_):
    pop = df[_mask(df, pop_conds)] if pop_conds else df
    den = int(len(pop))
    if den == 0:
        return _no_match(df, p, "percentage")
    if out_cond is None:
        num = den
        text = f"**{num:,}** of {len(df):,} rows ({_pct(num, len(df))}%) match {' AND '.join(_cond_text(c) for c in pop_conds)}."
        return _result("ok", "percentage", text, filters=pop_conds, values={"numerator": num, "denominator": int(len(df)), "percent": _pct(num, len(df))})
    num = int(_apply_cond(pop, out_cond).sum())
    pop_desc = (" among rows where " + " AND ".join(_cond_text(c) for c in pop_conds)) if pop_conds else " (all rows)"
    text = (f"**{_pct(num, den)}%** — {num:,} of {den:,} rows{pop_desc} have {_cond_text(out_cond)}.")
    return _result("ok", "percentage", text, filters=pop_conds + [out_cond],
                   values={"numerator": num, "denominator": den, "percent": _pct(num, den), "population_filters": [_cond_text(c) for c in pop_conds],
                           "outcome": _cond_text(out_cond)})


def op_crosstab(df, p, a, b, **_):
    if df[a].nunique(dropna=True) > MAX_GROUPS or df[b].nunique(dropna=True) > MAX_GROUPS:
        raise QueryError("Too many distinct values for a readable cross-tabulation (limit 30 per column).")
    sub, n = _filtered(df, p)
    if n == 0:
        return _no_match(df, p, "crosstab")
    ct = pd.crosstab(sub[a], sub[b])
    rows = []
    for idx, r in ct.iterrows():
        row = {a: _fmt(idx)}
        row.update({str(k): int(v) for k, v in r.items()})
        row["total"] = int(r.sum())
        rows.append(row)
    text = f"Cross-tabulation of **{a}** by **{b}** (n = {int(ct.values.sum()):,} rows with both values present):\n\n" + _md_table(rows, limit=MAX_GROUPS)
    lim = []
    dropped = n - int(ct.values.sum())
    if dropped:
        lim.append(f"{dropped:,} rows were excluded because {a} or {b} is missing.")
    return _result("ok", "crosstab", text, filters=p.conds, values={"rows_a": a, "cols_b": b, "table": rows, "n": int(ct.values.sum())}, limitations=lim)


# ═════════════════════════════════════════════════════════════════
# router
# ═════════════════════════════════════════════════════════════════
def answer_question(df: pd.DataFrame | None, message: str) -> dict:
    """Public entry point. Never raises; returns a structured result dict."""
    if df is None:
        return _result("error", "unavailable", "The original dataset is not available for this job, so I cannot calculate this. "
                       "Please re-upload the file.", limitations=["dataset_unavailable"])
    try:
        return _route(df, message)
    except Clarify as e:
        return _result("clarify", "clarification", str(e))
    except QueryError as e:
        return _result("no_match", "validation_error", str(e), limitations=[str(e)])
    except Exception as e:  # defensive: never let the engine crash chat
        return _result("unsupported", "internal_error", "", limitations=[f"engine error: {type(e).__name__}"])


def _route(df: pd.DataFrame, message: str) -> dict:
    low = message.lower().replace("non-", "non ")
    toks = set(_tokens(low))
    p = _Parser(df, message)
    p.parse_explicit()
    p.parse_implicit()
    mentioned = list(dict.fromkeys(c for c, _, _ in p.cols))
    def has(*ws):
        return any((w in low) if (" " in w or w == "%") else (w in toks) for w in ws)
    funcs = _agg_functions(low)
    outcome_ok = not funcs and not re.search(r"\b(each|every|by|per|distribution|crosstab|unique|distinct|missing|null|duplicate|duplicates|describe)\b", low)
    if outcome_ok and has("how many", "number", "count", "percentage", "percent", "proportion", "%"):
        p.add_binary_outcomes()

    def interp(res):
        if p.interpretations and res.get("status") == "ok":
            res["interpretations"] = p.interpretations
            res["text"] += "\n\n_Interpretation: " + "; ".join(p.interpretations) + "._"
        return res

    # ── 1. compare ───────────────────────────────────────────────
    if has("compare", "comparison", "versus", "vs", "difference", "differ", "differences"):
        by_col: dict[str, list[_Cond]] = {}
        for c in p.conds:
            if c["op"] in {"==", "!="}:
                by_col.setdefault(c["column"], []).append(c)
        multi = [v for v in by_col.values() if len(v) >= 2]
        if multi:
            groups = multi[0]
            return interp(op_compare(df, p, groups=groups))
        if len(by_col) == 1:
            gcol = next(iter(by_col))
            col_conds = by_col[gcol]
            bp = _binary_positive(df[gcol])
            if bp and len(col_conds) == 1:
                other = bp[1] if col_conds[0]["value"] == bp[0] else bp[0]
                groups = [col_conds[0], _Cond(column=gcol, op="==", value=other, source="label", negated=False, pos=0)]
                p.conds = [c for c in p.conds if c is not col_conds[0]]
                return interp(op_compare(df, p, groups=groups))
            if len(col_conds) == 1 and df[gcol].nunique(dropna=True) > 1:
                raise Clarify(f"Which categories of '{gcol}' should I compare with {col_conds[0]['value']!r}? "
                              f"Available: {', '.join(map(str, list(df[gcol].dropna().unique()[:8])))}.")
        cand = [c for c in mentioned if c not in p.consumed_cols and (df[c].nunique(dropna=True) <= MAX_GROUPS) and not _is_num(df[c])]
        cand += [c for c in mentioned if c not in p.consumed_cols and _binary_positive(df[c]) and c not in cand]
        if cand:
            return interp(op_compare_all(df, p, cand[0]))
        raise Clarify("What should I compare? Name two categories (e.g. 'compare A and B') or a column to group by.")

    # ── 2. percentage ────────────────────────────────────────────
    if has("percentage", "percent", "proportion", "%"):
        conds = sorted(p.conds, key=lambda c: c["pos"])
        if not conds:
            raise Clarify("Percentage of what? Mention a category or a condition (e.g. 'percentage of female passengers who survived').")
        out = conds[-1] if len(conds) >= 2 else (conds[0] if conds[0]["source"] == "label" else None)
        pop = conds[:-1] if len(conds) >= 2 else ([] if out is not None else conds)
        return interp(op_percentage(df, p, pop_conds=pop, out_cond=out))

    # ── 3. overview ──────────────────────────────────────────────
    if has("missing", "null", "nan", "blank") and not any(c["op"] in {"isna", "notna"} for c in p.conds):
        return op_missing(df, p)
    if has("duplicate", "duplicates", "duplicated"):
        return op_duplicates(df, p)
    if re.search(r"\b(rows?|records?|observations?)\b.*\b(columns?|features?|variables?|fields?)\b|\b(columns?|features?)\b.*\b(rows?|records?)\b|\bshape\b|\bsize of (the )?data", low) \
            and not p.conds and not funcs:
        return op_shape(df, p)
    if re.search(r"\b(column names?|list (of )?(all )?columns|what columns|which columns|data ?types?|dtypes?)\b", low) and not p.conds:
        return op_columns(df, p, dtypes=bool(re.search(r"data ?types?|dtypes?", low)))
    if re.search(r"\b(describe|descriptive statistics|summary statistics|summary stats|statistical summary)\b", low) and not p.conds and not funcs:
        return op_describe(df, p)
    if re.search(r"\b(how many|number of|count of)\s+(rows|records)\b", low) and not p.conds:
        return op_shape(df, p)
    if re.search(r"\b(how many|number of)\s+columns\b", low) and not p.conds:
        return op_shape(df, p)

    # ── 4. crosstab ──────────────────────────────────────────────
    free = [c for c in mentioned if c not in p.consumed_cols]
    if re.search(r"cross ?tab|cross[- ]tabulation|contingency", low) or (has("by") and has("count") and len(free) == 2 and not funcs):
        if len(free) != 2:
            raise Clarify("Which two columns should I cross-tabulate?")
        return interp(op_crosstab(df, p, free[0], free[1]))

    # ── 5. value counts / unique ─────────────────────────────────
    top_n = re.search(r"\btop\s+(\d{1,2})\b", low)
    if re.search(r"\b(each|every)\b.*\b(count|number)\b|\bcount of each\b|\bvalue counts?\b|\bdistribution of\b|\bbreakdown of\b|\bfrequency of\b|\bmost common\b|\btop\s+\d*\s*(categories|values)\b|\bcategor(y|ies) counts?\b", low) \
            or (has("count", "show", "list") and re.search(r"\beach\b", low)):
        cand = [c for c in mentioned if not _is_num(df[c]) or df[c].nunique(dropna=True) <= 20] or mentioned
        if not cand:
            raise Clarify("Which column should I count values for? Available columns: " + ", ".join(map(str, df.columns[:15])) + ".")
        if len(cand) > 1:
            raise Clarify(f"Which column — {', '.join(cand)}?")
        return interp(op_value_counts(df, p, cand[0], top=int(top_n.group(1)) if top_n else None))
    if re.search(r"\b(unique|distinct)\b", low):
        if not mentioned:
            raise Clarify("Which column's unique values should I count?")
        return op_unique(df, p, mentioned[0])

    # ── 6. aggregation (mean/median/... of numeric cols) ─────────
    if funcs:
        num_cols = [c for c in mentioned if _is_num(df[c]) and c not in p.consumed_cols]
        group_cols = []
        mg = re.search(r"\b(?:by|per|grouped by|group by|for each|across)\s+([a-z0-9_ ]+)", low)
        if mg:
            gtxt = _norm(mg.group(1))
            group_cols = [c for c in mentioned if _norm(c) and _norm(c) in gtxt]
            num_cols = [c for c in num_cols if c not in group_cols]
        if not num_cols:
            raise Clarify("Which numeric column should I calculate that for? Numeric columns: " + ", ".join(_numeric_cols(df)[:15]) + ".")
        bad = [c for c in group_cols if df[c].nunique(dropna=True) > MAX_GROUPS]
        if bad:
            raise QueryError(f"Column '{bad[0]}' has too many distinct values to group by (limit {MAX_GROUPS}).")
        return interp(op_aggregate(df, p, funcs=funcs, cols=num_cols, group_cols=group_cols or None))

    # ── 7. filtered counts / subset analysis ─────────────────────
    if p.conds:
        if re.search(r"\b(analy[sz]e|summar(y|ize|ise)|profile|statistics|stats|only|show)\b", low) and not has("how many", "count", "number"):
            return interp(op_analyze_subset(df, p))
        if has("how many", "count", "number", "total"):
            return interp(op_count(df, p))
        return interp(op_analyze_subset(df, p))

    # ── 8. bare column questions ─────────────────────────────────
    if mentioned and re.search(r"\b(how many|number of)\b.*\b(unique|distinct|different)\b", low):
        return op_unique(df, p, mentioned[0])

    return _result("unsupported", "unsupported", "")


# ═════════════════════════════════════════════════════════════════
# LLM evidence packaging
# ═════════════════════════════════════════════════════════════════
def evidence_for_llm(result: dict, max_chars: int = 3500) -> str:
    """Compact, trusted evidence string. Numbers here are authoritative."""
    import json
    payload = {k: result.get(k) for k in ("operation", "filters", "values", "limitations", "interpretations") if result.get(k)}
    s = json.dumps(payload, default=str)
    return s if len(s) <= max_chars else s[:max_chars] + "...(truncated)"
