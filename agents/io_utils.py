"""
Single place that reads an uploaded table, so the pipeline (detective) and the chat/query engine
(data_store) always see the SAME dataframe.

Handles: .xlsx/.xls (picks the sheet that actually holds the data), .csv/.tsv/.txt with delimiter
sniffing (, ; tab |), encoding fallback (utf-8-sig, utf-8, utf-16 with BOM, cp1252, latin-1), title rows
above the header, and text columns that are really numbers ("$1,234.56", "12.5%", "1234,56").
Anything it changes is recorded in df.attrs["io_notes"] (a list of plain-English strings).
"""
from __future__ import annotations
import csv
import logging
import re
import pandas as pd

logger = logging.getLogger("datamind")

TABLE_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_DELIMS = (",", ";", "\t", "|")
_CURRENCY = "$€£₹¥"


def detect_delimiter(sample: str) -> str:
    lines = [l for l in sample.splitlines() if l.strip()][:30]
    if not lines:
        return ","
    best, best_key = ",", (0.0, 0)
    for d in _DELIMS:
        try:
            widths = [len(next(csv.reader([l], delimiter=d))) for l in lines]
        except csv.Error:
            continue
        if widths[0] <= 1:
            continue
        consistency = sum(1 for w in widths if w == widths[0]) / len(widths)
        key = (round(consistency, 3), widths[0])
        if key > best_key:
            best, best_key = d, key
    return best


# ── numeric-looking text columns ────────────────────────────────────────────
_US = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$|^-?\d+(\.\d+)?$")          # 1,234.56 / 1234.56 / 1234
_EU = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$|^-?\d+,\d+$")               # 1.234,56 / 1234,56


def _strip_decor(v: str) -> tuple[str, bool]:
    s = v.strip()
    pct = s.endswith("%")
    s = s.rstrip("%").strip().lstrip(_CURRENCY).strip().replace("\u00a0", "").replace(" ", "")
    return s, pct


def _coerce_numeric_text(df: pd.DataFrame, decimal_comma_hint: bool) -> pd.DataFrame:
    notes = list(df.attrs.get("io_notes", []))
    for col in df.columns:
        ser = df[col]
        if ser.dtype != object:
            continue
        vals = ser.dropna().astype(str)
        vals = vals[vals.str.strip() != ""]
        if len(vals) < 5:
            continue
        cleaned = [_strip_decor(v) for v in vals]
        core = [c for c, _ in cleaned]
        # leading zeros ("00123", zip codes, ids) are labels, not numbers
        if any(re.match(r"^-?0\d", c) for c in core):
            continue
        us = [bool(_US.match(c)) for c in core]
        eu = [bool(_EU.match(c)) for c in core]
        # a column of plain integers needs no work (pandas already parsed those)
        decorated = any(c != v.strip() for c, v in zip(core, vals))
        has_comma_decimal = any(re.match(r"^-?\d+,\d{1,2}$", c) or re.match(r"^-?\d+,\d{4,}$", c) for c in core)
        use_eu = (sum(eu) / len(core) >= 0.9) and (has_comma_decimal or decimal_comma_hint) and sum(us) / len(core) < 0.9
        use_us = sum(us) / len(core) >= 0.9
        if not (use_eu or use_us):
            continue
        mixed_junk = use_us and sum(us) < len(core)       # e.g. 385 numbers + 15 x "unknown"
        if not (decorated or use_eu or mixed_junk or any("," in c for c in core)):
            continue

        def conv(v):
            if pd.isna(v) or str(v).strip() == "":
                return float("nan")
            c, _ = _strip_decor(str(v))
            if use_eu:
                c = c.replace(".", "").replace(",", ".")
            else:
                c = c.replace(",", "")
            try:
                return float(c)
            except ValueError:
                return float("nan")
        new = ser.map(conv)
        blank = ser.isna() | (ser.astype(str).str.strip() == "")
        lost = int((new.isna() & ~blank).sum())          # real text that could not be read as a number
        df[col] = new
        kind = ("decimal-comma" if use_eu else "percent" if any(p for _, p in cleaned)
                else "currency/thousands-separator" if decorated or any("," in c for c in core) else "numbers mixed with text")
        notes.append(f"Column '{col}' held {kind} text and was converted to numbers"
                     + (f" ({lost} unparseable value(s) became missing)" if lost > 0 else "."))
        logger.info("io_utils: converted '%s' (%s)", col, kind)
    if notes:
        df.attrs["io_notes"] = notes
    return df


# ── file readers ────────────────────────────────────────────────────────────
def _read_excel(path: str) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    if not sheets:
        raise ValueError("The workbook has no sheets.")
    name, df = max(sheets.items(), key=lambda kv: kv[1].shape[0] * kv[1].shape[1])
    if len(sheets) > 1:
        df.attrs["io_notes"] = [f"Workbook has {len(sheets)} sheets; analysed '{name}' (the largest). "
                                f"Other sheets: {', '.join(k for k in sheets if k != name)}."]
    return df


def _header_row(path: str, enc: str, sep: str) -> int:
    """Index of the first line whose field count equals the file's dominant field count (skips title rows)."""
    with open(path, "r", encoding=enc, errors="replace") as fh:
        lines = [l for l in (fh.readline() for _ in range(80)) if l != ""]
    widths = []
    for i, l in enumerate(lines):
        if l.strip():
            try:
                widths.append((i, len(next(csv.reader([l], delimiter=sep)))))
            except csv.Error:
                pass
    if not widths:
        return 0
    from collections import Counter
    mode = Counter(w for _, w in widths if w > 1).most_common(1)
    if not mode:
        return 0
    return next((i for i, w in widths if w == mode[0][0]), 0)


def _bom_encoding(head: bytes) -> str | None:
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    return None


def read_table(path: str) -> pd.DataFrame:
    if str(path).lower().endswith((".xlsx", ".xls")):
        return _coerce_numeric_text(_read_excel(path), False)
    with open(path, "rb") as fh:
        head = fh.read(8192)
    bom = _bom_encoding(head)
    if bom is None and b"\x00" in head:
        # latin-1 decodes ANY bytes, so without this a PNG/zip/exe would be "read" as garbage text
        raise ValueError("The file looks binary, not a text table (CSV/TSV) or Excel workbook.")
    encodings = (bom,) if bom else _ENCODINGS
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as fh:
                sample = fh.read(65536)
            sep = detect_delimiter(sample)
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            except pd.errors.ParserError as pe:
                skip = _header_row(path, enc, sep)
                if skip <= 0:
                    raise pe
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, skiprows=skip)
                df.attrs["io_notes"] = [f"Skipped {skip} title line(s) above the header row."]
            return _coerce_numeric_text(df, decimal_comma_hint=(sep == ";"))
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err or ValueError("could not decode file")
