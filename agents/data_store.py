"""
Per-job dataset store
─────────────────────
Keeps the ORIGINAL uploaded DataFrame reachable for the whole life of a job.

Why this exists
  * run_pipeline() only kept the *cleaned* DataFrame (missing values already
    imputed), so chat answered "0 missing values".
  * `jobs` lives in process memory. Cloud Run can route a later request to a
    different instance (or restart the container), so anything held only in a
    Python dict disappears.

Design
  * The raw upload is already written to UPLOAD_FOLDER. We register it with a
    small JSON sidecar (`<job_id>.meta.json`) holding the owner and file paths.
  * `get_dataframe()` validates ownership from the sidecar, loads the file with
    pandas, and caches a small number of frames in an LRU (bounded memory).
  * Nothing is fabricated: if the file or metadata is missing the caller gets
    (None, reason) and must return a clear error.

Cloud Run note
  The container filesystem is per-instance and ephemeral. For multi-instance
  deployments point UPLOAD_FOLDER at a shared volume (GCS FUSE / Filestore) or
  run with --max-instances=1. Without that, a request served by another
  instance will correctly report "dataset unavailable" instead of guessing.
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict

import pandas as pd

_CACHE_MAX = int(os.environ.get("DATASTORE_CACHE_ITEMS", 3))
_lock = threading.Lock()
_cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_JOB_RE = re.compile(r"^job_[0-9a-f]{32}$")


def _folder() -> str:
    from flask import current_app
    return current_app.config["UPLOAD_FOLDER"]


def _meta_path(folder: str, job_id: str) -> str:
    if not _JOB_RE.match(job_id or ""):
        raise ValueError("invalid job id")
    return os.path.join(folder, f"{job_id}.meta.json")


def register(folder: str, job_id: str, user: str, raw_path: str, filename: str,
             cleaned_path: str | None = None, extra: dict | None = None) -> None:
    meta = {"job_id": job_id, "user": user, "raw_path": raw_path, "filename": filename,
            "cleaned_path": cleaned_path}
    if extra:
        meta.update(extra)
    tmp = _meta_path(folder, job_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(tmp, _meta_path(folder, job_id))


def update(folder: str, job_id: str, **fields) -> None:
    meta = load_meta(folder, job_id)
    if meta is None:
        return
    meta.update(fields)
    tmp = _meta_path(folder, job_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(tmp, _meta_path(folder, job_id))


def load_meta(folder: str, job_id: str) -> dict | None:
    try:
        with open(_meta_path(folder, job_id), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def owner_allowed(meta: dict, user_email: str) -> bool:
    return meta.get("user") in ("guest@datamind.ai", user_email)


def get_dataframe(folder: str, job_id: str, user_email: str, kind: str = "raw"):
    """Return (DataFrame | None, error_code | None). kind: 'raw' or 'cleaned'."""
    try:
        meta = load_meta(folder, job_id)
    except ValueError:
        return None, "invalid_job"
    if meta is None:
        return None, "dataset_unavailable"
    if not owner_allowed(meta, user_email):
        return None, "forbidden"
    path = meta.get("raw_path") if kind == "raw" else (meta.get("cleaned_path") or meta.get("raw_path"))
    if not path or not os.path.exists(path):
        return None, "dataset_unavailable"
    try:
        key = (job_id, kind, os.path.getmtime(path))
    except OSError:
        return None, "dataset_unavailable"
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key], None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None, "dataset_unreadable"
    with _lock:
        _cache[key] = df
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return df, None


def clear_cache() -> None:
    with _lock:
        _cache.clear()


ERROR_MESSAGES = {
    "dataset_unavailable": ("The original dataset for this analysis is not available on this server instance, "
                            "so I cannot calculate statistics from it. Please re-upload the file."),
    "dataset_unreadable": "The original dataset could not be read from storage.",
    "forbidden": "You do not have access to this dataset.",
    "invalid_job": "Invalid job id.",
}
