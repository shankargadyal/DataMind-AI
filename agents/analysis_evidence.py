"""Authoritative dataset and evaluation facts for analysis sessions."""
import hashlib
import re

import pandas as pd


BANK_COLUMNS = {"age", "job", "marital", "education", "default", "balance", "housing",
                "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                "previous", "poutcome", "y"}
BANK_TARGETS = {
    "y": "whether the customer subscribed to a term deposit",
    "default": "whether the customer has credit in default",
    "housing": "whether the customer has a housing loan",
    "loan": "whether the customer has a personal loan",
}
NUMBER_CLAIM = re.compile(
    r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"\w+teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|percent(?:age)?|half|quarter|third|double|twice|triple|dozens?|both|most|all|none)\b",
    re.I,
)
UNSUPPORTED_INTERPRETATION = re.compile(
    r"\b(?:caus\w+|guarantee\w*|proves?|will definitely|will always)\b", re.I
)


def qualitative_text_is_safe(text):
    return not (NUMBER_CLAIM.search(text) or UNSUPPORTED_INTERPRETATION.search(text))


def is_bank_marketing(columns):
    return BANK_COLUMNS.issubset(columns)


def file_identity(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_evidence(raw, det, ml, selected_target, identity, availability):
    target = selected_target or ml.get("target_column")
    bank = is_bank_marketing(raw.columns)
    distribution = {}
    if target in raw.columns and (ml.get("task_type") == "classification" or not pd.api.types.is_numeric_dtype(raw[target])):
        distribution = {str(value): int(count) for value, count in raw[target].value_counts().items()}
    return {
        "dataset": {"name": det["filename"], "sha256": identity, "rows": len(raw),
                    "columns": len(raw.columns), "missing_cells": int(raw.isna().sum().sum()),
                    "missing_by_column": {col: int(n) for col, n in raw.isna().sum().items()}},
        "target": {"name": target, "selection": "explicit" if selected_target else "automatic",
                   "description": BANK_TARGETS.get(target) if bank else None,
                   "distribution": distribution, "missing": int(raw[target].isna().sum()) if target else None},
        "bank_marketing": bank,
        "quality": det["quality_dimensions"],
        "model": {"name": ml.get("best_model"), "task": ml.get("task_type"),
                  "models": ml.get("models", []), "evaluation": ml.get("evaluation_protocol", {}),
                  "importance": ml.get("feature_importances", []),
                  "shap": {k: v for k, v in ml.get("shap_plots", {}).items()
                           if k not in ("summary_plot", "waterfall_plot")},
                  "availability": availability, "availability_comparison": ml.get("availability_comparison")},
    }


def resolve_question(question, evidence, previous_question=""):
    """Return computed facts and the intents they cover. Unresolved compound clauses are reported."""
    q = question.lower().strip()
    if re.fullmatch(r"(?:and\s+)?(?:what(?:'s| is)\s+)?(?:the\s+)?(?:percentage|rate|count|recall|precision|f1|accuracy)[?. ]*", q):
        q += " (context: " + previous_question.lower() + ")"
    dataset, target, model = (evidence.get(k, {}) for k in ("dataset", "target", "model"))
    quality = evidence.get("quality", {})
    parts = []
    intents = set()

    def add(intent, source, statement):
        intents.add(intent)
        line = f"{statement} [{source}]"
        if line not in parts:
            parts.append(line)

    def missing(intent, source, subject):
        add(intent, source, f"{subject} is unavailable for this analysis.")

    target_name = target.get("name")
    counts = target.get("distribution", {})
    bank = evidence.get("bank_marketing")
    rows = dataset.get("rows")
    if re.search(r"\b(?:prediction|model|selected|active|chosen|analysis) target\b|\btarget (?:is|column|variable)\b|\bwhat.*\btarget\b", q):
        if target_name:
            description = target.get("description")
            add("target", "target", f"The prediction target is `{target_name}`" + (f", indicating {description}." if description else "."))
        else:
            missing("target", "target", "A prediction target")
    if re.search(r"\b(?:rows|records|observations|samples|customers)\b", q) and not re.search(r"subscri|\bwhere\b|\bwith\b|\bmissing\b|\byes\b|\bpositive\b|\bduplicate|\btrain|hold.out|\bfor\b|\bwhy\b|\baverage\b", q):
        if rows is not None:
            add("rows", "dataset", f"The uploaded dataset contains {rows:,} records.")
    if re.search(r"\b(?:columns|variables)\b", q) and not re.search(r"\bmissing|top|important|feature|list|which|target|numeric|categorical|datetime|date", q):
        if dataset.get("columns") is not None:
            add("columns", "dataset", f"The uploaded dataset contains {dataset['columns']:,} columns.")
    if re.search(r"\bmissing\b|\bnulls?\b", q) and not re.search(r"by column|per column|which columns|most missing|where", q):
        by_col = dataset.get("missing_by_column", {})
        named = [col for col in by_col if re.search(rf"(?<!\w){re.escape(col.lower())}(?!\w)", q)]
        if named and ("column" in q or len(named) == 1):
            for col in named:
                add("missing", "dataset", f"`{col}` has {by_col[col]:,} missing values in the uploaded data.")
        elif dataset.get("missing_cells") is not None:
            add("missing", "dataset", f"The uploaded dataset contains {dataset['missing_cells']:,} missing cells.")
    if re.search(r"\bquality\b|\bcompleteness\b|\bvalidity\b|\bconsistency\b|\buniqueness\b", q):
        if quality.get("composite_score") is not None and "quality" in q:
            add("quality", "quality", f"Composite data-quality score: {quality['composite_score']}/100.")
        for name, value in quality.get("dimensions", {}).items():
            if re.search(rf"\b{name}\b", q) and (name != "accuracy" or "quality" in q or "data accuracy" in q):
                add("quality", "quality", f"Data {name}: {value}%.")
    models = model.get("models", [])
    matched_models = [m for m in models if re.search(rf"(?<!\w){re.escape(m['name'].lower())}(?!\w)", q)]
    if matched_models:
        matched_models = [m for m in matched_models if not any(
            m["name"].lower() in other["name"].lower() and m is not other for other in matched_models
        )]
    chosen = matched_models or [m for m in models if m.get("is_best")]
    if re.search(r"\b(?:best|winning|selected|top) model\b|\bwhich model\b", q):
        if model.get("name"):
            add("model", "models", f"The selected model is {model['name']}.")
        else:
            missing("model", "models", "A selected model")
    metric_patterns = {
        "accuracy": r"\baccuracy\b", "macro_f1": r"\bmacro[- ](?:average[d]?[- ])?f1\b",
        "weighted_f1": r"\bweighted[- ](?:average[d]?[- ])?f1\b", "f1": r"\bf1\b",
        "recall": r"\brecall\b|\bsensitivity\b", "precision": r"\bprecision\b",
        "r2": r"\br[²2]\b", "mae": r"\bmae\b|\bmean absolute error\b",
        "rmse": r"\brmse\b|\broot mean squared error\b",
    }
    requested = [key for key, pattern in metric_patterns.items() if re.search(pattern, q)]
    if "macro_f1" in requested or "weighted_f1" in requested:
        requested.remove("f1")
    if "accuracy" in requested and "data accuracy" in q:
        requested.remove("accuracy")
        if "accuracy" in quality.get("dimensions", {}):
            add("quality", "quality", f"Data accuracy: {quality['dimensions']['accuracy']}%.")
    refers_positive = bool(re.search(r"subscri|\bpositive\b|\by\s*=\s*yes\b|\bclass\s+yes\b", q))
    matrix_requested = bool(re.search(r"confusion matrix|\b(?:true|false) (?:positives?|negatives?)\b|\b(?:tp|fp|tn|fn)\b", q))
    baseline_requested = "baseline" in q and not re.search(r"shap|waterfall", q)
    if requested or matrix_requested or (baseline_requested and not re.search(r"dataset|overall class", q)):
        if not chosen:
            missing("metrics", "models", "The requested model evaluation")
        for item in chosen:
            metrics = item.get("metrics", {})
            source = f"models/{item['name']}"
            if refers_positive and bank and target_name != "y" and "subscri" in q:
                missing("metrics", source, "A subscriber-class evaluation for the active target")
                continue
            for key in requested:
                if key in {"f1", "recall", "precision"} and refers_positive:
                    key = "positive_" + key
                value = metrics.get(key)
                label = {"f1": "weighted F1", "precision": "weighted precision", "recall": "weighted recall",
                         "macro_f1": "macro F1", "weighted_f1": "weighted F1", "positive_f1": "positive-class F1",
                         "positive_recall": "positive-class recall", "positive_precision": "positive-class precision",
                         "r2": "R²"}.get(key, key.replace("_", "-"))
                if value is None:
                    missing("metrics", source, f"{item['name']} {label}")
                else:
                    formatted = f"{value:.4g}" if key in {"r2", "mae", "rmse"} else f"{value:.1%}"
                    add("metrics", source, f"{item['name']} hold-out {label}: {formatted}.")
            if matrix_requested:
                if metrics.get("confusion_matrix"):
                    detail = f"Rows are actual and columns predicted, in label order {metrics.get('confusion_labels')}: {metrics['confusion_matrix']}."
                    if metrics.get("binary_confusion"):
                        detail += " " + ", ".join(f"{k.upper()}={v:,}" for k, v in metrics["binary_confusion"].items()) + "."
                    add("metrics", source, detail)
                else:
                    missing("metrics", source, "A labeled confusion matrix")
            if baseline_requested and not re.search(r"dataset|overall class", q):
                value = metrics.get("majority_baseline_accuracy")
                if value is None:
                    missing("metrics", source, "The hold-out majority-class baseline")
                else:
                    message = f"The hold-out majority-class baseline is {value:.2%}."
                    if metrics.get("accuracy") is not None:
                        message += f" The model accuracy difference is {(metrics['accuracy'] - value) * 100:+.2f} percentage points."
                    add("metrics", source, message)
    dataset_baseline = baseline_requested and re.search(r"dataset|overall class", q)
    if dataset_baseline and counts and rows:
        add("distribution", "target/distribution", f"The dataset majority class is {max(counts.values()) / rows:.2%} of all records.")
    qualified_dataset_query = bool(re.search(r"\b(?:among|where|for the group|with)\b", q))
    distribution_requested = not qualified_dataset_query and re.search(
        r"target (?:class )?distribution|class (?:counts?|distribution|balance)|label distribution|"
        r"subscription rate|how many.*(?:subscri|\bpositive\b|\byes\b|\bno\b)|"
        r"(?:percent(?:age)?|proportion|share|count|rate).*?(?:subscri|\bpositive\b|\byes\b|\by\s*=)|"
        r"(?:\by\s*=\s*(?:yes|no)\b)|non[- ]?subscribers?|subscription percentage", q
    )
    if distribution_requested and (not requested or re.search(r"how many|subscription rate|class distribution|percentage of", q)):
        negative = bool(re.search(r"non[- ]?subscri|did not subscribe|\by\s*=\s*no\b|\bnegative\b", q))
        positive = refers_positive or "subscribed" in q or "subscription" in q
        percentage = bool(re.search(r"percent|proportion|rate|share", q))
        explicit_bank = next((col for col in BANK_TARGETS if re.search(rf"`{col}`|\b{col}\s*=", q)), None)
        if (explicit_bank and bank and explicit_bank != target_name) or (bank and "subscri" in q and target_name != "y"):
            add("distribution", "target", f"The active target is `{target_name}`. This analysis has no class distribution for the requested target.")
        elif counts and rows:
            pos = model.get("evaluation", {}).get("positive_class")
            if pos is None and set(counts).issubset({"yes", "no"}):
                pos = "yes"
            selected = [k for k in counts if (k != pos if negative else k == pos)] if (positive or negative) and pos is not None else list(counts)
            if re.search(r"\b(?:and|versus|vs\.?|compared)\s+(?:the\s+)?(?:non[- ]?subscri|no\b|negative\b)", q):
                selected = list(counts)
            selected = selected or list(counts)
            detail = ", ".join(f"`{k}`: {counts[k]:,}" + (f" ({counts[k] / rows:.2%} of all records)" if percentage else "") for k in selected)
            add("distribution", "target/distribution", f"For `{target_name}`, {detail}.")
        else:
            missing("distribution", "target/distribution", "The requested class distribution")
    if re.search(r"\bshap\b|\b(?:top|important|importance) features?\b|\bfeature importance\b", q):
        shap = model.get("shap", {})
        local = bool(re.search(r"\blocal\b|waterfall|single prediction", q))
        ranking = shap.get("top_features") if "shap" in q or local else model.get("importance")
        if local and shap.get("available"):
            add("shap", "models/shap", shap["waterfall_caption"])
        elif ranking:
            scope = ""
            if "shap" in q and shap.get("explained_class") is not None:
                scope = f" for class `{shap['explained_class']}` on the {shap['output_scale']} scale"
            add("shap", "models/shap" if "shap" in q else "models/importance",
                f"Leading features for `{target_name}`{scope}: " + ", ".join(f["feature"] for f in ranking[:5]) +
                ". These describe model behavior and statistical associations.")
        else:
            missing("shap", "models/shap", "The requested feature explanation")
    if re.search(r"\bduration\b|feature availability|target leakage|before the call", q) and model.get("availability"):
        for col, entry in model["availability"].items():
            add("availability", "models/availability", f"`{col}`: {entry['reason']}")
        comparison = model.get("availability_comparison")
        if comparison and comparison.get("best_model") and re.search(r"compar|without|with and", q):
            add("availability", "models/availability", f"The restricted comparison selected {comparison['best_model']}. Metrics are reported in the analysis.")
    return "\n".join(parts), intents
