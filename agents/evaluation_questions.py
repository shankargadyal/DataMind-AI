"""Interpret evaluation questions against the active model and its labeled hold-out."""
import re


def _confusion_answer(q, metrics, positive):
    labels = metrics.get("confusion_labels") or []
    matrix = metrics.get("confusion_matrix") or []
    if not labels or len(labels) != len(matrix) or any(len(row) != len(labels) for row in matrix):
        return "A complete labeled hold-out confusion matrix is unavailable."
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(labels)))
    wrong = bool(re.search(r"incorrect|misclass|wrong|missed|mistak|false|mislabel", q))
    if len(labels) == 2 and positive in labels:
        other = next(label for label in labels if label != positive)
        positive_term = rf"(?:positive|subscrib(?:er|ed)?s?|{re.escape(str(positive).lower())})"
        negative_term = rf"(?:negative|non[- ]?subscribers?|did not subscribe|not (?:to )?subscrib(?:e|ing|ed)|{re.escape(str(other).lower())})"
        actual, predicted = None, None
        for label, term in ((positive, positive_term), (other, negative_term)):
            if re.search(rf"(?:actual(?:ly)?|true class|in reality|really|were in fact)\s+(?:were |are )?{term}\b", q):
                actual = label
            if re.search(rf"(?:predict(?:ed)?|classif(?:y|ied)|label(?:ed|led)?)\s+(?:(?:them|as|to be|as a)\s+)*{term}\b", q):
                predicted = label
        if re.search(r"false negative|missed (?:positive|subscriber)", q):
            actual, predicted = positive, other
        elif re.search(r"false positive", q):
            actual, predicted = other, positive
        elif wrong and actual is None and predicted is None and re.search(r"non[- ]?subscriber|negative case", q):
            actual = other
        elif wrong and actual is None and predicted is None and re.search(r"subscriber|positive case", q):
            actual = positive
        if actual is not None or predicted is not None:
            if actual is None:
                actual = (other if predicted == positive else positive) if wrong else predicted
            if predicted is None:
                predicted = (other if actual == positive else positive) if wrong else actual
            count = matrix[labels.index(actual)][labels.index(predicted)]
            return f"Actual {actual}, predicted {predicted}: {count:,} hold-out cases (rows are actual classes; columns are predictions)."
    if re.search(r"subscri|positive|negative|actual|predicted as", q) and wrong:
        return "The requested class direction is unclear. Specify the actual class and the predicted class."
    if wrong:
        return f"{total - correct:,} of {total:,} hold-out predictions fall outside the diagonal and were incorrect."
    diagonal = " + ".join(str(matrix[i][i]) for i in range(len(labels)))
    return f"The diagonal sums to {diagonal} = {correct:,} correct predictions out of {total:,} hold-out cases."


def interpret(q, evidence):
    model = evidence.get("model", {})
    winner = next((m for m in model.get("models", []) if m.get("is_best") or m.get("name") == model.get("name")), {})
    named = [item for item in model.get("models", []) if item.get("name", "").lower() in q and item.get("name")]
    if len(named) == 1:
        winner = named[0]
    metrics = winner.get("metrics", {})
    source = f"[models/{winner['name']}]" if winner else "[models]"
    if re.search(r"correct|misclass|wrong|missed|mistak|mislabel|false (?:positive|negative)|actual.*predict", q) and re.search(r"how many|number of|count|total|how often", q) and re.search(r"predict|classif|hold.out|subscri|false |missed", q):
        if "subscri" in q and evidence.get("bank_marketing") and evidence.get("target", {}).get("name") != "y":
            return "Subscriber prediction errors are unavailable for the active target. [target]"
        positive = metrics.get("positive_class", model.get("evaluation", {}).get("positive_class"))
        return _confusion_answer(q, metrics, positive) + " " + source
    if re.search(r"future|next campaign|new customers|unseen customers", q) and re.search(r"guarantee|same|equivalent|perform|accuracy|certain|expect", q):
        return "Hold-out evaluation estimates performance on the sampled population. Changes in customers, collection or behavior can change future performance. Validate on a later representative cohort. [evaluation guidance]"
    if "quality" in q and re.search(r"predictive|model accuracy|reliab|accurate predict", q):
        return "Data quality measures properties such as completeness and consistency. Predictive reliability requires separate evaluation on representative held-out outcomes. A high quality score alone cannot guarantee accuracy. [quality]"
    if re.search(r"caus|cause.and.effect|lead to|effect of changing", q):
        return "Predictive associations and SHAP attributions describe model behavior. Confounding and selection can produce associations. Establishing a causal effect requires separate causal evidence, such as a randomized intervention. [interpretation guidance]"
    if "accuracy" in q and re.search(r"mislead|imbalan|decept|why.*high|mask|hide", q):
        pieces = ["Accuracy weights each case equally, so the majority class can dominate it."]
        for key, label in (("accuracy", "Hold-out accuracy"), ("majority_baseline_accuracy", "Majority baseline"), ("positive_recall", "Positive-class recall"), ("macro_f1", "Macro F1")):
            value = metrics.get(key)
            pieces.append(f"{label}: {value:.1%}." if value is not None else f"{label} unavailable.")
        pieces.append("Recall measures the fraction of actual positives detected. Macro F1 gives each class equal weight.")
        return " ".join(pieces) + " " + source
    return ""
