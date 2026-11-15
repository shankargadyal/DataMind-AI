"""Evidence-limited, dataset-agnostic recommendations for a trained prediction.

This module does not invoke an LLM or claim causal effects. Recommendations are
clearly marked as suggestions, not empirical conclusions.
"""
import math


def build_prediction_advice(*, prediction, task_type, target, inputs,
                            class_probabilities=None, feature_importances=None,
                            evaluation_warnings=None, best_score=None,
                            score_metric=None):
    target_name = str(target or "the selected target").replace("_", " ")
    classification = task_type == "classification"
    probability = None
    if classification and isinstance(class_probabilities, dict):
        probability = class_probabilities.get(str(prediction))
    if classification:
        summary = f"The trained model predicts {target_name}: {prediction}."
        if isinstance(probability, (int, float)) and math.isfinite(probability):
            summary += f" The model assigned this class {probability:.1f}% probability; this is not a guarantee."
    else:
        summary = f"The trained model estimates {target_name} at {prediction}. This is an estimate, not a guaranteed outcome."

    ranked = []
    for item in feature_importances or []:
        if not isinstance(item, dict):
            continue
        name = item.get("feature") or item.get("name")
        weight = item.get("importance")
        if name in inputs and isinstance(weight, (int, float)) and math.isfinite(weight):
            ranked.append((name, abs(weight)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_features = [name for name, _ in ranked[:3]]
    recommendations = [
        {"title": "Compare with similar records", "description":
         "Check how this result compares with records that have similar input values and verify the comparison is representative.",
         "basis": "Suggested validation step"},
        {"title": "Explore alternative scenarios", "description":
         "Use what-if analysis on relevant, realistically changeable inputs. A change in prediction does not establish a causal effect.",
         "basis": "Model sensitivity exploration"},
    ]
    if top_features:
        recommendations.insert(0, {"title": "Review influential model features",
            "description": "Inspect the dataset and model explanation for " + ", ".join(top_features) + ". Global importance does not explain this individual prediction.",
            "basis": "Global feature importance"})
    else:
        recommendations.insert(0, {"title": "Verify input quality",
            "description": "Check the entered values against the source dataset, including missing values, unusual ranges and category definitions.",
            "basis": "Input quality check"})
    # Domain-aware suggestions are framed as review prompts, never findings.
    # They use the uploaded target name, not any fixed dataset schema.
    target_key = str(target or "").lower().replace("_", " ")
    domain_prompts = [
        (("churn", "attrition", "turnover"), "Review retention signals",
         "Compare this prediction with historical retention patterns and investigate available engagement or satisfaction measures before deciding on outreach."),
        (("spend", "sales", "revenue", "purchase"), "Review commercial drivers",
         "Compare the estimate with recent sales or spending distributions and inspect relevant demand, seasonality and customer-segment variables."),
        (("grade", "score", "performance", "exam"), "Review learning support",
         "Compare this estimate with historical performance and investigate whether additional learning support could be appropriate."),
        (("default", "credit", "loan"), "Use qualified financial review",
         "Treat this model output as a screening signal only; review data quality, fairness and applicable lending policy before any decision."),
        (("disease", "diagnosis", "health", "risk"), "Seek qualified domain review",
         "Do not use a model prediction as a diagnosis or treatment decision; ask a qualified professional to review the relevant evidence."),
    ]
    for keywords, title, description in domain_prompts:
        if any(word in target_key for word in keywords):
            recommendations[1] = {"title": title, "description": description,
                                  "basis": "Target-domain review prompt, not a causal model finding"}
            break
    warnings = [str(w) for w in (evaluation_warnings or []) if w][:3]
    if isinstance(best_score, (int, float)) and math.isfinite(best_score) and score_metric:
        warnings.append(f"Reported evaluation metric: {score_metric} = {best_score:.3f}. Interpret using its scale and validation protocol.")
    return {"summary": summary, "recommendations": recommendations[:3],
            "warnings": warnings, "feature_evidence": top_features,
            "method": "Evidence-limited guidance; not causal or personalized professional advice"}
