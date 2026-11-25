# Generalization evaluation, local candidate

Continued from 7a062d3. Existing research artifacts and transcripts were retained. Release labels remain as previously published at the owner's request.

## Evidence handling

The evaluation resolver reads the selected model's labeled hold-out matrix. It uses actual classes for rows, predictions for columns, the diagonal for correct predictions and off-diagonal cells for errors. In the supplied development matrix, actual yes/predicted no gives 62; the diagonal is 766 + 42 = 808 of 905. The opposite error direction yields 35. These are distinct from whole-file counts and from the independently fitted paired experiment.

Other development checks cover majority baseline, minority recall, macro F1 versus support-weighted F1, feature timing, the evaluator's stored baseline verdict, missing salary, distribution shift, the limits of data quality and causal interpretation. Pattern-based language routing still has coverage gaps. Unavailable historical verdicts and metrics must be reported as unavailable. Hold-out metrics and class names are read from analysis evidence.

## Tests and PDF

The complete backend suite passed 39 tests in 26.65 seconds. All four frontend tests passed with Node's in-process runner. An initial backend run failed one existing revenue wording assertion, which was corrected before the final full run. PDF checks extracted identical ROC-AUC and average precision values from the chat and generated report fixture and verified selection and CV results. New binary classifiers with supported probability scores record both ranking metrics. Existing SHAP graph, Guardrails and fallback sections were retained. Poppler rendered all three pages of a representative report using persisted Bank evidence. No overlapping headers were seen. The historical metrics fixture lacks probability metrics and SHAP image payloads; visual inspection verifies table layout, not a fresh complete graph-rendering integration. Local tests cover Prediction Studio, review state and PDF fallback. Live Cloud Run and live LLM calls were not tested.

## Fresh question set

The ten prompts in evaluation/final_holdout.json were evaluated once after the resolver was frozen. Each prompt used the authenticated Flask chat endpoint without an LLM key or prior chat turns, with persisted Bank Marketing evidence. The exact replies and predefined required/excluded fragments are in evaluation/final_holdout_results.json. Eight met their rubric. One actual-yes/labeled-no paraphrase returned whole-dataset class counts; one clean-data/reliable-predictions paraphrase returned a generic fallback. These are unresolved limitations. The test uses a small author-created set, one dataset and substring grading. It is independent of tuning this candidate, not an externally blinded or population-level estimate. The supplied 3.5/10 used different prompts and scoring and does not establish a directly comparable baseline.

## Controlled experiment

Executed python -m experiments.duration BANK_CSV --output experiments/final_results. The dataset hash, index hashes and every previously recorded arm metric matched the prior experiment. Both arms used 3,616 training and 905 held-out rows, seed 42, identical XGBoost settings, and preprocessing fit independently on training data. Only duration differed. With duration, ROC-AUC was 0.9132 and average precision was 0.5119. Without duration, these were 0.7153 and 0.3047. Duration is observed after the call. The restricted model addresses a different prediction-time assumption. Results apply to this split. The original experiments/results directory is preserved.

Install by overlaying the ZIP on the reviewed main checkout. To reproduce verification, run python -m pytest -q and node --test --test-isolation=none templates/tests/analysisSession.test.js from the repository root. No deployment or remote updates were performed.
