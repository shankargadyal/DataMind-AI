# DataMind chat evidence and controlled duration evaluation

## Review basis

The combined chat patch reverse-applied cleanly to the starting workspace. GitHub compared e8f00df against main and reported one additional merge commit and zero changed files. Local origin/main is 36386401c7825ac831bee151dc3ba8eab1d81b9f. Development continued from the greeting and evidence changes. The available specifications were the v3.2 attachment, its supplied continuation and the ten current v3.3 instructions. A separate v3.3 master document was unavailable, so coverage of any additional requirements remains open.

## Observed questions and repairs

| Case | Cause and implemented behavior |
|---|---|
| Greeting | Standalone salutations previously entered retrieval. The authenticated route now handles them before dataset lookup. |
| Subscribers and non-subscribers | The supplied live answer was correct. Regression coverage checks both counts through one compound API question. |
| Selection criterion | The saved evaluation protocol contained a rule the resolver did not expose. Selection queries now read that run's rule. |
| SHAP ranking and values | Formatting discarded stored mean absolute SHAP magnitudes. Chat reads ranking entries, includes available values and reports unavailable magnitudes. |
| SHAP causation | A ranking previously displaced the interpretation question. The causal interpretation describes attribution limits. |
| Guardrails flags | The evaluator result existed outside the chat evidence. New runs persist the final merged evaluation. In-memory older runs can reuse their evaluation. Older persisted sessions without it report unavailable. |
| Production readiness | Matching the model name hid the readiness question. Chat explains validation limits and includes the recorded feature availability constraints. |
| Approval and reliability | The resolver now distinguishes a workflow decision from evidence of generalization. |
| Individual probability | Returning aggregate accuracy did not answer the interpretation question. Chat explains its denominator and the role of probability calibration. |
| ROC-AUC | New binary evaluations with identifiable positive class and probability output calculate hold-out ROC-AUC. Missing historical metrics remain unavailable. |
| Revenue | A reference to campaign matched a numeric contact-count column. A shared financial-measure check examines schema names before aggregation and requests clarification for missing measures. |

The resolver recognizes intent with patterns and uses reusable explanations for statistical definitions. Counts, flags, scores, model identity, timing constraints and feature magnitudes come from the active analysis. The supported vocabulary is finite. Fields with unconventional monetary names require clarification. The general deployment response cannot certify a model. Some compound or novel questions may still need clarification. The paraphrase set was authored for this change and is a regression check, not an independent blind benchmark.

## Controlled XGBoost experiment

Run from the repository root with `python -m experiments.duration path/to/bank.csv`. The input was the original semicolon-delimited 4,521-row file. The experiment uses all non-target columns in the first arm and drops only duration in the second. Both arms clone the application XGBoost classifier, use one thread and 100 estimators, fit preprocessing exclusively on the same 3,616 training rows and score the same 905 held-out rows. Seed 42, stratification by y, and an 80/20 split were fixed before fitting. The CSV and JSON record weighted metrics, positive-class metrics, baselines and labeled confusion matrices. The JSON additionally records dataset and split hashes, library versions and configured parameters.

| Hold-out measure | With duration | Without duration |
|---|---:|---:|
| Accuracy | 89.06% | 88.07% |
| Macro F1 | 0.6961 | 0.5977 |
| Subscriber recall | 39.42% | 18.27% |
| ROC-AUC | 0.9132 | 0.7153 |
| Actual subscribers correctly identified | 41 of 104 | 19 of 104 |

The majority baseline was 88.51% accuracy. The restricted arm fell below it on accuracy and exceeded its macro F1 of 0.4695. Duration is known after a call, so the full arm uses information unavailable for pre-call selection. Results describe one fixed split. No independent external validation, multiple-seed uncertainty analysis, causal effect or production certification is implied. Historical 89.3% application accuracy belongs to a separate fit. These results should be cited as a separate experiment.

## Verification and limits

Python full-suite command: `python -m pytest -q`. Frontend full-suite command: `node --test --test-isolation=none templates/tests/analysisSession.test.js`. The default Node child-process runner encountered Windows spawn EPERM. The in-process runner executed the same four tests successfully. Run with the repository's runtime and development dependencies. Local Python used the existing dependency directory, with OMP_NUM_THREADS and OPENBLAS_NUM_THREADS set to 1.

The suite exercises classification and regression, preprocessing within folds, target persistence, Guardrails baselines, approval and rejection conflicts, guest isolation, PDF generation and extraction of quality metrics, Prediction Studio readiness, predictions, what-if scenarios, dataset switching, SHAP alignment, missing-key behavior, app assets and the container port contract. Integration fixtures use fast estimators. The separate experiment uses real XGBoost. No production deployment, managed Cloud Run smoke test, live LLM service test, browser visual PDF review or independent calibration study was performed.

## Files in the update

- app.py handles greetings, evidence context and final evaluator results.
- agents/analysis_evidence.py resolves facts and interpretation intents.
- agents/query_engine.py guards missing financial measures on the raw-data path.
- agents/ml_engineer.py computes supported binary hold-out ROC-AUC.
- tests/test_reliability.py extends the existing regression suite.
- evaluation/unseen_questions.json contains eleven separate paraphrased prompts and expected answer fragments.
- experiments/duration.py implements the paired experiment.
- experiments/results/duration.csv and experiments/results/duration.json contain the measured results.
- DATAMIND_VALIDATION.md is this report.

To install from the archive, extract over a checkout of the reviewed main revision and run both suites before release. For rollback after a reviewed merge, revert the merge in Git and restore the previous Cloud Run revision under your normal release process. Archive outputs are local research results. No source dataset, user session, database, credentials or front-end assets are distributed with this update.

Final results: 27 Python tests passed in 27.23 seconds, with zero failures or skips. All four frontend tests passed in 55.56 milliseconds with isolation disabled. The eleven paraphrases execute through the authenticated Flask chat endpoint within one Python test.
