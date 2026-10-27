# Sentiment Analysis of E-Commerce Reviews: Classical ML vs. Transformer (DistilBERT)

Three-class sentiment analysis (negative / neutral / positive) of women's clothing reviews, comparing three classical models on TF-IDF features with a
fine-tuned DistilBERT, plus a six-page Streamlit dashboard.

> **Status of results.** Every number in this file comes from a file in `results/` produced by this project's own code (see the file names next to each table).
> Training was done once, with seed 42. `README.md` is the phase-tracking file kept by the training notebooks; this file is the full project README.

## What is in the project

```
app.py                  Streamlit entry point        webapp/    dashboard code (pages in webapp/views/)
src/                    data prep + model training   scripts/   setup, tests, evaluation, smoke check, launcher
data/                   raw + processed splits       tests/     automated tests
models/                 saved models (not in git)    docs/      presentation outline, methodology, viva Q&A, templates
results/                measured results + figures   notebooks/  Phase 1 to 3 notebooks
```

## Data and method (short)
- Women's E-Commerce Clothing Reviews (Kaggle, CC0): 23,486 rows; after cleaning and de-duplication 22,633 reviews.
- Ratings 1-2 = negative, 3 = neutral, 4-5 = positive. Strongly imbalanced: about 77% positive.
- Stratified split with seed 42: 15,843 train / 3,395 validation / 3,395 test, saved in `data/processed/`.
- TF-IDF (1-2-grams) fitted on the training set only. Settings chosen on the validation set by macro-F1. The test set is used once, for the final scoring.
- Details: `docs/METHODOLOGY_SUMMARY.md`.

## Results

### Validation scores of the classical models (real, from `results/classical_val_results.csv`)
These are **validation** scores used to choose settings, not final results.

| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| Naive Bayes | 0.8274 | 0.6519 | 0.5668 | 0.4602 | 0.9288 |
| Logistic Regression | 0.8297 | 0.6398 | 0.5698 | 0.4197 | 0.9301 |
| Linear SVM | 0.8351 | 0.6385 | 0.5722 | 0.4102 | 0.9330 |
| Majority baseline (always "positive") | 0.7705 | 0.2901 | 0.0 | 0.0 | 0.8704 |

All three classical models are within 0.014 macro-F1 of each other on validation; with one split that gap is small. The neutral class is clearly the hardest.

### Final test-set results (3,395 reviews, scored once)
From `results/test_results_summary.md`. Macro-F1 is the main metric.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | 95% interval for macro-F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.810 | 0.600 | 0.595 | 0.598 | 0.575 to 0.619 |
| Naive Bayes | 0.819 | 0.625 | 0.628 | 0.624 | 0.601 to 0.646 |
| Linear SVM | 0.819 | 0.610 | 0.591 | 0.599 | 0.575 to 0.622 |
| **DistilBERT** | **0.828** | **0.658** | **0.689** | **0.669** | **0.647 to 0.691** |
| Majority baseline (always "positive") | 0.771 | | | 0.290 | |

The intervals come from `scripts/bootstrap_ci.py` (2,000 resamples of the test reviews, seed 42; output in `results/bootstrap_ci.md`). DistilBERT's macro-F1 is higher than each
classical model's, with gap intervals that exclude 0 (+0.046 to +0.096 vs Logistic Regression, +0.019 to +0.070 vs Naive Bayes, +0.043 to +0.096 vs Linear SVM).
The intervals cover test-set sampling only, not training randomness.

| F1 per class | Negative | Neutral | Positive |
|---|---|---|---|
| Logistic Regression | 0.513 | 0.358 | 0.921 |
| Naive Bayes | 0.497 | 0.446 | 0.929 |
| Linear SVM | 0.514 | 0.357 | 0.927 |
| DistilBERT | 0.613 | 0.463 | 0.931 |

| Cost | Training | Prediction per review (CPU) | Model size |
|---|---|---|---|
| Logistic Regression | 31.3 s (re-fit, CPU) | 0.345 ms | 3.3 MB |
| Naive Bayes | 8.7 s (re-fit, CPU) | 0.352 ms | 4.9 MB |
| Linear SVM | 9.0 s (re-fit, CPU) | 0.270 ms | 3.3 MB |
| DistilBERT | 60.3 min for the best run on a GTX 1650 (177.3 min for all four runs) | 242 ms in the full test pass; 97 ms in a smaller speed check | 268.5 MB |

Discussion and conclusion text: `docs/RESULTS_DISCUSSION.md`, `docs/CONCLUSION.md`.

### DistilBERT
Fine-tuned `distilbert-base-uncased` (max length 160, 4 epochs at most with early stopping), four runs, compared on **validation** macro-F1 (`results/distilbert_search.csv`):

| Run | Learning rate | Class weights | Best epoch | Validation macro-F1 |
|---|---|---|---|---|
| 2 (chosen) | 2e-5 | yes | 4 | 0.6677 |
| 3 | 5e-5 | no | 3 | 0.6649 |
| 1 | 2e-5 | no | 2 | 0.6643 |
| 4 | 5e-5 | yes | 4 | 0.6629 |

The four runs are within 0.005 of each other, so the choice between them is not strongly supported. Training ran in `notebooks/03_distilbert.ipynb`
(log: `results/distilbert_training_log.txt`); the model is saved in `models/distilbert/`. The final test evaluation ran DistilBERT on the CPU.

### Validation versus test
The classical models scored lower on test (macro-F1 0.598 to 0.624) than on validation (0.639 to 0.652), because their settings were chosen on the validation set.
DistilBERT scored 0.668 on validation and 0.669 on test.

## How to run (Windows PowerShell, from the project folder)

```powershell
# 1. one-time setup: creates .venv and installs the packages (needs internet, several minutes)
python scripts/setup_env.py

# 2. activate the environment
.\.venv\Scripts\Activate.ps1

# 3. check everything
python scripts/run_tests.py          # report: results/test_report.txt
python scripts/smoke_check.py        # report: results/smoke_check.txt

# 4. final scoring on the test set (after DistilBERT training has finished)
python scripts/evaluate_models.py

# 5. start the dashboard
python -m streamlit run app.py       # opens http://localhost:8501
```

If PowerShell refuses to run `Activate.ps1`, skip step 2 and prefix commands with `.\.venv\Scripts\python.exe`, for example
`.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Dashboard pages
Overview Dashboard, Sentiment Analyzer (type a review, choose a model or compare all), Model Comparison, Data Explorer, Model Insights, About Project.
The dashboard only loads saved models and result files; it never trains. DistilBERT runs on the CPU by default (set `SENTIMENT_DEVICE=cuda` to use the GPU).

## Reading the scores correctly
- Logistic Regression and Naive Bayes show probabilities. The Linear SVM shows **decision scores, not probabilities**, so no percentage is displayed for it.
- DistilBERT shows softmax scores that add up to 1, but they are **not guaranteed to be calibrated**.
- Macro-F1 is the main metric because the data is imbalanced; accuracy alone rewards always answering "positive".

## Tests
`tests/` covers text preprocessing, label mapping, dataset validation, model loading, prediction output, empty input, evaluation maths (against hand-computed values),
missing and damaged files, and a static check that the dashboard never trains. Streamlit page tests (`tests/test_app_ui.py`) run when Streamlit is installed.
The measured outcome of the latest run is in `results/test_report.txt`.

## Limitations
One product domain and one language; star ratings are noisy labels; one test split and one training seed (the bootstrap intervals cover test-set sampling only); the models were trained on clothing reviews and may do worse on other products; the neutral class is small and hard; sarcasm and mixed opinions are difficult;
DistilBERT confidence is not calibrated; the CPU speed of DistilBERT depends on the computer.

## Documents for the presentation
`docs/PRESENTATION_OUTLINE.md`, `docs/METHODOLOGY_SUMMARY.md`, `docs/RESULTS_DISCUSSION.md`, `docs/CONCLUSION.md` (blank versions: `*_TEMPLATE.md`), `docs/VIVA_QA.md`.
