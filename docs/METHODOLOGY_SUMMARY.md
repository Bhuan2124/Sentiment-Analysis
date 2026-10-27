# Methodology summary

## 1. Data
- Source: Women's E-Commerce Clothing Reviews (Kaggle, `nicapotato/womens-ecommerce-clothing-reviews`, CC0).
- Text used: the "Review Text" column only (the title is not used). Whitespace is trimmed; see `src/data_preprocessing.py`.
- Cleaning: rows without review text removed; invalid ratings removed; exact and normalised duplicates removed, with a check that duplicates
  never disagree on the label. Counts for every step are in `results/dataset_summary.json`.
- Labels: rating 1 to 2 = negative (0), 3 = neutral (1), 4 to 5 = positive (2).
- Split: stratified train / validation / test with seed 42, saved to `data/processed/` so every model sees identical data.

## 2. Models
| Model | Input | Output scores |
|---|---|---|
| Logistic Regression | TF-IDF (1 to 2-grams) | probabilities |
| Multinomial Naive Bayes | TF-IDF (1 to 2-grams) | probabilities |
| Linear SVM | TF-IDF (1 to 2-grams) | decision scores (not probabilities) |
| DistilBERT (`distilbert-base-uncased`, fine-tuned) | word pieces, max length from `src/config.py` | softmax (not guaranteed calibrated) |

TF-IDF is fitted on the training set only, inside a scikit-learn pipeline, so the vocabulary never sees validation or test text.

## 3. Tuning and selection
- Classical models: grid search over the grids listed in `results/classical_summary.json`, scored by macro-F1 on the validation set.
- DistilBERT: four runs (learning rate 2e-5 or 5e-5, with or without class weights), several epochs each, the best epoch and the best run chosen on validation macro-F1
  (the exact settings actually used are in the training log and `results/distilbert_summary.json`).
  Details: `results/distilbert_search.csv`, `results/distilbert_summary.json`.
- The test set is not used for any choice.

## 4. Final evaluation
`scripts/evaluate_models.py` loads every saved model, predicts on the one test set, and writes metrics (accuracy, macro and weighted precision, recall, F1,
per-class scores, confusion matrices), a majority baseline, and cost measures (training time, speed per review, model size, parameters).
Classical training time is measured by re-fitting the chosen settings (median of three runs) and is labelled as such.
DistilBERT training time is taken from the training log.

## 5. The dashboard
Reads saved models and result files only. It never trains, tunes or overwrites a model. Each page shows a "pending" message when its result files do not exist.

## 6. Reproducibility
Fixed seed, pinned library versions (`requirements-app.txt`), a project-local `.venv`, saved splits and saved result files.

## 7. Threats to validity
- Star ratings are noisy proxies for sentiment.
- A single test split gives one estimate without a confidence interval; small differences between models may not be meaningful.
- One domain (women's clothing) and one language.
- DistilBERT training involves randomness; results for another seed may differ a little.
