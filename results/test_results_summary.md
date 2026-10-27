# Test-set results (3395 reviews)

Evaluated: 2026-10-28 00:10:13

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Training time | Per review | Size (MB) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.810 | 0.600 | 0.595 | 0.598 | 31.3 s | 0.345 ms | 3.28 |
| Naive Bayes | 0.819 | 0.625 | 0.628 | 0.624 | 8.7 s | 0.352 ms | 4.92 |
| Linear SVM | 0.819 | 0.610 | 0.591 | 0.599 | 9.0 s | 0.270 ms | 3.28 |
| DistilBERT | 0.828 | 0.658 | 0.689 | 0.669 | 3618.0 s | 242.039 ms | 268.55 |
| Majority baseline (always 'positive') | 0.771 | | | 0.290 | | | |

Training time: see `train_note` for each model in `test_results.json`.
