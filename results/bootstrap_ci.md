# Bootstrap 95% intervals for macro-F1 (test set, 3395 reviews, 2000 resamples)

| Model | Macro-F1 | 95% interval |
|---|---|---|
| Logistic Regression | 0.598 | 0.575 to 0.619 |
| Naive Bayes | 0.624 | 0.601 to 0.646 |
| Linear SVM | 0.599 | 0.575 to 0.622 |
| DistilBERT | 0.669 | 0.647 to 0.691 |

DistilBERT minus each classical model (same resampled reviews):

| Compared with | Difference | 95% interval | Resamples where DistilBERT is better |
|---|---|---|---|
| Logistic Regression | +0.072 | +0.046 to +0.096 | 100.0% |
| Naive Bayes | +0.045 | +0.019 to +0.070 | 99.9% |
| Linear SVM | +0.070 | +0.043 to +0.096 | 100.0% |

An interval that does not include 0 means the gap is larger than test-set luck alone would usually explain.
It does not cover randomness in training (another seed could move DistilBERT a little).
