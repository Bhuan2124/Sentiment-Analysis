# Sentiment Analysis of E-Commerce Reviews: Classical ML vs. Transformer (DistilBERT)

Work in progress. The full README (overview, results table, run instructions, limitations)
is written in Phase 6, once real results exist.

## Status

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Dataset, setup, exploration, train/val/test splits | done |
| 2 | TF-IDF + Logistic Regression / Naive Bayes / Linear SVM | done (tuned on validation; test held out for Phase 4) |
| 3 | DistilBERT fine-tuning | done (tuned on validation; test held out for Phase 4) |
| 4 | Evaluation and comparison | not started |
| 5 | Streamlit web app | not started |
| 6 | Documentation, GitHub, slides, viva | not started |

## Dataset

Women's E-Commerce Clothing Reviews (Kaggle, uploaded by nicapotato), licence CC0: Public Domain.
Ratings are mapped to sentiment: 1-2 stars = negative, 3 = neutral, 4-5 = positive.
