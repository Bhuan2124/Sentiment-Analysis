# Conclusion (filled with the real results)

This project compared three classical machine-learning models (Logistic Regression, Multinomial Naive Bayes and Linear SVM on TF-IDF features) with a fine-tuned
DistilBERT transformer for three-class sentiment analysis of 22,633 cleaned e-commerce (women's clothing) reviews.

**Main finding.** DistilBERT performed best on the held-out test set of 3,395 reviews, with macro-F1 0.669 (accuracy 0.828), compared with 0.624 (accuracy 0.819) for the best
classical model, Naive Bayes. The 95% bootstrap interval for the gap (+0.045) is +0.019 to +0.070, so the advantage is real for this test set but modest, and it comes from one training run.

**Trade-off.** DistilBERT needs about 60 minutes of GPU training for its best run (177 minutes for the whole search), a 269 MB model and roughly 0.1 to 0.25 seconds per review on a laptop CPU.
Naive Bayes trains in about 9 seconds, is 4.9 MB and predicts in about 0.35 ms. Use DistilBERT when accuracy on hard (neutral and mixed) reviews matters more than speed and size;
use Naive Bayes or another classical model for a small, instant tool.

**Weakness.** All models find the neutral class hardest (test F1 0.357 to 0.463), which is expected because neutral reviews are rare (424 of 3,395) and mix positive and negative wording.

**Contribution.** A reproducible pipeline (leak-free splits, one shared test set, tuning on validation only), an automated test suite (119 tests passed on the project laptop) and an interactive
dashboard that loads the saved models without retraining.

**Future work.** Cross-validation and several training seeds for confidence intervals, human-checked error analysis, probability calibration, larger transformers, other product categories, aspect-based sentiment.
