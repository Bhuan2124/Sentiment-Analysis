# Conclusion (template)

Fill the `<...>` parts from the real results. Nothing here is a finding until it is filled in.

This project compared three classical machine-learning models (Logistic Regression, Multinomial Naive Bayes, Linear SVM on TF-IDF features) with a fine-tuned
DistilBERT transformer for three-class sentiment analysis of <n> e-commerce reviews.

**Main finding.** <Best model> performed best on the held-out test set with macro-F1 <value> (accuracy <value>), compared with <value> for the best <classical / transformer> alternative.
<One sentence on whether the gap is practically meaningful.>

**Trade-off.** <Model> needs <training time> and <size> compared with <time> and <size> for <other model>. <State which you would use in which situation.>

**Weakness.** All models find the neutral class hardest (F1 <range>), which is expected because neutral reviews are rare and mix positive and negative wording.

**Contribution.** A reproducible pipeline (leak-free splits, shared test set, tuning on validation only) and an interactive dashboard that loads the saved models without retraining.

**Future work.** Cross-validation for confidence intervals, human-checked error analysis, probability calibration, larger transformers, other product categories, aspect-based sentiment.
