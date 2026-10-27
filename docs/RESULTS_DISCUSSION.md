# Results discussion (filled with the real test-set numbers)

Every number below comes from `results/test_results_summary.md`, `results/test_results.json`, `results/test_classification_reports.txt`,
`results/distilbert_summary.json` or `results/bootstrap_ci.md`. The earlier file `RESULTS_DISCUSSION_TEMPLATE.md` is the blank version.

## Overall performance
On the held-out test set of 3,395 reviews, DistilBERT reached the highest macro-F1 (0.669) and the highest accuracy (0.828).
The classical models scored macro-F1 0.598 to 0.624 (Logistic Regression 0.598, Linear SVM 0.599, Naive Bayes 0.624). The majority baseline ("always positive")
scored macro-F1 0.290 and accuracy 0.771, so every model learned something beyond always answering "positive".

## Classical versus transformer
DistilBERT beats the best classical model (Naive Bayes) by 0.045 macro-F1, and Logistic Regression and Linear SVM by about 0.07.
A bootstrap over the 3,395 test reviews (2,000 resamples) gives these 95% intervals for the gap: +0.046 to +0.096 against Logistic Regression,
+0.019 to +0.070 against Naive Bayes and +0.043 to +0.096 against Linear SVM. None of them includes 0, and DistilBERT was better in at least 99.9% of resamples.
So the gap is larger than test-set luck alone would explain. It does not cover training randomness: DistilBERT was trained once with one seed.
In plain words: the transformer is better, but the advantage over Naive Bayes is modest.

Accuracy tells a smaller story: 0.828 for DistilBERT against 0.810 to 0.819 for the classical models, because most reviews are positive and every model gets those right.
This is why macro-F1 is the main metric.

## Per-class behaviour (F1 on the test set)
- Positive: 0.921 to 0.931 for all four models. Easy, because it is the biggest class (2,616 of 3,395 test reviews).
- Negative: 0.497 to 0.613. DistilBERT is best at 0.613; Naive Bayes is lowest at 0.497.
- Neutral: 0.357 to 0.463. This is the weakest class for every model. Logistic Regression and Linear SVM are lowest (0.358 and 0.357); DistilBERT is highest (0.463).
  Neutral reviews are few (424 in the test set) and share words with both other classes.

## Error analysis
The confusion matrices (`results/test_results.json`, Model Comparison page) show that most mistakes involve the neutral class.
For DistilBERT the largest error counts are 232 positive reviews predicted as neutral, 120 negative reviews predicted as neutral and 113 neutral reviews predicted as negative.
For Logistic Regression the largest are 159 neutral reviews predicted as positive and 144 positive reviews predicted as neutral.
DistilBERT catches more real neutral reviews (recall 0.552 against 0.351 for Logistic Regression) but at the price of lower neutral precision (0.399), because it also labels many positive reviews as neutral.
For real misclassified reviews to quote in the presentation, use the Model Insights page or `results/test_predictions.csv`.

## Cost
- Training: the classical models take 8.7 to 31.3 seconds to re-fit on a CPU (median of three runs, measured by the evaluation script).
  DistilBERT's best run took 60.3 minutes on an NVIDIA GeForce GTX 1650; all four tuning runs together took 177.3 minutes.
- Prediction: about 0.27 to 0.35 ms per review for the classical models on the CPU. DistilBERT took 242 ms per review in the full test-set pass on the CPU, and 97 ms in a smaller
  speed check (`inference_ms_per_review_cpu`). Quote it as "roughly 0.1 to 0.25 seconds per review on a laptop CPU", which depends on the computer.
- Size: 3.3 to 4.9 MB for the classical models, 268.5 MB (67.0 million parameters) for DistilBERT.
- Judgement: DistilBERT gives the best macro-F1 and finds more neutral reviews, but it is roughly 300 to 900 times slower to predict and about 55 to 80 times larger.
  If speed, memory or simplicity matters most, Naive Bayes is a reasonable choice, since it is the best classical model here.

## Caveats
- One split and one training seed. The bootstrap intervals measure test-set sampling noise only.
- Rating-based labels are noisy, particularly the 3-star neutral class.
- The classical models' validation macro-F1 (0.639 to 0.652) was higher than their test macro-F1 (0.598 to 0.624), because their settings were chosen on the validation set. DistilBERT's validation (0.668) and test (0.669) scores match closely.
- The models were trained on women's clothing reviews and may do worse on other products.
- DistilBERT confidence values are not guaranteed to be calibrated; SVM scores are not probabilities.
