# Presentation outline (about 12 slides, 10 to 12 minutes)

Take every number from the dashboard (Model Comparison page) or `results/test_results_summary.md`. Leave a slide out rather than
quote a number you cannot find in those files.

1. **Title.** Sentiment Analysis of E-Commerce Reviews: Classical ML vs. Transformer (DistilBERT). Your name, guide, date.
2. **Problem and motivation.** Too many reviews to read; automatic sentiment helps sellers and shoppers.
3. **Objectives.** Build three classical models and one fine-tuned transformer; compare them fairly; deliver an interactive dashboard.
4. **Dataset.** Women's E-Commerce Clothing Reviews (Kaggle, CC0). Show the cleaning counts and the class distribution (Overview page).
5. **Labels and splits.** Rating to class mapping; train/validation/test sizes; seed 42; "test set touched once".
6. **Approach A: classical.** TF-IDF + Logistic Regression, Naive Bayes, Linear SVM. One sentence on each.
7. **Approach B: DistilBERT.** Pre-training then fine-tuning; mention the 4 tuning runs and that they were compared on validation only.
8. **Fair evaluation.** No leakage, same test set for all, macro-F1 as main metric, majority baseline for reference.
9. **Results.** Model Comparison table and bar chart. Say which model won on macro-F1 and by how much.
10. **Where models fail.** Confusion matrices and per-class F1; the neutral class; two or three real misclassified reviews (Model Insights page).
11. **Cost versus quality.** Training time, speed per review, model size. Is the gain worth the cost?
12. **Live demo.** Sentiment Analyzer: one clear positive, one clear negative, one mixed review; compare all models.
13. **Limitations and future work.**
14. **Conclusion and questions.**

## Demo script (2 minutes)
1. Overview page: point at the dataset size and class imbalance.
2. Analyzer: paste "Lovely dress, fits perfectly and the fabric is soft. I love it!" and run all models.
3. Paste "It is okay. Nice colour but it runs small and the material is a bit thin." and show that models can disagree on neutral reviews.
4. Model Comparison: show the winner and the cost table.
5. Say clearly: SVM numbers are scores, not probabilities; DistilBERT confidence is not guaranteed calibrated.
