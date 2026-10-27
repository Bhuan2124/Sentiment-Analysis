# Results discussion (template)

Fill each `<...>` from `results/test_results_summary.md` or the Model Comparison page. Delete any sentence you cannot back with a number from those files.
Nothing in this file is a result until you have filled it in.

## Overall performance
On the held-out test set of <n_test> reviews, <best model> reached the highest macro-F1 (<value>) and accuracy <value>.
The classical models scored <range of macro-F1> and the majority baseline scored <value>, so every model learned something beyond always predicting "positive".

## Classical versus transformer
The difference in macro-F1 between DistilBERT and the best classical model is <difference>. <Say whether this is large or small compared with the
size of the test set; remember there is only one split, so a very small difference may be noise.>

## Per-class behaviour
- Positive: F1 <values>. All models do well because this is the biggest class.
- Negative: F1 <values>.
- Neutral: F1 <values>. This is the weakest class for <models>, because neutral reviews are few and share words with both other classes.

## Error analysis
The confusion matrices show the most common mistake is <true class> predicted as <predicted class> (<count> reviews). Examples from the Model Insights page: <quote one or two real reviews that were misclassified and a possible reason>.

## Cost
Training took <time> for the classical models and <time> for DistilBERT. Prediction takes about <ms> per review for <model> and <ms> for DistilBERT on <CPU/GPU>.
Model sizes are <MB>. <State whether the quality gain justifies the cost.>

## Caveats
- Single split, no confidence interval.
- Rating-based labels are noisy.
- DistilBERT confidence values are not guaranteed to be calibrated; SVM scores are not probabilities.
