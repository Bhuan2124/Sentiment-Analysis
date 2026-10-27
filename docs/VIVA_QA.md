# Viva questions and answers

Short answers in simple language. Numbers are not written here on purpose: read them from the dashboard's
**Model Comparison** page or from `results/test_results_summary.md`, so they always match the files.

## The problem and the data

**1. What problem does the project solve?**
Online shops get thousands of reviews. Reading them all is impossible, so the project teaches a computer to
label each review as *positive*, *negative* or *neutral*.

**2. Which dataset did you use and why?**
The Women's E-Commerce Clothing Reviews dataset from Kaggle (CC0 licence, so free to use). It has real review text
and a 1 to 5 star rating, which gives us a ready-made label without hand labelling.

**3. How did you turn star ratings into three classes?**
1 and 2 stars are negative, 3 stars is neutral, 4 and 5 stars are positive. This is a common convention, but it is a
choice: a 3-star review can be a happy review and a 4-star review can contain complaints.

**4. Is the data balanced?**
No. Most reviews are positive. That is why accuracy alone is misleading: a model that always says "positive" already
gets a high accuracy. The project includes that "always positive" model as a *majority baseline* so you can see how much
the real models improve on it.

**5. What cleaning did you do?**
Removed rows with no review text, removed duplicate reviews (checking that duplicates never carry conflicting labels),
and split the data once into train, validation and test sets with a fixed random seed (42).

## Methods

**6. What is TF-IDF?**
A way to turn text into numbers. Each word (or pair of words) gets a score: high if it appears often in a review but is
rare across all reviews. Common words like "the" get low scores.

**7. Why three classical models?**
They are simple, fast and strong baselines on text: Logistic Regression (weighs words to compute class probabilities),
Multinomial Naive Bayes (counts how likely each word is in each class), Linear SVM (finds a separating line with a wide margin).

**8. What is DistilBERT and why is it different?**
A smaller, faster version of BERT, a neural network pre-trained on huge amounts of English text. Unlike TF-IDF it reads
words in context ("not good" is different from "good"). We *fine-tune* it, which means we continue training it on our reviews.

**9. What does fine-tuning mean?**
Starting from a model that already knows English and training it a little more on our labelled reviews, with a small
learning rate so it adapts without forgetting.

**10. How did you pick settings (hyper-parameters)?**
By comparing validation scores only. The test set was kept locked until the very end.

**11. What is data leakage and how did you prevent it?**
Leakage means information from validation or test data sneaks into training, which makes results look better than they are.
The TF-IDF vocabulary is learned from the training set only, tuning uses the validation set only, and every model is scored once
on the same untouched test set.

**12. Why macro-F1 instead of accuracy?**
Macro-F1 averages the F1 score of the three classes equally, so a weak result on the small neutral class cannot hide behind the
large positive class. We use it to choose models.

**13. What are precision, recall and F1?**
Precision: of the reviews the model called "negative", how many really were. Recall: of all truly negative reviews, how many the
model found. F1 combines the two (their harmonic mean).

**14. What does the confusion matrix show?**
Rows are the true classes and columns are the predicted classes. The diagonal is correct answers; everything else is a specific
kind of mistake, such as neutral reviews predicted as positive.

## Results and honesty

**15. Which model is best?**
Read it from the Model Comparison page. Use the test-set macro-F1 and also mention accuracy, the neutral-class result and the cost
(training time, speed, size). Do not claim a winner from validation numbers once test numbers exist.

**16. Why is the neutral class the hardest?**
Neutral reviews are few, and they mix praise with complaints, so their words overlap with both other classes. All models score
lowest on neutral.

**17. Are the confidence numbers trustworthy?**
Not automatically. Logistic Regression and Naive Bayes give probabilities, but they may be over- or under-confident.
DistilBERT's softmax numbers add up to 1 but are not guaranteed to be *calibrated*, meaning "90%" does not necessarily mean it is right 90% of the time.
The Linear SVM gives distance-like decision scores, not probabilities, so the dashboard shows no percentage for it.

**18. Does the dashboard train a model when I open it?**
No. Models are trained beforehand and saved. The dashboard only loads the saved files and predicts. There is a test that checks the
dashboard code contains no training calls.

**19. What if a model file is missing?**
The dashboard says which model is unavailable and why, and keeps working with the others. Pages for results that do not exist yet show a
"pending" message instead of invented numbers.

**20. What are the limitations?**
One dataset about women's clothing; English only; star ratings are noisy labels; three coarse classes; the test set is one split (no
confidence interval); sarcasm and mixed opinions are hard; DistilBERT needs more compute than the classical models.

**21. How would you improve it?**
Cross-validation or repeated seeds for confidence intervals, error analysis with a human-checked sample of reviews, a larger transformer,
probability calibration, other product categories, and aspect-based sentiment (fit, fabric, delivery).

## Practical

**22. How do I run the project?**
`python -m streamlit run app.py` from the project folder (after `python scripts/setup_env.py` has created the `.venv`).

**23. How is the code organised?**
`src/` holds the data and model training code, `scripts/` holds the run, test and evaluation commands, `webapp/` holds the dashboard,
`tests/` holds the automated tests, `models/` and `results/` hold the saved models and measured results.

**24. How do you know the code works?**
Automated tests cover text preprocessing, label mapping, dataset checks, model loading, prediction output, empty input, evaluation maths
(checked against hand-computed values) and missing files. The test report is saved in `results/test_report.txt`.

**25. Why does DistilBERT run on the CPU in the dashboard?**
So the dashboard works on any laptop and does not fight with training for the graphics card. Predicting one review on a CPU takes
a fraction of a second. Set `SENTIMENT_DEVICE=cuda` to use the GPU.
