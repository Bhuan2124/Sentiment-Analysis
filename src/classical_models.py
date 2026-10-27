"""
Phase 2: TF-IDF features + three classical classifiers.

    Logistic Regression | Naive Bayes | Linear SVM

Used by notebooks/02_classical_models.ipynb now, and by the evaluation notebook
(Phase 4) and the web app (Phase 5), which load the saved models from disk.

Rules followed here:
  * the TF-IDF vectorizer is fitted on the TRAINING split only
  * settings (C, alpha, ...) are chosen on the VALIDATION split only
  * the TEST split is never used in this file
"""

from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config as C


# ---------------------------------------------------------------------------
# 1. TF-IDF settings (shared by all three models)
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),   # single words AND word pairs, so "not good" becomes one feature
    "min_df": 2,             # ignore words/pairs seen in only one review (mostly typos)
    "max_df": 0.95,          # ignore terms that appear in almost every review
    "sublinear_tf": True,    # 1 + log(count): a word repeated 5 times is not 5x as important
    "stop_words": None,      # NO stop-word list: it would delete "not", "no" (see Notebook 1)
}


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(**TFIDF_PARAMS)


# ---------------------------------------------------------------------------
# 2. The models and the settings we try for each
# ---------------------------------------------------------------------------
# Each grid is small on purpose: every combination is trained once and scored
# on the validation split.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "logreg": {
        "C": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0],   # bigger C = less regularisation
        "class_weight": [None, "balanced"],         # "balanced" = rare classes count more
    },
    "naive_bayes": {
        "alpha": [0.01, 0.03, 0.1, 0.3, 1.0],       # smoothing for unseen words
        "fit_prior": [True, False],                 # False = treat the 3 classes as equally likely
    },
    "linear_svm": {
        "C": [0.03, 0.1, 0.3, 1.0, 3.0],
        "class_weight": [None, "balanced"],
    },
}


def make_classifier(key: str, **params):
    """Create an (untrained) classifier for model `key` with the given settings."""
    if key == "logreg":
        return LogisticRegression(max_iter=3000, random_state=C.RANDOM_SEED, **params)
    if key == "naive_bayes":
        return MultinomialNB(**params)
    if key == "linear_svm":
        return LinearSVC(random_state=C.RANDOM_SEED, **params)
    raise ValueError(f"Unknown model key {key!r}. Choose from {list(C.CLASSICAL_MODEL_NAMES)}")


def param_combinations(grid: Dict[str, list]) -> List[Dict]:
    """{"C": [1, 3], "w": [a, b]} -> [{"C": 1, "w": a}, {"C": 1, "w": b}, ...]"""
    names = list(grid)
    return [dict(zip(names, values)) for values in product(*(grid[n] for n in names))]


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Accuracy, macro-F1, weighted-F1 and the F1 of each class."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(C.LABEL_NAMES))), zero_division=0
    )
    for name, f1 in zip(C.LABEL_NAMES, f1_per_class):
        out[f"f1_{name}"] = f1
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 4. Tuning on the validation split
# ---------------------------------------------------------------------------
def tune_model(
    key: str,
    X_train,
    y_train,
    X_val,
    y_val,
    grid: Dict[str, list] = None,
) -> Tuple[pd.DataFrame, object, Dict]:
    """
    Train model `key` once per setting in its grid and score each on validation.

    X_train / X_val are TF-IDF matrices (the vectorizer is fitted once outside,
    so we do not repeat that work for every setting).

    Returns (results_table, best_trained_classifier, best_params).
    """
    grid = PARAM_GRIDS[key] if grid is None else grid
    rows, best = [], None
    for params in param_combinations(grid):
        clf = make_classifier(key, **params).fit(X_train, y_train)
        metrics = compute_metrics(y_val, clf.predict(X_val))
        rows.append({**{k: str(v) for k, v in params.items()}, **metrics})
        if best is None or metrics[C.SELECTION_METRIC] > best[0]:
            best = (metrics[C.SELECTION_METRIC], clf, params)

    table = pd.DataFrame(rows).sort_values(C.SELECTION_METRIC, ascending=False).reset_index(drop=True)
    return table, best[1], best[2]


def majority_baseline(y_train, y_val) -> Dict[str, float]:
    """Scores of a 'model' that always predicts the most common class (positive)."""
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    return compute_metrics(y_val, dummy.predict(np.zeros((len(y_val), 1))))


# ---------------------------------------------------------------------------
# 5. Explaining a linear model
# ---------------------------------------------------------------------------
def top_features(vectorizer: TfidfVectorizer, clf, n: int = 15) -> Dict[str, List[str]]:
    """
    The n words/pairs that push most strongly towards each class.
    Works for Logistic Regression and Linear SVM (both have one weight per feature per class).
    """
    vocab = np.array(vectorizer.get_feature_names_out())
    return {
        name: list(vocab[np.argsort(clf.coef_[i])[::-1][:n]])
        for i, name in enumerate(C.LABEL_NAMES)
    }


# ---------------------------------------------------------------------------
# 6. Saving and loading (vectorizer + classifier together)
# ---------------------------------------------------------------------------
def build_pipeline(vectorizer: TfidfVectorizer, clf) -> Pipeline:
    """
    Wrap an already-fitted vectorizer and classifier into one object, so later
    code can call  model.predict(["some raw review text"])  directly.
    """
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def model_path(key: str, models_dir: Path = C.CLASSICAL_MODELS_DIR) -> Path:
    return Path(models_dir) / f"{key}.joblib"


def save_model(pipeline: Pipeline, key: str, models_dir: Path = C.CLASSICAL_MODELS_DIR) -> Path:
    path = model_path(key, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(key: str, models_dir: Path = C.CLASSICAL_MODELS_DIR) -> Pipeline:
    """Load a saved model. Phase 4 and the web app start with this."""
    path = model_path(key, models_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebooks/02_classical_models.ipynb first.")
    return joblib.load(path)
