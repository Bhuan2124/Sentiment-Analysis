"""
Central configuration for the whole project.

Every other file (notebooks, training scripts, the web app) imports its paths,
column names and label mappings from HERE, so they can never disagree.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders (all relative to the project root, wherever you put the project)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # the original Kaggle CSV goes here (never committed to Git)
PROCESSED_DIR = DATA_DIR / "processed"  # train/val/test CSVs created by Phase 1
MODELS_DIR = PROJECT_DIR / "models"     # saved models (Phases 2-3)
RESULTS_DIR = PROJECT_DIR / "results"   # metrics tables, summaries
FIGURES_DIR = RESULTS_DIR / "figures"   # saved charts

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Raw dataset columns: "Women's E-Commerce Clothing Reviews" (Kaggle, nicapotato)
# ---------------------------------------------------------------------------
RAW_TEXT_COL = "Review Text"
RAW_TITLE_COL = "Title"
RAW_RATING_COL = "Rating"
REQUIRED_RAW_COLUMNS = [RAW_TITLE_COL, RAW_TEXT_COL, RAW_RATING_COL]
# Columns we only use for extra analysis if they exist:
OPTIONAL_RAW_COLUMNS = ["Recommended IND", "Class Name", "Department Name", "Age"]

# ---------------------------------------------------------------------------
# Standardised column names used everywhere AFTER cleaning
# ---------------------------------------------------------------------------
TEXT_COL = "text"
RATING_COL = "rating"
LABEL_COL = "label"        # "negative" / "neutral" / "positive"
LABEL_ID_COL = "label_id"  # 0 / 1 / 2
PROCESSED_COLUMNS = [TEXT_COL, RATING_COL, LABEL_COL, LABEL_ID_COL]

# ---------------------------------------------------------------------------
# Label mapping (ONE fixed order, used by all four models)
# ---------------------------------------------------------------------------
LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}   # {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

# Star rating -> sentiment
RATING_TO_LABEL = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

# ---------------------------------------------------------------------------
# Train / validation / test split (fractions of the cleaned, de-duplicated data)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15   # train gets the remaining 70%
SPLIT_FILENAMES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}


# ---------------------------------------------------------------------------
# Phase 2: classical models (TF-IDF + linear classifiers)
# ---------------------------------------------------------------------------
CLASSICAL_MODELS_DIR = MODELS_DIR / "classical"
# Short key (used in file names) -> name shown in tables and charts
CLASSICAL_MODEL_NAMES = {
    "logreg": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "linear_svm": "Linear SVM",
}
# The ONE number used to choose settings and compare models. Macro-F1 gives each
# class equal weight, so a model cannot score well by ignoring negative/neutral.
SELECTION_METRIC = "macro_f1"

# ---------------------------------------------------------------------------
# Phase 3: transformer (DistilBERT)
# ---------------------------------------------------------------------------
TRANSFORMER_NAME = "distilbert-base-uncased"   # pre-trained model from the Hugging Face Hub
TRANSFORMER_DISPLAY_NAME = "DistilBERT"
TRANSFORMER_DIR = MODELS_DIR / "distilbert"     # the fine-tuned model is saved here
MAX_LENGTH = 160   # tokens per review; chosen in Notebook 3 Step 4 from the measured token lengths


def ensure_dirs() -> None:
    """Create the folders the project writes into (safe to call many times)."""
    for folder in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, CLASSICAL_MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
