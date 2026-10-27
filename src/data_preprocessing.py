"""
Data loading, cleaning, de-duplication and splitting.

Used by notebooks/01_data_exploration.ipynb now, and by the training scripts
in later phases (so every model sees exactly the same data).
"""

import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C


# ---------------------------------------------------------------------------
# 1. Finding and loading the raw file
# ---------------------------------------------------------------------------
def find_raw_file(raw_dir: Path = C.RAW_DIR) -> Path:
    """
    Return the path of the raw review CSV inside `raw_dir`.

    If only a .zip (what Kaggle gives you when you click Download) is found,
    the CSV inside it is extracted first.
    """
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    zip_files = sorted(raw_dir.glob("*.zip"))
    if zip_files:
        with zipfile.ZipFile(zip_files[0]) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if members:
                # Write only the file name (not the folder path stored in the zip)
                target = raw_dir / Path(members[0]).name
                target.write_bytes(zf.read(members[0]))
                return target

    raise FileNotFoundError(
        f"No CSV or ZIP file found in {raw_dir}.\n"
        "Download the dataset from Kaggle ('Women's E-Commerce Clothing Reviews' "
        "by nicapotato) and put the downloaded file into that folder."
    )


def load_raw_reviews(path: Path) -> pd.DataFrame:
    """Read the raw CSV exactly as it is (no cleaning yet)."""
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> List[str]:
    """
    Check the columns we depend on are present.
    Raises a clear error if not; returns the optional columns that exist.
    """
    missing = [c for c in C.REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}.\n"
            f"Columns found in the file: {list(df.columns)}"
        )
    return [c for c in C.OPTIONAL_RAW_COLUMNS if c in df.columns]


# ---------------------------------------------------------------------------
# 2. Cleaning + label creation
# ---------------------------------------------------------------------------
def clean_reviews(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Turn the raw table into the standard 4-column table:
        text | rating | label | label_id

    Steps:
      * keep only rows that have review text (can't classify an empty review)
      * keep only rows with a valid 1-5 star rating
      * map stars -> negative / neutral / positive -> 0 / 1 / 2

    Returns (clean_df, report) where report counts what was dropped.
    """
    validate_columns(df_raw)
    report = {"rows_in": len(df_raw)}

    df = pd.DataFrame(
        {
            C.TEXT_COL: df_raw[C.RAW_TEXT_COL],
            C.RATING_COL: pd.to_numeric(df_raw[C.RAW_RATING_COL], errors="coerce"),
        }
    )

    # Missing / empty review text
    df[C.TEXT_COL] = df[C.TEXT_COL].astype("string").str.strip()
    no_text = df[C.TEXT_COL].isna() | (df[C.TEXT_COL] == "")
    report["dropped_missing_text"] = int(no_text.sum())
    df = df.loc[~no_text].copy()

    # Rating -> label (anything outside 1..5 becomes NaN and is dropped)
    df[C.LABEL_COL] = df[C.RATING_COL].map(C.RATING_TO_LABEL)
    bad_rating = df[C.LABEL_COL].isna()
    report["dropped_invalid_rating"] = int(bad_rating.sum())
    df = df.loc[~bad_rating].copy()

    df[C.TEXT_COL] = df[C.TEXT_COL].astype(str)
    df[C.RATING_COL] = df[C.RATING_COL].astype(int)
    df[C.LABEL_ID_COL] = df[C.LABEL_COL].map(C.LABEL2ID).astype(int)
    df = df[C.PROCESSED_COLUMNS].reset_index(drop=True)

    report["rows_out"] = len(df)
    return df, report


# ---------------------------------------------------------------------------
# 3. Duplicates
# ---------------------------------------------------------------------------
def normalize_for_dedup(text: str) -> str:
    """
    A 'fingerprint' of a review: lower-case, punctuation removed, spaces collapsed.
    "Love it!!" and "love it" get the same fingerprint, so they count as duplicates.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def duplicate_report(df: pd.DataFrame) -> Dict[str, int]:
    """Count exact duplicates, fingerprint duplicates and label conflicts."""
    key = df[C.TEXT_COL].map(normalize_for_dedup)
    label_counts = df.groupby(key)[C.LABEL_COL].nunique()
    return {
        "rows": len(df),
        "exact_duplicate_rows": int(df[C.TEXT_COL].duplicated().sum()),
        "normalized_duplicate_rows": int(key.duplicated().sum()),
        "groups_with_conflicting_labels": int((label_counts > 1).sum()),
    }


def drop_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Remove duplicate reviews (by fingerprint) so the same text cannot appear
    in both the training and test sets.

    * If copies of one review carry DIFFERENT labels, the label is ambiguous,
      so ALL copies are dropped.
    * Otherwise the first copy is kept.
    """
    key = df[C.TEXT_COL].map(normalize_for_dedup)
    n_labels = df.groupby(key)[C.LABEL_COL].transform("nunique")
    conflicting = n_labels > 1

    report = {
        "rows_in": len(df),
        "dropped_conflicting_label_rows": int(conflicting.sum()),
    }
    df = df.loc[~conflicting].copy()
    key = key.loc[df.index]

    dup = key.duplicated(keep="first")
    report["dropped_duplicate_rows"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


# ---------------------------------------------------------------------------
# 4. Splitting (reproducible, stratified)
# ---------------------------------------------------------------------------
def make_splits(
    df: pd.DataFrame,
    test_size: float = C.TEST_SIZE,
    val_size: float = C.VAL_SIZE,
    seed: int = C.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test.

    'Stratified' means each split keeps the same percentage of negative /
    neutral / positive reviews as the full dataset. The seed makes the split
    identical every time.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[C.LABEL_ID_COL], random_state=seed
    )
    # val_size is a fraction of the WHOLE data, so rescale it for the remainder
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[C.LABEL_ID_COL], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def check_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict[str, int]:
    """Count reviews (by fingerprint) that appear in more than one split. Should all be 0."""
    keys = {
        name: set(split[C.TEXT_COL].map(normalize_for_dedup))
        for name, split in (("train", train), ("val", val), ("test", test))
    }
    return {
        "train_and_val": len(keys["train"] & keys["val"]),
        "train_and_test": len(keys["train"] & keys["test"]),
        "val_and_test": len(keys["val"] & keys["test"]),
    }


def save_splits(train, val, test, out_dir: Path = C.PROCESSED_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        split.to_csv(out_dir / C.SPLIT_FILENAMES[name], index=False)


def load_splits(in_dir: Path = C.PROCESSED_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the saved splits. Every later phase starts with this one line."""
    in_dir = Path(in_dir)
    out = []
    for name in ("train", "val", "test"):
        path = in_dir / C.SPLIT_FILENAMES[name]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run notebooks/01_data_exploration.ipynb first."
            )
        out.append(pd.read_csv(path, keep_default_na=False))
    return tuple(out)
