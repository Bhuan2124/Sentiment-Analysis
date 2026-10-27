"""
Phase 4 helpers: score models on the TEST split and measure what each model costs.

These are small, pure functions (no file or model loading) so they can be tested with tiny
made-up arrays. The script scripts/evaluate_models.py uses them with the real saved models.

Rules followed here (same as the earlier phases):
  * the test split is only used to SCORE models; nothing is tuned or fitted on it
  * the same test reviews are given to every model
  * every number written to disk was measured; a model that is not available is marked unavailable
"""

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src import config as C


# ---------------------------------------------------------------------------
# 1. Scores
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    """Accuracy, macro precision / recall / F1, per-class scores and the confusion matrix."""
    labels = list(range(len(C.LABEL_NAMES)))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same length, got {y_true.shape} and {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty set of predictions.")

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "weighted_f1": float(w_f1),
        },
        "weighted": {"precision": float(w_precision), "recall": float(w_recall), "f1": float(w_f1)},
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(C.LABEL_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def majority_baseline(y_train: Sequence[int], y_test: Sequence[int]) -> Dict:
    """Scores of 'always predict the most common training class'. Anything useful must beat this."""
    most_common = int(np.bincount(np.asarray(y_train), minlength=len(C.LABEL_NAMES)).argmax())
    result = evaluate_predictions(y_test, np.full(len(y_test), most_common))
    return {
        "name": "Majority baseline",
        "predicts": C.ID2LABEL[most_common],
        "accuracy": result["metrics"]["accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
    }


# ---------------------------------------------------------------------------
# 2. Cost
# ---------------------------------------------------------------------------
def median_seconds(fn: Callable[[], object], repeats: int = 3) -> float:
    """Run fn() `repeats` times and return the median wall-clock time in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def count_classical_parameters(pipeline) -> int:
    """Number of learned numbers in the classifier of a saved TF-IDF pipeline."""
    clf = pipeline.steps[-1][1]
    if hasattr(clf, "coef_"):
        return int(np.size(clf.coef_) + np.size(getattr(clf, "intercept_", [])))
    if hasattr(clf, "feature_log_prob_"):
        return int(np.size(clf.feature_log_prob_) + np.size(getattr(clf, "class_log_prior_", [])))
    return 0


def classical_training_time(
    key: str, params: Dict, train_texts: Sequence[str], train_labels: Sequence[int], repeats: int = 3
) -> float:
    """Seconds to fit TF-IDF + classifier with the chosen settings on the training split (median of `repeats`).

    Phase 2 did not time its final fits, so this re-fits the same settings once more, only to measure time.
    The re-fitted models are thrown away; the evaluation always uses the saved models.
    """
    from src import classical_models as cm

    texts = list(train_texts)

    def fit_once():
        vectorizer = cm.make_vectorizer()
        X = vectorizer.fit_transform(texts)
        cm.make_classifier(key, **params).fit(X, train_labels)

    return median_seconds(fit_once, repeats)


# ---------------------------------------------------------------------------
# 3. Readable summary
# ---------------------------------------------------------------------------
def summary_markdown(doc: Dict) -> str:
    """A small Markdown table of the evaluated models (used for the report and the README)."""
    lines = [
        f"# Test-set results ({doc['n_test']} reviews)",
        "",
        f"Evaluated: {doc.get('created_at', '')}",
        "",
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Training time | Per review | Size (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, entry in doc["models"].items():
        name = entry.get("display_name", key)
        if entry.get("status") != "evaluated":
            lines.append(f"| {name} | unavailable | | | | | | |")
            continue
        m, cost = entry["metrics"], entry.get("cost", {})
        train = cost.get("train_seconds")
        infer = cost.get("inference_ms_per_review")
        size = cost.get("model_size_mb")
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['macro_precision']:.3f} | {m['macro_recall']:.3f} | {m['macro_f1']:.3f} "
            f"| {'n/a' if train is None else f'{train:.1f} s'} | {'n/a' if infer is None else f'{infer:.3f} ms'} "
            f"| {'n/a' if size is None else f'{size:.2f}'} |"
        )
    base = doc.get("baseline")
    if base:
        lines.append(f"| {base['name']} (always '{base['predicts']}') | {base['accuracy']:.3f} | | | {base['macro_f1']:.3f} | | | |")
    lines += ["", "Training time: see `train_note` for each model in `test_results.json`.", ""]
    return "\n".join(lines)


def classification_report_text(name: str, entry: Dict) -> str:
    """A plain-text classification report built from the saved numbers."""
    pc, m = entry["per_class"], entry["metrics"]
    lines = [f"== {name} ==", f"{'':<14}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>9}"]
    for label in C.LABEL_NAMES:
        r = pc[label]
        lines.append(f"{label:<14}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>10.3f}{r['support']:>9d}")
    n = sum(pc[label]["support"] for label in C.LABEL_NAMES)
    lines.append(f"{'macro avg':<14}{m['macro_precision']:>10.3f}{m['macro_recall']:>10.3f}{m['macro_f1']:>10.3f}{n:>9d}")
    w = entry.get("weighted")
    if w:
        lines.append(f"{'weighted avg':<14}{w['precision']:>10.3f}{w['recall']:>10.3f}{w['f1']:>10.3f}{n:>9d}")
    lines.append(f"{'accuracy':<14}{'':>10}{'':>10}{m['accuracy']:>10.3f}{n:>9d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Figures for the report (matplotlib; the dashboard draws its own interactive charts)
# ---------------------------------------------------------------------------
def save_figures(doc: Dict, out_dir) -> List[str]:
    """Write test_model_comparison.png, test_per_class_f1.png and test_confusion_matrices.png. Returns the file names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated = {k: v for k, v in doc["models"].items() if v.get("status") == "evaluated"}
    if not evaluated:
        return []
    names = [v["display_name"] for v in evaluated.values()]
    colors = ["#6366F1", "#0EA5E9", "#0D9488", "#0F172A"][: len(names)]
    saved = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "macro_f1", "Macro F1-score")):
        vals = [v["metrics"][metric] for v in evaluated.values()]
        bars = ax.bar(names, vals, color=colors)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title} on the test set")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "test_model_comparison.png", dpi=150)
    plt.close(fig)
    saved.append("test_model_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(names)
    for i, (name, entry) in enumerate(zip(names, evaluated.values())):
        vals = [entry["per_class"][c]["f1"] for c in C.LABEL_NAMES]
        ax.bar(np.arange(len(vals)) + i * width, vals, width, label=name, color=colors[i])
    ax.set_xticks(np.arange(len(C.LABEL_NAMES)) + width * (len(names) - 1) / 2, [c.capitalize() for c in C.LABEL_NAMES])
    ax.set_ylim(0, 1.05)
    ax.set_title("F1-score per class on the test set")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "test_per_class_f1.png", dpi=150)
    plt.close(fig)
    saved.append("test_per_class_f1.png")

    cols = min(len(names), 2)
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, entry) in zip(axes.ravel(), zip(names, evaluated.values())):
        ax.axis("on")
        cm = np.array(entry["confusion_matrix"], dtype=float)
        pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for (i, j), v in np.ndenumerate(pct):
            ax.text(j, i, f"{v:.1f}%\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if v > 55 else "black", fontsize=9)
        ax.set_xticks(range(3), [c[:3] for c in C.LABEL_NAMES])
        ax.set_yticks(range(3), C.LABEL_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "test_confusion_matrices.png", dpi=150)
    plt.close(fig)
    saved.append("test_confusion_matrices.png")
    return saved
