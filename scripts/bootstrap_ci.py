"""
How much could the test scores change just by luck? (Bootstrap confidence intervals.)

    python scripts/bootstrap_ci.py

Reads results/test_predictions.csv (written by evaluate_models.py). It re-draws the 3,395 test reviews with replacement
2,000 times and recomputes macro-F1 each time, for every model and for DistilBERT minus each classical model on the SAME
re-drawn reviews. Writes results/bootstrap_ci.json and results/bootstrap_ci.md. Nothing is trained and no model is loaded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

N_BOOT, SEED = 2000, 42
MODELS = {"logreg": "Logistic Regression", "naive_bayes": "Naive Bayes", "linear_svm": "Linear SVM", "distilbert": "DistilBERT"}


def macro_f1_from_codes(y, p, k=3):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return f1.mean()


def main():
    df = pd.read_csv(ROOT / "results" / "test_predictions.csv")
    code = {name: i for i, name in enumerate(C.LABEL_NAMES)}
    y = df[C.LABEL_COL].map(code).to_numpy()
    preds = {k: df[f"pred_{k}"].map(code).to_numpy() for k in MODELS if f"pred_{k}" in df}
    rng = np.random.default_rng(SEED)
    n = len(df)
    scores = {k: np.empty(N_BOOT) for k in preds}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for k, p in preds.items():
            scores[k][b] = macro_f1_from_codes(y[idx], p[idx])
    out = {"n_test": n, "n_bootstrap": N_BOOT, "seed": SEED, "macro_f1": {}, "distilbert_minus": {}}
    for k, s in scores.items():
        out["macro_f1"][k] = {"point": float(macro_f1_from_codes(y, preds[k])), "low": float(np.percentile(s, 2.5)), "high": float(np.percentile(s, 97.5))}
    if "distilbert" in scores:
        for k in scores:
            if k == "distilbert":
                continue
            d = scores["distilbert"] - scores[k]
            out["distilbert_minus"][k] = {"point": out["macro_f1"]["distilbert"]["point"] - out["macro_f1"][k]["point"],
                                          "low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5)),
                                          "share_of_resamples_distilbert_better": float((d > 0).mean())}
    (ROOT / "results" / "bootstrap_ci.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [f"# Bootstrap 95% intervals for macro-F1 (test set, {n} reviews, {N_BOOT} resamples)", "",
             "| Model | Macro-F1 | 95% interval |", "|---|---|---|"]
    for k, v in out["macro_f1"].items():
        lines.append(f"| {MODELS[k]} | {v['point']:.3f} | {v['low']:.3f} to {v['high']:.3f} |")
    lines += ["", "DistilBERT minus each classical model (same resampled reviews):", "",
              "| Compared with | Difference | 95% interval | Resamples where DistilBERT is better |", "|---|---|---|---|"]
    for k, v in out["distilbert_minus"].items():
        lines.append(f"| {MODELS[k]} | {v['point']:+.3f} | {v['low']:+.3f} to {v['high']:+.3f} | {v['share_of_resamples_distilbert_better'] * 100:.1f}% |")
    lines += ["", "An interval that does not include 0 means the gap is larger than test-set luck alone would usually explain.",
              "It does not cover randomness in training (another seed could move DistilBERT a little)."]
    (ROOT / "results" / "bootstrap_ci.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
