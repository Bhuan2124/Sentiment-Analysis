"""
Phase 4: score every saved model ONCE on the untouched test split and measure what each model costs.

    python scripts/evaluate_models.py                 # all models that exist
    python scripts/evaluate_models.py --skip-distilbert

Writes (all real measurements):
    results/test_results.json            read by the dashboard
    results/test_results_summary.md       a table for the report
    results/test_classification_reports.txt
    results/test_predictions.csv         every test review with every model's prediction
    results/figures/test_*.png

It loads saved models only. Nothing is trained here (the classical models are re-fitted once, only to time them,
and then thrown away). A model that is missing is recorded as "not_available" instead of being invented.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import classical_models as cm  # noqa: E402
from src import config as C  # noqa: E402
from src import data_preprocessing as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402

MODEL_KEYS = ["logreg", "naive_bayes", "linear_svm", "distilbert"]


def read_json(path: Path):
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()}
    try:
        import sklearn

        env["scikit_learn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = "not installed"
    return env


def evaluate_classical(key, test, train, params):
    texts = test[C.TEXT_COL].tolist()
    y_test = test[C.LABEL_ID_COL].to_numpy()
    pipeline = cm.load_model(key)
    preds = pipeline.predict(texts)
    result = ev.evaluate_predictions(y_test, preds)

    infer_s = ev.median_seconds(lambda: pipeline.predict(texts), repeats=3)
    print(f"  inference: {infer_s / len(texts) * 1000:.3f} ms per review (CPU)")
    train_s = ev.classical_training_time(key, params, train[C.TEXT_COL], train[C.LABEL_ID_COL], repeats=3)
    print(f"  training (re-fit of the chosen settings): {train_s:.2f} s")

    path = cm.model_path(key)
    entry = {
        "display_name": C.CLASSICAL_MODEL_NAMES[key],
        "family": "classical",
        "status": "evaluated",
        **result,
        "settings": params,
        "cost": {
            "train_seconds": train_s,
            "train_note": "Re-fit of the chosen settings on the training split (TF-IDF + classifier), CPU, median of 3 runs.",
            "inference_ms_per_review": infer_s / len(texts) * 1000,
            "inference_device": "cpu",
            "model_size_mb": path.stat().st_size / 1e6,
            "n_parameters": ev.count_classical_parameters(pipeline),
        },
    }
    return entry, preds


def distilbert_training_cost():
    """Training minutes of the best DistilBERT run, from the notebook's own saved results."""
    summary = read_json(C.RESULTS_DIR / "distilbert_summary.json")
    if not summary:
        return None, "Training time was not recorded (results/distilbert_summary.json is missing).", None
    best = summary.get("best_settings", {})
    minutes, note = None, ""
    search_path = C.RESULTS_DIR / "distilbert_search.csv"
    if search_path.is_file():
        search = pd.read_csv(search_path)
        mask = pd.Series(True, index=search.index)
        for k, v in best.items():
            if k in search.columns:
                mask &= search[k].astype(str) == str(v)
        if mask.any():
            minutes = float(search[mask].iloc[0]["minutes"])
    total = summary.get("total_training_minutes")
    device = summary.get("device", "unknown device")
    if minutes is not None:
        note = (f"Best run only: {minutes:.1f} min on {device} (up to {summary.get('train_defaults', {}).get('epochs', '?')} epochs, "
                f"best epoch {summary.get('best_epoch')}). All runs together: {total} min.")
        return minutes * 60, note, summary
    if total is not None:
        return float(total) * 60, f"Total of all runs: {total} min on {device} (per-run time not found).", summary
    return None, "Training time was not recorded.", summary


def evaluate_distilbert(test, cpu_bench_n):
    import torch

    from src import transformer_model as tm

    texts = test[C.TEXT_COL].tolist()
    y_test = test[C.LABEL_ID_COL].to_numpy()
    device = tm.get_device()
    print(f"  loading DistilBERT on {device} ...")
    tokenizer, model = tm.load_model(device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    tm.predict_texts(texts[:64], tokenizer, model)  # warm-up
    sync()
    start = time.perf_counter()
    preds, probs = tm.predict_texts(texts, tokenizer, model)
    sync()
    device_s = time.perf_counter() - start
    result = ev.evaluate_predictions(y_test, preds)
    print(f"  inference on {device}: {device_s / len(texts) * 1000:.3f} ms per review")

    cpu_ms = None
    if cpu_bench_n > 0:
        model.to("cpu")
        n = min(cpu_bench_n, len(texts))
        tm.predict_texts(texts[:16], tokenizer, model)
        start = time.perf_counter()
        tm.predict_texts(texts[:n], tokenizer, model)
        cpu_ms = (time.perf_counter() - start) / n * 1000
        print(f"  inference on cpu ({n} reviews): {cpu_ms:.3f} ms per review")

    train_s, train_note, summary = distilbert_training_cost()
    entry = {
        "display_name": C.TRANSFORMER_DISPLAY_NAME,
        "family": "transformer",
        "status": "evaluated",
        **result,
        "settings": (summary or {}).get("best_settings", {}),
        "cost": {
            "train_seconds": train_s,
            "train_note": train_note,
            "inference_ms_per_review": device_s / len(texts) * 1000,
            "inference_device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
            "inference_ms_per_review_cpu": cpu_ms,
            "model_size_mb": ev_dir_size_mb(C.TRANSFORMER_DIR),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        },
    }
    return entry, preds, probs


def ev_dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-distilbert", action="store_true", help="evaluate only the classical models")
    ap.add_argument("--cpu-bench-n", type=int, default=200, help="reviews used to time DistilBERT on the CPU (0 = skip)")
    args = ap.parse_args()

    C.ensure_dirs()
    train, val, test = dp.load_splits()
    overlap = dp.check_no_overlap(train, val, test)
    if any(overlap.values()):
        raise SystemExit(f"Data leakage: the splits share reviews {overlap}. Fix the splits before evaluating.")
    print(f"Test set: {len(test)} reviews (no overlap with train/val: {overlap})")

    csum = read_json(C.RESULTS_DIR / "classical_summary.json") or {}
    best_params = csum.get("best_params", {})
    models, predictions = {}, {}

    for key in C.CLASSICAL_MODEL_NAMES:
        print(f"\n[{C.CLASSICAL_MODEL_NAMES[key]}]")
        try:
            entry, preds = evaluate_classical(key, test, train, best_params.get(key, {}))
        except FileNotFoundError as exc:
            models[key] = {"display_name": C.CLASSICAL_MODEL_NAMES[key], "family": "classical",
                           "status": "not_available", "reason": str(exc)}
            print(f"  unavailable: {exc}")
            continue
        models[key] = entry
        predictions[key] = preds
        m = entry["metrics"]
        print(f"  test accuracy {m['accuracy']:.4f} | macro F1 {m['macro_f1']:.4f}")

    print(f"\n[{C.TRANSFORMER_DISPLAY_NAME}]")
    if args.skip_distilbert:
        models["distilbert"] = {"display_name": C.TRANSFORMER_DISPLAY_NAME, "family": "transformer",
                                "status": "not_available", "reason": "Skipped with --skip-distilbert."}
        print("  skipped")
    elif not (C.TRANSFORMER_DIR / "config.json").is_file():
        reason = "No fine-tuned model found in models/distilbert/. Finish notebooks/03_distilbert.ipynb first."
        models["distilbert"] = {"display_name": C.TRANSFORMER_DISPLAY_NAME, "family": "transformer",
                                "status": "not_available", "reason": reason}
        print(f"  unavailable: {reason}")
    else:
        entry, preds, probs = evaluate_distilbert(test, args.cpu_bench_n)
        models["distilbert"] = entry
        predictions["distilbert"] = preds
        m = entry["metrics"]
        print(f"  test accuracy {m['accuracy']:.4f} | macro F1 {m['macro_f1']:.4f}")

    if not predictions:
        raise SystemExit("No model could be evaluated. Run notebooks 02 and 03 first.")

    doc = {
        "schema_version": 1,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": "test",
        "n_test": int(len(test)),
        "labels": list(C.LABEL_NAMES),
        "environment": environment(),
        "baseline": ev.majority_baseline(train[C.LABEL_ID_COL], test[C.LABEL_ID_COL]),
        "models": {k: models[k] for k in MODEL_KEYS if k in models},
    }

    out = C.RESULTS_DIR
    tmp = out / "test_results.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    tmp.replace(out / "test_results.json")

    (out / "test_results_summary.md").write_text(ev.summary_markdown(doc), encoding="utf-8")
    reports = [ev.classification_report_text(v["display_name"], v) for v in doc["models"].values() if v.get("status") == "evaluated"]
    (out / "test_classification_reports.txt").write_text("\n\n".join(reports) + "\n", encoding="utf-8")

    table = test[[C.TEXT_COL, C.RATING_COL, C.LABEL_COL]].copy()
    table.insert(0, "id", range(len(table)))
    for key, preds in predictions.items():
        table[f"pred_{key}"] = [C.ID2LABEL[int(p)] for p in preds]
    table.to_csv(out / "test_predictions.csv", index=False)

    figures = ev.save_figures(doc, C.FIGURES_DIR)

    print("\n" + ev.summary_markdown(doc))
    print("Saved:", ", ".join(["results/test_results.json", "results/test_results_summary.md",
                               "results/test_classification_reports.txt", "results/test_predictions.csv"]
                              + [f"results/figures/{f}" for f in figures]))
    print("EVALUATION FINISHED")


if __name__ == "__main__":
    main()
