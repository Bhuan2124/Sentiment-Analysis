"""
Quick end-to-end check that does not need the browser. Writes results/smoke_check.txt.

    python scripts/smoke_check.py

Checks: files that exist, label consistency and leakage in the saved splits, and a real prediction from every
saved model (the same code the Sentiment Analyzer page uses). For DistilBERT it also checks that the dashboard's
predictions agree with the training code's own prediction function.
"""
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import data_preprocessing as dp  # noqa: E402
from webapp import artifacts, inference  # noqa: E402

SAMPLES = [
    "Lovely dress, fits perfectly and the fabric is soft. I love it!",
    "The fabric feels cheap and it fell apart after one wash. I am returning it.",
    "It is okay. Nice colour but it runs small and the material is a bit thin.",
    "Not good at all.",
]
lines, failures = [], []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


def fail(text):
    failures.append(text)
    say("FAIL: " + text)


def main() -> int:
    say(f"Smoke check {time.strftime('%Y-%m-%d %H:%M:%S')} | Python {sys.version.split()[0]}")

    say("\n[1] Project files")
    inv = artifacts.inventory()
    for row in inv.itertuples():
        say(f"  {'found  ' if row.found == 'yes' else 'MISSING'} {row.file}")

    say("\n[2] Splits: labels and leakage")
    try:
        train, val, test = dp.load_splits()
        for name, df in (("train", train), ("val", val), ("test", test)):
            problems = artifacts.validate_split(df)
            say(f"  {name}: {len(df)} reviews, " + ("labels consistent" if not problems else "PROBLEMS " + "; ".join(problems)))
            if problems:
                fail(f"{name} split: {problems}")
        overlap = dp.check_no_overlap(train, val, test)
        say(f"  overlap between splits: {overlap}")
        if any(overlap.values()):
            fail("splits overlap (data leakage)")
    except Exception as exc:
        fail(f"could not load splits: {exc}")

    say("\n[3] Predictions from every saved model")
    loaded = {}
    for key in inference.MODEL_ORDER:
        ok, reason = inference.availability(key)
        name = inference.MODEL_INFOS[key].name
        if not ok:
            say(f"  {name}: not available ({reason})")
            continue
        try:
            model = inference.load_model(key)
            loaded[key] = model
            say(f"  {name}: loaded")
            for text in SAMPLES:
                p = inference.predict_text(model, text)
                score = f"{p.confidence:.3f}" if p.confidence is not None else f"score {max(p.scores.values()):+.2f}"
                say(f"    {p.label:<9} {score:>11}  {text[:60]}")
            for bad in ("", "   ", "12345 !!!"):
                try:
                    inference.predict_text(model, bad)
                    fail(f"{name} accepted invalid input {bad!r}")
                except inference.InvalidInputError:
                    pass
            say("    invalid input rejected: OK")
        except Exception as exc:
            fail(f"{name}: {type(exc).__name__}: {exc}")
            say(traceback.format_exc())

    if "distilbert" in loaded:
        say("\n[4] DistilBERT: dashboard code agrees with the training code")
        try:
            from src import transformer_model as tm

            tok, mdl = tm.load_model(device=tm.get_device())
            ref_pred, _ = tm.predict_texts(SAMPLES, tok, mdl)
            app_pred = [inference.predict_text(loaded["distilbert"], t).label_id for t in SAMPLES]
            same = list(map(int, ref_pred)) == app_pred
            say(f"  training-code labels {list(map(int, ref_pred))} | dashboard labels {app_pred} | agree: {same}")
            if not same:
                fail("dashboard and training code disagree for DistilBERT")
        except Exception as exc:
            fail(f"DistilBERT agreement check: {type(exc).__name__}: {exc}")

    say("\nRESULT: " + ("SMOKE CHECK PASSED" if not failures else f"SMOKE CHECK FAILED ({len(failures)} problems)"))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "smoke_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
