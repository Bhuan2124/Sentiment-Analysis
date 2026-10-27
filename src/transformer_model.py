"""
Phase 3: fine-tuning DistilBERT for 3-class sentiment.

Used by notebooks/03_distilbert.ipynb now, and by the evaluation notebook
(Phase 4) and the web app (Phase 5), which load the saved model from disk.

Same rules as Phase 2:
  * the model learns from the TRAINING split only
  * settings and the best epoch are chosen on the VALIDATION split only
  * the TEST split is never used in this file

A plain PyTorch training loop is used (instead of the Hugging Face `Trainer`)
so every step is visible and the code works the same on Colab and on a laptop.
"""

import copy
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config as C
from src.classical_models import compute_metrics


# ---------------------------------------------------------------------------
# 1. Reproducibility and device
# ---------------------------------------------------------------------------
def set_seed(seed: int = C.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU if there is one (Colab: Runtime > Change runtime type > GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_mixed_precision(device: torch.device) -> bool:
    """
    Half precision (float16) roughly halves memory and is much faster on GPUs with
    tensor cores (e.g. Colab's T4). GTX 16xx cards have none and run float16
    training ~3x SLOWER than float32 (measured on a GTX 1650), so skip it there.
    """
    if device.type != "cuda":
        return False
    name = torch.cuda.get_device_name(device)
    return torch.cuda.get_device_capability(device) >= (7, 0) and "GTX 16" not in name


# ---------------------------------------------------------------------------
# 2. Tokenizer, model, data
# ---------------------------------------------------------------------------
def load_tokenizer(name_or_dir=C.TRANSFORMER_NAME):
    return AutoTokenizer.from_pretrained(str(name_or_dir))


def load_pretrained_model(name: str = C.TRANSFORMER_NAME):
    """The pre-trained DistilBERT with a NEW, untrained 3-class output layer on top."""
    return AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=len(C.LABEL_NAMES),
        id2label=C.ID2LABEL,
        label2id=C.LABEL2ID,
    )


def token_lengths(tokenizer, texts: Sequence[str]) -> np.ndarray:
    """Number of tokens per text (including the special [CLS] and [SEP] tokens), without cutting."""
    return np.array([len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]])


class ReviewDataset(Dataset):
    """
    Tokenises every review once, up front. Padding is NOT done here: each batch is
    padded only to its own longest review (see `collate`), which is much faster
    than padding everything to MAX_LENGTH.
    """

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int = C.MAX_LENGTH):
        self.input_ids = tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.labels[i]


def make_collate(pad_token_id: int) -> Callable:
    def collate(batch):
        longest = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), longest), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for row, (ids, _) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1   # 1 = real token, 0 = padding
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def make_loader(dataset: ReviewDataset, tokenizer, batch_size: int, shuffle: bool, seed: int = C.RANDOM_SEED):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )


# ---------------------------------------------------------------------------
# 3. Training settings
# ---------------------------------------------------------------------------
# Standard fine-tuning values from the BERT/DistilBERT papers. Only the ones in
# SEARCH_GRID are tuned; the rest stay fixed.
TRAIN_DEFAULTS = {
    "learning_rate": 3e-5,
    "class_weighted": False,  # True = mistakes on rare classes cost more (like class_weight="balanced")
    "epochs": 4,              # maximum; the best epoch on validation is kept
    "batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,      # first 10% of steps: learning rate rises from 0 (stabilises the start)
    "patience": 2,            # stop early if validation macro-F1 has not improved for this many epochs
    "max_length": C.MAX_LENGTH,
}
SEARCH_GRID = {
    "learning_rate": [2e-5, 5e-5],
    "class_weighted": [False, True],
}


def balanced_class_weights(y: Sequence[int]) -> torch.Tensor:
    """weight_c = n_samples / (n_classes * n_samples_in_c), same formula as scikit-learn."""
    counts = np.bincount(np.asarray(y), minlength=len(C.LABEL_NAMES))
    return torch.tensor(len(y) / (len(counts) * counts), dtype=torch.float)


def linear_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Learning rate goes 0 -> max during warm-up, then down linearly to 0."""
    def factor(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------------------
# 4. Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    use_amp = use_mixed_precision(device)
    out = []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 5. Training one configuration
# ---------------------------------------------------------------------------
def train_one(
    train_ds: ReviewDataset,
    val_ds: ReviewDataset,
    tokenizer,
    settings: Dict,
    device: torch.device = None,
    log: Callable = print,
) -> Dict:
    """
    Fine-tune a fresh DistilBERT with `settings` (missing keys come from TRAIN_DEFAULTS).
    After every epoch the model is scored on validation; the weights of the BEST
    epoch are kept.

    Returns {"settings", "history" (one row per epoch), "best_epoch",
             "best_metrics", "best_state" (weights, on CPU), "minutes"}.
    """
    s = {**TRAIN_DEFAULTS, **settings}
    device = device or get_device()
    set_seed()
    start = time.time()

    model = load_pretrained_model().to(device)
    train_loader = make_loader(train_ds, tokenizer, s["batch_size"], shuffle=True)
    val_loader = make_loader(val_ds, tokenizer, s["batch_size"] * 2, shuffle=False)
    y_val = np.array(val_ds.labels)

    weights = balanced_class_weights(train_ds.labels).to(device) if s["class_weighted"] else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    # No weight decay on biases and LayerNorm weights (standard for BERT-style models)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": s["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=s["learning_rate"])
    total_steps = len(train_loader) * s["epochs"]
    scheduler = linear_warmup_schedule(optimizer, int(s["warmup_ratio"] * total_steps), total_steps)

    use_amp = use_mixed_precision(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best = [], {"score": -1.0}
    for epoch in range(1, s["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits.float(), batch["labels"])

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        val_logits = predict_logits(model, val_loader, device)
        metrics = compute_metrics(y_val, val_logits.argmax(axis=1))
        row = {"epoch": epoch, "train_loss": running_loss / len(train_loader), **metrics}
        history.append(row)
        log(
            f"  epoch {epoch}: train loss {row['train_loss']:.4f} | val accuracy {metrics['accuracy']:.3f}"
            f" | val macro-F1 {metrics['macro_f1']:.3f} | {(time.time() - start) / 60:.1f} min"
        )

        if metrics[C.SELECTION_METRIC] > best["score"]:
            best = {
                "score": metrics[C.SELECTION_METRIC],
                "epoch": epoch,
                "metrics": metrics,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= s["patience"] and epoch < s["epochs"]:
            log(f"  stopping early: no improvement for {s['patience']} epochs")
            break

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "settings": s,
        "history": pd.DataFrame(history),
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "best_state": best["state"],
        "minutes": (time.time() - start) / 60,
    }


def search(
    train_ds: ReviewDataset,
    val_ds: ReviewDataset,
    tokenizer,
    grid: Dict[str, list] = None,
    device: torch.device = None,
    log: Callable = print,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run `train_one` for every combination in `grid` (default SEARCH_GRID).
    Returns (results_table, per_epoch_history, best_run).
    Only the best run's weights are kept in memory.
    """
    from src.classical_models import param_combinations

    grid = SEARCH_GRID if grid is None else grid
    combos = param_combinations(grid)
    rows, histories, best_run = [], [], None
    for i, settings in enumerate(combos, 1):
        log(f"Run {i}/{len(combos)}: {settings}")
        run = train_one(train_ds, val_ds, tokenizer, settings, device=device, log=log)
        rows.append({"run": i, **{k: str(v) for k, v in settings.items()},
                     "best_epoch": run["best_epoch"], "minutes": round(run["minutes"], 1),
                     **run["best_metrics"]})
        histories.append(run["history"].assign(run=i, **{k: str(v) for k, v in settings.items()}))
        if best_run is None or run["best_metrics"][C.SELECTION_METRIC] > best_run["best_metrics"][C.SELECTION_METRIC]:
            best_run = run
        else:
            run.pop("best_state")   # free memory: only the best weights are needed

    table = pd.DataFrame(rows).sort_values(C.SELECTION_METRIC, ascending=False).reset_index(drop=True)
    return table, pd.concat(histories, ignore_index=True), best_run


# ---------------------------------------------------------------------------
# 6. Saving and loading
# ---------------------------------------------------------------------------
def save_model(state: Dict, tokenizer, out_dir: Path = C.TRANSFORMER_DIR) -> Path:
    """Save the fine-tuned weights + tokenizer in the standard Hugging Face format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_pretrained_model()
    model.load_state_dict(state)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def load_model(model_dir: Path = C.TRANSFORMER_DIR, device: torch.device = None):
    """Load the fine-tuned model. Phase 4 and the web app start with this."""
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"{model_dir} has no saved model. Run notebooks/03_distilbert.ipynb first.")
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
    return tokenizer, model


def predict_texts(
    texts: Sequence[str], tokenizer, model, batch_size: int = 64, max_length: int = C.MAX_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw review texts -> (predicted label ids, class probabilities)."""
    device = next(model.parameters()).device
    ds = ReviewDataset(texts, [0] * len(texts), tokenizer, max_length)
    probs = softmax(predict_logits(model, make_loader(ds, tokenizer, batch_size, shuffle=False), device))
    return probs.argmax(axis=1), probs
