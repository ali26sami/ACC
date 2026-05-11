"""Single-run training and evaluation for one held-out topic."""
from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import TOPICS, UKPDataset, build_splits, collate
from .metrics import compute_metrics
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, preds, golds = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item() * batch["labels"].size(0))
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
        golds.append(batch["labels"].cpu().numpy())
    n = sum(len(g) for g in golds)
    return (
        sum(losses) / max(n, 1),
        np.concatenate(preds),
        np.concatenate(golds),
    )


def run_one(
    data_dir: Path,
    test_topic: str,
    num_labels: int,
    model_name: str = "roberta-base",
    seed: int = 0,
    epochs: int = 10,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_length: int = 128,
    device: str | None = None,
) -> dict:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    splits = build_splits(data_dir, test_topic, num_labels)
    model, tokenizer = build_model(model_name, num_labels)
    model.to(device)

    train_ds = UKPDataset(splits["train"], tokenizer, max_length=max_length)
    val_ds = UKPDataset(splits["val"], tokenizer, max_length=max_length)
    test_ds = UKPDataset(splits["test"], tokenizer, max_length=max_length)

    coll = partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, collate_fn=coll)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = AdamW(params, lr=lr)
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * warmup_ratio), total_steps
    )

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            pbar.set_postfix(loss=float(out.loss.item()))

        val_loss, _, _ = evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    _, preds, golds = evaluate(model, test_loader, device)
    metrics = compute_metrics(golds, preds, num_labels)
    metrics["best_val_loss"] = best_val
    metrics["test_topic"] = test_topic
    metrics["seed"] = seed
    metrics["num_labels"] = num_labels
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data/ukp"))
    ap.add_argument("--test_topic", required=True, choices=TOPICS)
    ap.add_argument("--labels", type=int, choices=(2, 3), default=2)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    res = run_one(
        data_dir=args.data_dir,
        test_topic=args.test_topic,
        num_labels=args.labels,
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )
    print(json.dumps(res, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
