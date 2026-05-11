"""Training + evaluation for one held-out topic.

Supports two modes:
  - single-task: RoBERTa on UKP only (replaces biclstm).
  - mtl: RoBERTa shared encoder + UKP head + DIP relevance head, alternating
    epochs (replaces mtl+biclstm+dip2016).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import (
    DIPDataset,
    TOPICS,
    UKPDataset,
    build_ukp_splits,
    collate,
    load_dip2016,
)
from .metrics import compute_metrics
from .model import build_mtl_model, build_single_task_model


# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------

@dataclass
class HParams:
    # Paths
    ukp_csv: Path = Path("data/ukp.csv")
    dip_dir: Path = Path("data/dip2016")
    output_dir: Path = Path("results")
    # Model
    model_name: str = "roberta-base"
    max_length: int = 128
    # Optim
    epochs: int = 10
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    # Setup
    num_labels: int = 2          # 2 or 3
    test_topic: str = "gun control"
    seed: int = 0
    # MTL
    use_mtl: bool = False
    dip_max_examples: int | None = 300_000  # paper uses 300K of 600K
    dip_query_text_map: dict[str, str] = field(default_factory=dict)
    # Misc
    device: str | None = None
    num_workers: int = 2


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_optim(model: torch.nn.Module, hp: HParams, total_steps: int):
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": hp.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = AdamW(params, lr=hp.lr)
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * hp.warmup_ratio), total_steps
    )
    return optim, sched


@torch.no_grad()
def _evaluate(model, loader, device, mtl_task: str | None = None):
    model.eval()
    losses, preds, golds = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if mtl_task is None:
            out = model(**batch)
            loss = out.loss
            logits = out.logits
        else:
            out = model(**batch, task=mtl_task)
            loss = out["loss"]
            logits = out["logits"]
        losses.append(loss.item() * batch["labels"].size(0))
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        golds.append(batch["labels"].cpu().numpy())
    n = sum(len(g) for g in golds)
    return (
        sum(losses) / max(n, 1),
        np.concatenate(preds),
        np.concatenate(golds),
    )


# --------------------------------------------------------------------------
# Single-task training
# --------------------------------------------------------------------------

def _train_single(hp: HParams, device: str) -> dict:
    splits = build_ukp_splits(hp.ukp_csv, hp.test_topic, hp.num_labels)
    model, tokenizer = build_single_task_model(hp.model_name, hp.num_labels)
    model.to(device)

    train_ds = UKPDataset(splits["train"], tokenizer, hp.max_length)
    val_ds = UKPDataset(splits["val"], tokenizer, hp.max_length)
    test_ds = UKPDataset(splits["test"], tokenizer, hp.max_length)

    coll = partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=hp.batch_size, shuffle=True,
        collate_fn=coll, num_workers=hp.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=hp.eval_batch_size, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=hp.eval_batch_size, collate_fn=coll)

    optim, sched = _make_optim(model, hp, len(train_loader) * hp.epochs)

    best_val, best_state = float("inf"), None
    for epoch in range(hp.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{hp.epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optim.step(); sched.step(); optim.zero_grad()
            pbar.set_postfix(loss=float(out.loss.item()))
        val_loss, _, _ = _evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    _, preds, golds = _evaluate(model, test_loader, device)
    return compute_metrics(golds, preds, hp.num_labels) | {"best_val_loss": best_val}


# --------------------------------------------------------------------------
# MTL training (alternating epochs)
# --------------------------------------------------------------------------

def _train_mtl(hp: HParams, device: str) -> dict:
    splits = build_ukp_splits(hp.ukp_csv, hp.test_topic, hp.num_labels)
    model, tokenizer = build_mtl_model(hp.model_name, hp.num_labels)
    model.to(device)

    # Main task loaders
    train_ds = UKPDataset(splits["train"], tokenizer, hp.max_length)
    val_ds = UKPDataset(splits["val"], tokenizer, hp.max_length)
    test_ds = UKPDataset(splits["test"], tokenizer, hp.max_length)
    coll = partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=hp.batch_size, shuffle=True,
        collate_fn=coll, num_workers=hp.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=hp.eval_batch_size, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=hp.eval_batch_size, collate_fn=coll)

    # Aux task loader (DIP2016)
    dip_examples = load_dip2016(
        hp.dip_dir,
        query_text_map=hp.dip_query_text_map or None,
        max_examples=hp.dip_max_examples,
    )
    aux_ds = DIPDataset(dip_examples, tokenizer, hp.max_length)
    aux_loader = DataLoader(
        aux_ds, batch_size=hp.batch_size, shuffle=True,
        collate_fn=coll, num_workers=hp.num_workers,
    )

    total_steps = (len(train_loader) + len(aux_loader)) * hp.epochs
    optim, sched = _make_optim(model, hp, total_steps)

    def _run_epoch(loader, task: str, tag: str):
        model.train()
        pbar = tqdm(loader, desc=tag, leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, task=task)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optim.step(); sched.step(); optim.zero_grad()
            pbar.set_postfix(loss=float(out["loss"].item()))

    best_val, best_state = float("inf"), None
    for epoch in range(hp.epochs):
        # Alternate: aux first, then main (paper notes "the last epoch is
        # always executed on our dataset")
        _run_epoch(aux_loader, "aux", f"epoch {epoch + 1}/{hp.epochs} [aux]")
        _run_epoch(train_loader, "main", f"epoch {epoch + 1}/{hp.epochs} [main]")
        val_loss, _, _ = _evaluate(model, val_loader, device, mtl_task="main")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    _, preds, golds = _evaluate(model, test_loader, device, mtl_task="main")
    return compute_metrics(golds, preds, hp.num_labels) | {"best_val_loss": best_val}


# --------------------------------------------------------------------------
# Entrypoints
# --------------------------------------------------------------------------

def run_one(hp: HParams) -> dict:
    set_seed(hp.seed)
    device = hp.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fn = _train_mtl if hp.use_mtl else _train_single
    metrics = fn(hp, device)
    metrics.update({
        "test_topic": hp.test_topic,
        "seed": hp.seed,
        "num_labels": hp.num_labels,
        "use_mtl": hp.use_mtl,
        "model_name": hp.model_name,
    })
    return metrics


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ukp_csv", type=Path, required=True)
    ap.add_argument("--dip_dir", type=Path, default=Path("data/dip2016"))
    ap.add_argument("--output_dir", type=Path, default=Path("results"))
    ap.add_argument("--test_topic", required=True, choices=TOPICS)
    ap.add_argument("--labels", type=int, choices=(2, 3), default=2)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--mtl", action="store_true", help="Enable MTL with DIP2016")
    args = ap.parse_args()

    hp = HParams(
        ukp_csv=args.ukp_csv,
        dip_dir=args.dip_dir,
        output_dir=args.output_dir,
        test_topic=args.test_topic,
        num_labels=args.labels,
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        use_mtl=args.mtl,
    )
    res = run_one(hp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{'mtl' if hp.use_mtl else 'single'}_L{hp.num_labels}_{hp.test_topic.replace(' ', '_')}_seed{hp.seed}.json"
    (args.output_dir / tag).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _cli()
