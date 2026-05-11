"""Metrics matching Table 4 of Stab et al. (2018).

- Macro-F1 over the label set.
- 2-label: P/R for the `Argument` class (P_arg, R_arg).
- 3-label: P/R for `Argument_for` (P_arg+, R_arg+)
           and `Argument_against` (P_arg-, R_arg-).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .data import LABELS_2, LABELS_3


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> dict:
    out: dict[str, float] = {}
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if num_labels == 2:
        p, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], zero_division=0
        )
        out["P_arg"] = float(p[0])
        out["R_arg"] = float(r[0])
    else:
        # Index per LABELS_3 = ["NoArgument", "Argument_against", "Argument_for"]
        labels = [LABELS_3.index("Argument_for"), LABELS_3.index("Argument_against")]
        p, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        out["P_arg+"] = float(p[0])
        out["R_arg+"] = float(r[0])
        out["P_arg-"] = float(p[1])
        out["R_arg-"] = float(r[1])
    return out


def aggregate_runs(runs: list[dict]) -> dict:
    """Average metric dicts across runs and report mean ± std."""
    keys = runs[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        agg[k] = (float(vals.mean()), float(vals.std()))
    return agg
