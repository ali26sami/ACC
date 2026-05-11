"""RoBERTa classifiers.

- `build_single_task_model`: plain RobertaForSequenceClassification, replaces
  the paper's `biclstm` (single-task).
- `RobertaMTL`: shared RoBERTa encoder + two private classification heads
  (main UKP task + DIP relevance aux task), replaces `mtl+biclstm+dip2016`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaTokenizerFast,
)


def build_single_task_model(model_name: str, num_labels: int):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tokenizer


class RobertaMTL(nn.Module):
    """Shared RoBERTa encoder with two private classification heads.

    Forward pass requires `task` in {"main", "aux"}.
    """

    def __init__(
        self,
        model_name: str,
        main_num_labels: int,
        aux_num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.main_head = nn.Linear(hidden, main_num_labels)
        self.aux_head = nn.Linear(hidden, aux_num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        task: str = "main",
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Pool the <s> token (position 0), as RoBERTa does in its
        # sequence classification head.
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        head = self.main_head if task == "main" else self.aux_head
        logits = head(pooled)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}


def build_mtl_model(model_name: str, main_num_labels: int):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaMTL(model_name, main_num_labels=main_num_labels)
    return model, tokenizer
