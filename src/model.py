"""RoBERTa sentence-pair classifier — drop-in replacement for the BiCLSTM.

The paper's BiCLSTM fuses the topic vector into LSTM gates; here the topic is
the first segment of a RoBERTa sentence-pair input, so cross-token attention
inside the transformer plays the same conditioning role.
"""
from __future__ import annotations

from transformers import RobertaForSequenceClassification, RobertaTokenizerFast


def build_model(model_name: str, num_labels: int):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tokenizer
