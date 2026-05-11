"""UKP Sentential Argument Mining corpus loader.

The official release ships one TSV per topic with the columns:
    topic, retrievedUrl, archivedUrl, sentenceHash, sentence, annotation, set
where `annotation` is one of {NoArgument, Argument_for, Argument_against}
and `set` is one of {train, val, test} (the paper's 70/10/20 split).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

TOPICS = [
    "abortion",
    "cloning",
    "death penalty",
    "gun control",
    "marijuana legalization",
    "minimum wage",
    "nuclear energy",
    "school uniforms",
]

LABELS_3 = ["NoArgument", "Argument_against", "Argument_for"]
LABELS_2 = ["NoArgument", "Argument"]


def _topic_to_filename(topic: str) -> str:
    return topic.replace(" ", "_").lower() + ".tsv"


def label_to_id(annotation: str, num_labels: int) -> int:
    if num_labels == 3:
        return LABELS_3.index(annotation)
    # 2-label: collapse for/against -> Argument
    return 0 if annotation == "NoArgument" else 1


@dataclass
class Example:
    topic: str
    sentence: str
    label: int


def load_topic_tsv(data_dir: Path, topic: str) -> pd.DataFrame:
    path = data_dir / _topic_to_filename(topic)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing UKP file for topic '{topic}': {path}. "
            "See README for download instructions."
        )
    df = pd.read_csv(path, sep="\t", quoting=3, encoding="utf-8")
    expected = {"topic", "sentence", "annotation", "set"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def build_splits(
    data_dir: Path,
    test_topic: str,
    num_labels: int,
) -> dict[str, list[Example]]:
    """Cross-topic split: train/val from 7 topics, test from the held-out topic.

    Uses each topic's official `train`/`val` rows for training topics and the
    held-out topic's `test` rows for test.
    """
    if test_topic not in TOPICS:
        raise ValueError(f"Unknown topic: {test_topic!r}")

    train, val, test = [], [], []
    for topic in TOPICS:
        df = load_topic_tsv(data_dir, topic)
        for _, row in df.iterrows():
            ex = Example(
                topic=row["topic"],
                sentence=str(row["sentence"]),
                label=label_to_id(row["annotation"], num_labels),
            )
            if topic == test_topic:
                if row["set"] == "test":
                    test.append(ex)
            else:
                if row["set"] == "train":
                    train.append(ex)
                elif row["set"] == "val":
                    val.append(ex)
    return {"train": train, "val": val, "test": test}


class UKPDataset(Dataset):
    """Encodes (topic, sentence) as a RoBERTa sentence pair."""

    def __init__(self, examples: Iterable[Example], tokenizer, max_length: int = 128):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.topic,
            ex.sentence,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {k: torch.as_tensor(v) for k, v in enc.items()}
        item["labels"] = torch.as_tensor(ex.label, dtype=torch.long)
        return item


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(x["input_ids"].size(0) for x in batch)
    out = {}
    for key in ("input_ids", "attention_mask"):
        if key not in batch[0]:
            continue
        pad_val = pad_token_id if key == "input_ids" else 0
        stacked = torch.full((len(batch), max_len), pad_val, dtype=torch.long)
        for i, x in enumerate(batch):
            stacked[i, : x[key].size(0)] = x[key]
        out[key] = stacked
    out["labels"] = torch.stack([x["labels"] for x in batch])
    return out
