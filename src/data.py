"""UKP Sentential Argument Mining + DIP2016 loaders.

UKP: a single CSV with columns
    topic, retrievedUrl, archivedUrl, sentenceHash, sentence, annotation, set
where annotation in {NoArgument, Argument_for, Argument_against}
and set in {train, val, test}.

DIP2016: a folder of XML files, one per query, with this shape:
    <singleQueryResults queryID="...">
      <documents>
        <document clueWebID="...">
          <sentences>
            <s relevant="true|false"><content>...</content></s>
            ...
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
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


def label_to_id(annotation: str, num_labels: int) -> int:
    if num_labels == 3:
        return LABELS_3.index(annotation)
    return 0 if annotation == "NoArgument" else 1


@dataclass
class Example:
    topic: str
    sentence: str
    label: int


# --------------------------------------------------------------------------
# UKP
# --------------------------------------------------------------------------

def load_ukp_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"topic", "sentence", "annotation", "set"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"UKP CSV {csv_path} missing columns: {missing}")
    # normalize topic strings to lowercase for matching
    df["topic_norm"] = df["topic"].str.strip().str.lower()
    return df


def build_ukp_splits(
    csv_path: Path,
    test_topic: str,
    num_labels: int,
) -> dict[str, list[Example]]:
    """Cross-topic split: train+val from 7 topics, test from the held-out topic."""
    df = load_ukp_csv(csv_path)
    test_topic_n = test_topic.strip().lower()
    if test_topic_n not in df["topic_norm"].unique():
        raise ValueError(
            f"Topic {test_topic!r} not found. Available: "
            f"{sorted(df['topic_norm'].unique())}"
        )

    train, val, test = [], [], []
    for _, row in df.iterrows():
        ex = Example(
            topic=str(row["topic"]),
            sentence=str(row["sentence"]),
            label=label_to_id(str(row["annotation"]), num_labels),
        )
        if row["topic_norm"] == test_topic_n:
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


# --------------------------------------------------------------------------
# DIP2016
# --------------------------------------------------------------------------

@dataclass
class DIPExample:
    query: str
    sentence: str
    label: int  # 1 = relevant, 0 = not relevant


def _parse_dip_xml(path: Path, query_text_map: dict[str, str] | None = None) -> list[DIPExample]:
    tree = ET.parse(path)
    root = tree.getroot()
    query_id = root.attrib.get("queryID", path.stem)
    query_text = (
        query_text_map.get(str(query_id))
        if query_text_map is not None
        else None
    ) or str(query_id)

    out: list[DIPExample] = []
    for s in root.iter("s"):
        rel_attr = s.attrib.get("relevant", "false").strip().lower()
        label = 1 if rel_attr == "true" else 0
        content = s.find("content")
        if content is None or content.text is None:
            continue
        text = content.text.strip()
        if not text:
            continue
        out.append(DIPExample(query=query_text, sentence=text, label=label))
    return out


def load_dip2016(
    dip_dir: Path,
    query_text_map: dict[str, str] | None = None,
    limit_files: int | None = None,
    max_examples: int | None = None,
) -> list[DIPExample]:
    """Parse every XML in dip_dir into (query, sentence, relevance) examples.

    Parameters
    ----------
    query_text_map : dict[str, str] | None
        Optional mapping from queryID -> human-readable query text. If None,
        the queryID itself is used as the text segment (this still works for
        MTL since the shared encoder learns from sentence content; supplying
        real query strings stays closer to the paper).
    limit_files : int | None
        For debugging, only parse the first N files.
    max_examples : int | None
        For the paper's "use 300K of 600K samples" hyperparameter-tuning
        regime: random subsample to this many examples (no, deterministic
        prefix here; pass via a separate shuffle if you want randomness).
    """
    files = sorted(dip_dir.glob("*.xml"))
    if limit_files is not None:
        files = files[:limit_files]
    out: list[DIPExample] = []
    for f in files:
        out.extend(_parse_dip_xml(f, query_text_map))
        if max_examples is not None and len(out) >= max_examples:
            return out[:max_examples]
    return out


class DIPDataset(Dataset):
    """Encodes (query, sentence) as a RoBERTa sentence pair for binary
    relevance classification."""

    def __init__(
        self,
        examples: Iterable[DIPExample],
        tokenizer,
        max_length: int = 128,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.query,
            ex.sentence,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {k: torch.as_tensor(v) for k, v in enc.items()}
        item["labels"] = torch.as_tensor(ex.label, dtype=torch.long)
        return item


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------

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
