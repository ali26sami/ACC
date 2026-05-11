# Cross-topic Argument Mining with RoBERTa

Re-implementation of Stab et al. (EMNLP 2018), *Cross-topic Argument Mining
from Heterogeneous Sources*, with **RoBERTa replacing the Contextual BiLSTM
(BiCLSTM)**.

The paper integrates the topic vector into the i- and c-gates of an LSTM
cell to make the encoder topic-aware. The RoBERTa analogue is a sentence-pair
input — `<s> topic </s></s> sentence </s>` — so cross-token self-attention
inside the transformer conditions sentence tokens on topic tokens.

Two model variants are provided, matching the paper:

| Variant     | Paper model                 | Description |
|-------------|-----------------------------|-------------|
| single-task | `biclstm`                   | RoBERTa fine-tuned on UKP only. |
| MTL         | `mtl+biclstm+dip2016`       | Shared RoBERTa encoder + private head for UKP + private head for DIP2016 relevance, alternating-epoch training. |

Both 2-label (argument / non-argument) and 3-label (supporting / opposing
/ non-argument) setups, cross-topic protocol (hold one topic out for test),
mean ± std over multiple seeds, macro-F1 + P_arg / R_arg metrics — same
as Table 4 of the paper.

## Running on Google Colab

Open `notebooks/roberta_argmining_colab.ipynb`. It contains:

1. Drive mount
2. Dependency install
3. Repo clone
4. **A single config cell** with every path, hyperparameter, and seed
5. Data sanity check
6. Smoke-test run
7. Full grid (resumable — cached JSONs are skipped)
8. Summary inspection

Set these in the config cell to point at your Drive files:

```python
UKP_CSV    = DRIVE_ROOT / 'UKP' / 'ukp_sentential_argument_mining.csv'
DIP_DIR    = DRIVE_ROOT / 'DIP2016'        # 50 XML files
OUTPUT_DIR = DRIVE_ROOT / 'roberta_argmining_runs'
```

Per-run JSONs and `summary.json` are written to `OUTPUT_DIR` on Drive.

## Data formats

**UKP** — single CSV with columns:
`topic, retrievedUrl, archivedUrl, sentenceHash, sentence, annotation, set`
where `annotation ∈ {NoArgument, Argument_for, Argument_against}` and
`set ∈ {train, val, test}`.

**DIP2016** — folder of per-query XML files of shape:
```xml
<singleQueryResults queryID="...">
  <documents>
    <document clueWebID="...">
      <sentences>
        <s relevant="true|false"><content>...</content></s>
        ...
```
The XMLs only carry `queryID`, not the query text. Supply an optional
`queryID,query_text` CSV to use real query strings in the MTL head; otherwise
the queryID is used as the topic text segment.

## Local CLI (optional)

```bash
pip install -r requirements.txt

# Single run
python -m src.train --ukp_csv data/ukp.csv --test_topic "gun control" --labels 2 --seed 0
# Same, with MTL+DIP2016
python -m src.train --ukp_csv data/ukp.csv --dip_dir data/dip2016 --test_topic "gun control" --labels 2 --seed 0 --mtl

# Full grid
python -m src.run_all --ukp_csv data/ukp.csv --dip_dir data/dip2016 --output_dir results
```
