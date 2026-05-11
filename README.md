# Cross-topic Argument Mining with RoBERTa

Re-implementation of Stab et al. (EMNLP 2018), *Cross-topic Argument Mining
from Heterogeneous Sources*, replacing the **Contextual BiLSTM (biclstm)**
with **RoBERTa**.

The paper integrates the topic vector into the i- and c-gates of an LSTM
cell so the encoder is topic-aware. In a transformer, the analogous design
is to feed the topic and the candidate sentence as a **sentence pair**
(`<s> topic </s></s> sentence </s>`). Topic information then conditions the
sentence representation via cross-token self-attention inside RoBERTa.

Everything else follows the paper:

- **Dataset**: UKP Sentential Argument Mining corpus (8 topics, 25,492
  sentences).
- **Setups**:
  - 2-label: `argument` vs `non-argument`
  - 3-label: `supporting argument` / `opposing argument` / `non-argument`
- **Protocol**: cross-topic — hold one topic out as test, train on the
  remaining 7; report mean ± std over 10 seeds.
- **Metrics**: macro-F1; precision/recall for the argument class(es)
  (`P_arg`, `R_arg`, or `P_arg+`, `P_arg-`, `R_arg+`, `R_arg-`).
- **Training**: AdamW, cross-entropy, 10 epochs, best model by validation
  loss (10% of train), max sequence length 128.

## Data

Download the UKP corpus
(https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345) and place the
per-topic TSV files under `data/ukp/`:

```
data/ukp/abortion.tsv
data/ukp/cloning.tsv
data/ukp/death_penalty.tsv
data/ukp/gun_control.tsv
data/ukp/marijuana_legalization.tsv
data/ukp/minimum_wage.tsv
data/ukp/nuclear_energy.tsv
data/ukp/school_uniforms.tsv
```

Each TSV is the official release with columns `topic`, `retrievedUrl`,
`archivedUrl`, `sentenceHash`, `sentence`, `annotation`, `set` (the `set`
column gives the official train/val/test split per topic).

## Install

```bash
pip install -r requirements.txt
```

## Run

Single held-out topic, one seed:

```bash
python -m src.train --test_topic "gun control" --labels 2 --seed 0
```

Full paper protocol (all 8 topics × 10 seeds, both 2- and 3-label):

```bash
python -m src.run_all
```

Results are written to `results/`.
