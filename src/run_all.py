"""Run the full paper protocol: all 8 held-out topics x 10 seeds, both setups.

Writes one JSON per (setup, topic, seed) under results/ and an aggregated
summary table at the end.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import TOPICS
from .metrics import aggregate_runs
from .train import run_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data/ukp"))
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--labels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, dict]] = {}

    for num_labels in args.labels:
        per_topic: dict[str, list[dict]] = {}
        for topic in TOPICS:
            runs: list[dict] = []
            for seed in args.seeds:
                tag = f"L{num_labels}_{topic.replace(' ', '_')}_seed{seed}"
                out_path = args.results_dir / f"{tag}.json"
                if out_path.exists():
                    runs.append(json.loads(out_path.read_text()))
                    continue
                res = run_one(
                    data_dir=args.data_dir,
                    test_topic=topic,
                    num_labels=num_labels,
                    model_name=args.model,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    max_length=args.max_length,
                )
                out_path.write_text(json.dumps(res, indent=2))
                runs.append(res)
            per_topic[topic] = runs
        summary[f"{num_labels}-label"] = {
            topic: aggregate_runs(runs) for topic, runs in per_topic.items()
        }
        # overall (average across topics, same as the paper's Table 4)
        all_runs = [r for runs in per_topic.values() for r in runs]
        summary[f"{num_labels}-label"]["__overall__"] = aggregate_runs(all_runs)

    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
