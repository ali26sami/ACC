"""Full paper protocol: all 8 held-out topics x N seeds x both setups,
optionally with the MTL+DIP2016 variant.

Skips runs whose output JSON already exists, so it's safe to resume.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .data import TOPICS
from .metrics import aggregate_runs
from .train import HParams, run_one


def _run_grid(base_hp: HParams, label_sets, seeds, mtl_modes, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    for use_mtl in mtl_modes:
        for num_labels in label_sets:
            per_topic: dict[str, list[dict]] = {}
            for topic in TOPICS:
                runs: list[dict] = []
                for seed in seeds:
                    mtl_tag = "mtl" if use_mtl else "single"
                    tag = f"{mtl_tag}_L{num_labels}_{topic.replace(' ', '_')}_seed{seed}"
                    out_path = output_dir / f"{tag}.json"
                    if out_path.exists():
                        runs.append(json.loads(out_path.read_text()))
                        continue
                    hp = replace(
                        base_hp,
                        test_topic=topic,
                        num_labels=num_labels,
                        seed=seed,
                        use_mtl=use_mtl,
                        output_dir=output_dir,
                    )
                    res = run_one(hp)
                    out_path.write_text(json.dumps(res, indent=2))
                    runs.append(res)
                per_topic[topic] = runs
            key = f"{'mtl' if use_mtl else 'single'}_{num_labels}label"
            summary[key] = {t: aggregate_runs(r) for t, r in per_topic.items()}
            all_runs = [r for rs in per_topic.values() for r in rs]
            summary[key]["__overall__"] = aggregate_runs(all_runs)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ukp_csv", type=Path, required=True)
    ap.add_argument("--dip_dir", type=Path, default=Path("data/dip2016"))
    ap.add_argument("--output_dir", type=Path, default=Path("results"))
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--labels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--mtl_modes", type=int, nargs="+", default=[0, 1],
                    help="0 = single-task, 1 = MTL+DIP2016")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    base_hp = HParams(
        ukp_csv=args.ukp_csv,
        dip_dir=args.dip_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )
    summary = _run_grid(
        base_hp,
        label_sets=args.labels,
        seeds=args.seeds,
        mtl_modes=[bool(m) for m in args.mtl_modes],
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
