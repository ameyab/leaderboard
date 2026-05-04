#!/usr/bin/env python3
"""
Upload the openai/gdpval Hugging Face dataset to a Braintrust Dataset.

Rows match the shape used by gdpval_eval.py (input + metadata). Each row id is the
dataset task_id so re-running the script upserts the same records.

Usage:
  python publish_gdpval_dataset.py
  python publish_gdpval_dataset.py --project gdpval --name gdpval --limit 50

Environment:
  BRAINTRUST_API_KEY   Required (unless already logged in via the SDK).
  BRAINTRUST_ORG_NAME  Optional, if you belong to multiple orgs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from braintrust import init_dataset

HF_DATASET = "openai/gdpval"


def _load_hf_split(split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    if split == "auto":
        try:
            return load_dataset(HF_DATASET, split="test")
        except ValueError as err:
            if "Unknown split" not in str(err):
                raise
            return load_dataset(HF_DATASET, split="train")
    return load_dataset(HF_DATASET, split=split)


def hf_rows_to_eval_records(ds) -> list[dict[str, Any]]:
    return [
        {
            "input": {
                "prompt": row["prompt"],
                "task_id": row["task_id"],
                "sector": row["sector"],
                "occupation": row["occupation"],
                "reference_file_urls": row.get("reference_file_urls") or [],
                "rubric_json": row["rubric_json"],
                "rubric_pretty": row["rubric_pretty"],
            },
            "metadata": {
                "task_id": row["task_id"],
                "sector": row["sector"],
                "occupation": row["occupation"],
            },
        }
        for row in ds
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Publish GDPVal to a Braintrust dataset.")
    p.add_argument(
        "--project",
        default=os.environ.get("BRAINTRUST_PROJECT", "gdpval"),
        help="Braintrust project name",
    )
    p.add_argument(
        "--name",
        default=os.environ.get("BRAINTRUST_DATASET_NAME", "gdpval"),
        help="Braintrust dataset name",
    )
    p.add_argument("--description", default=None, help="Dataset description")
    p.add_argument(
        "--split",
        choices=("auto", "train", "test"),
        default="auto",
        help="Hugging Face split (auto tries test then train)",
    )
    p.add_argument("--limit", type=int, default=None, help="Upload only the first N rows")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load from Hugging Face and print counts only",
    )
    args = p.parse_args()

    print(f"Loading {HF_DATASET} (split={args.split})...", flush=True)
    ds = _load_hf_split(args.split)
    records = hf_rows_to_eval_records(ds)
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Prepared {len(records)} rows", flush=True)

    if args.dry_run:
        return

    org_name = os.environ.get("BRAINTRUST_ORG_NAME")
    dataset = init_dataset(
        project=args.project,
        name=args.name,
        description=args.description,
        org_name=org_name,
        metadata={"source": HF_DATASET, "hf_split_resolved": args.split},
    )

    for i, rec in enumerate(records):
        task_id = rec["input"]["task_id"]
        dataset.insert(
            id=str(task_id),
            input=rec["input"],
            metadata=rec["metadata"],
        )
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f"  queued {i + 1}/{len(records)}", flush=True)

    summary = dataset.summarize()
    print(summary, flush=True)


if __name__ == "__main__":
    main()
