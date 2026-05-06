#!/usr/bin/env python3
"""
Braintrust evals for the gdpval benchmark dataset.
https://huggingface.co/datasets/openai/gdpval

Usage:
    bt eval gdpval_eval.py
    bt eval gdpval_eval.py -- --model gpt-4o --limit 20
    python gdpval_eval.py --model gpt-4o --limit 20

Environment:
    OPENAI_API_KEY
    BRAINTRUST_API_KEY
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

from braintrust import Eval, wrap_openai
from openai import OpenAI

from benchmark_utils import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_MODEL,
    build_artifacts,
    build_output_payload,
    create_chat_completion_with_retries,
    format_reference_files_section,
    normalize_rubric_score,
    output_to_scoring_text,
)

DEFAULT_PROJECT = "Leaderboard"
DEFAULT_ARTIFACT_OUTPUT_DIR = "gdpval_generated_artifacts"
USER_AGENT = "gdpval-eval/1.0 (reference fetcher)"


def infer_target_suffixes(input_data: dict[str, Any]) -> list[str]:
    text = f"{input_data.get('prompt', '')}\n{input_data.get('rubric_pretty', '')}".lower()
    suffixes: list[str] = []

    def add(suffix: str) -> None:
        if suffix not in suffixes:
            suffixes.append(suffix)

    if any(token in text for token in [".xlsx", ".xlsm", ".xls", "workbook", "worksheet", "excel"]):
        add(".xlsx")
    if any(token in text for token in [".docx", "word document", "microsoft word"]):
        add(".docx")
    if any(token in text for token in [".pptx", ".ppt", "powerpoint", "slide deck", "presentation", "slides"]):
        add(".pptx")
    if any(token in text for token in [".pdf", "compiled pdf", "form 1040", "memo file in pdf"]):
        add(".pdf")

    return suffixes or [".docx"]


def load_gdpval() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install the datasets package: pip install datasets", file=sys.stderr)
        sys.exit(1)

    dataset_name = "openai/gdpval"
    try:
        ds = load_dataset(dataset_name, split="test")
    except ValueError as err:
        if "Unknown split" not in str(err):
            raise
        ds = load_dataset(dataset_name, split="train")
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


def enrich_reference_inputs(data: list[dict[str, Any]]) -> None:
    for row in data:
        input_payload = row.get("input")
        if not isinstance(input_payload, dict):
            continue
        ref_urls = input_payload.get("reference_file_urls") or []
        if not ref_urls:
            continue
        reference_section, reference_attachments = format_reference_files_section(
            ref_urls,
            user_agent=USER_AGENT,
        )
        if reference_section:
            input_payload["reference_prompt_section"] = reference_section
        if reference_attachments:
            input_payload["reference_files"] = reference_attachments


def default_experiment_name(model: str) -> str:
    timestamp = datetime.now().strftime("%m-%d-%y-%H-%M")
    return f"GDPVAL-{model}-{timestamp}"


def make_task(model: str, client: OpenAI):
    def task(input: dict[str, Any], hooks: Any | None = None) -> dict[str, Any]:
        del hooks
        allowed_suffixes = infer_target_suffixes(input)
        suffix_list = ", ".join(allowed_suffixes)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skilled professional completing workplace tasks. "
                    "Provide detailed, accurate, and complete responses. "
                    "Return valid JSON only with this exact top-level structure:\n"
                    "{\n"
                    '  "narrative_response": string,\n'
                    '  "artifacts": [\n'
                    "    {\n"
                    '      "filename": string,\n'
                    '      "file_type": "xlsx" | "docx" | "pptx" | "pdf",\n'
                    '      "title": string (optional),\n'
                    '      "content": string (optional),\n'
                    '      "paragraphs": string[] (optional),\n'
                    '      "tables": [{"rows": string[][]}] (optional),\n'
                    '      "sheets": [{"name": string, "rows": string[][]}] (optional),\n'
                    '      "slides": [{"title": string, "bullets": string[]}] (optional)\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    f"Only generate artifact file types from this allow-list: {suffix_list}. "
                    "Never include non-JSON text."
                ),
            },
            {"role": "user", "content": input["prompt"]},
        ]

        ref_urls = input.get("reference_file_urls") or []
        reference_section = input.get("reference_prompt_section")
        if isinstance(reference_section, str) and reference_section:
            messages[1]["content"] += reference_section
        elif ref_urls:
            fallback_section, _ = format_reference_files_section(
                ref_urls,
                user_agent=USER_AGENT,
            )
            messages[1]["content"] += fallback_section

        response = create_chat_completion_with_retries(
            client,
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"narrative_response": raw, "artifacts": []}

        task_id = str(input.get("task_id", "unknown-task"))
        artifacts = build_artifacts(
            task_id,
            payload,
            allowed_suffixes,
            artifact_output_dir=DEFAULT_ARTIFACT_OUTPUT_DIR,
        )
        narrative = str(payload.get("narrative_response") or "")
        return build_output_payload(narrative, artifacts)

    return task


def make_rubric_scorer(judge_model: str, client: OpenAI):
    def rubric_scorer(input: dict[str, Any], output: Any, expected: Any = None) -> float:
        del expected
        rubric_items: list[dict[str, Any]] = json.loads(input["rubric_json"])
        if not rubric_items:
            return 0.0

        earned_points = 0.0
        output_text = output_to_scoring_text(output)

        for item in rubric_items:
            criterion = item["criterion"]
            item_score = float(item["score"])

            response = create_chat_completion_with_retries(
                client,
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert evaluator assessing whether a task response "
                            "meets a specific criterion. Respond with exactly 'YES' or 'NO'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"TASK:\n{input['prompt'][:2000]}\n\n"
                            f"RESPONSE:\n{output_text[:12000]}\n\n"
                            f"CRITERION: {criterion}\n\n"
                            "Does the response satisfy this criterion? Answer YES or NO only."
                        ),
                    },
                ],
                max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
                temperature=0,
            )

            answer = (response.choices[0].message.content or "").strip().upper()
            if "YES" in answer:
                earned_points += item_score

        return normalize_rubric_score(earned_points, rubric_items)

    return rubric_scorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Braintrust evals on the gdpval benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model under evaluation")
    p.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model used for rubric scoring (LLM-as-judge)",
    )
    p.add_argument("--project", default=DEFAULT_PROJECT, help="Braintrust project name")
    p.add_argument(
        "--experiment",
        default=None,
        help="Experiment name (defaults to GDPVAL-<model>-<mm-dd-yy-hh-mm>)",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent eval tasks",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (useful for testing)",
    )
    p.add_argument(
        "--range",
        dest="row_range",
        default=None,
        help="Half-open row range START:END, e.g. 10:25 runs rows 10-24",
    )
    args, _ = p.parse_known_args()
    return args


def apply_row_range(data: list[dict[str, Any]], row_range: str | None) -> list[dict[str, Any]]:
    if not row_range:
        return data

    parts = row_range.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid --range value {row_range!r}. Expected START:END")

    start_str, end_str = parts
    if not start_str or not end_str:
        raise ValueError(f"Invalid --range value {row_range!r}. Expected START:END")

    start = int(start_str)
    end = int(end_str)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"Invalid --range value {row_range!r}. Expected 0 <= START <= END")

    return data[start:end]


def main() -> None:
    args = parse_args()
    run_eval(args)


def run_eval(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = wrap_openai(OpenAI(api_key=api_key))

    print("Loading gdpval dataset...", flush=True)
    data = load_gdpval()
    data = apply_row_range(data, args.row_range)
    if args.limit:
        data = data[: args.limit]
    enrich_reference_inputs(data)
    print(f"Loaded {len(data)} rows", flush=True)

    Eval(
        args.project,
        experiment_name=args.experiment or default_experiment_name(args.model),
        data=data,
        task=make_task(args.model, client),
        scores=[make_rubric_scorer(args.judge_model, client)],
        metadata={
            "model": args.model,
            "judge_model": args.judge_model,
            "dataset": "openai/gdpval",
        },
        max_concurrency=args.max_concurrency,
    )


if __name__ != "__main__":
    run_eval(parse_args())


if __name__ == "__main__":
    main()
