#!/usr/bin/env python3
"""
Braintrust evals for the gdpval benchmark dataset.
https://huggingface.co/datasets/openai/gdpval

Usage:
    bt eval gdpval_eval.py
    bt eval gdpval_eval.py -- --model gpt-4o --limit 20
    python gdpval_eval.py --model gpt-4o --limit 20

Requires:
    pip install braintrust autoevals openai datasets openpyxl python-docx python-pptx

Environment:
    OPENAI_API_KEY
    BRAINTRUST_API_KEY
"""

import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import unquote, urlparse
from typing import Any

from braintrust import Eval
from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_PROJECT = "gdpval"
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"

OFFICE_SUFFIXES = frozenset({".xlsx", ".docx", ".pptx"})
MAX_OFFICE_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_EXTRACT_CHARS_PER_FILE = 250_000
MAX_REFERENCE_CONTENT_TOTAL_CHARS = 500_000
FETCH_TIMEOUT_SEC = 90
USER_AGENT = "gdpval-eval/1.0 (reference fetcher)"


def _suffix_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    return os.path.splitext(path)[1].lower()


def fetch_url_bytes(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SEC) as resp:
            chunks: list[bytes] = []
            total = 0
            while total < MAX_OFFICE_DOWNLOAD_BYTES:
                chunk = resp.read(min(65536, MAX_OFFICE_DOWNLOAD_BYTES - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            if resp.read(1):
                print(f"WARNING: Skipping oversized reference file: {url}", file=sys.stderr)
                return None
            return b"".join(chunks)
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        print(f"WARNING: Could not download reference file {url}: {err}", file=sys.stderr)
        return None


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated {len(text) - limit} characters ...]"


def extract_xlsx_text(data: bytes) -> str:
    from openpyxl import load_workbook

    buf = io.BytesIO(data)
    wb = load_workbook(buf, read_only=True, data_only=True)
    try:
        parts: list[str] = []
        for sheet in wb:
            rows_out: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                if any(cells):
                    rows_out.append("\t".join(cells))
            if rows_out:
                parts.append(f"## Sheet: {sheet.title}\n" + "\n".join(rows_out))
        return "\n\n".join(parts) if parts else "(empty workbook)"
    finally:
        wb.close()


def extract_docx_text(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts: list[str] = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts) if parts else "(empty document)"


def extract_pptx_text(data: bytes) -> str:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(io.BytesIO(data))

    def shapes_text(shapes) -> list[str]:
        lines: list[str] = []
        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                lines.extend(shapes_text(shape.shapes))
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    t = "".join(run.text for run in paragraph.runs).strip()
                    if t:
                        lines.append(t)
        return lines

    parts: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        slide_lines = shapes_text(slide.shapes)
        if slide_lines:
            parts.append(f"--- Slide {i} ---\n" + "\n".join(slide_lines))
    return "\n\n".join(parts) if parts else "(empty presentation)"


def extract_office_text(url: str, data: bytes, suffix: str) -> str:
    try:
        if suffix == ".xlsx":
            raw = extract_xlsx_text(data)
        elif suffix == ".docx":
            raw = extract_docx_text(data)
        elif suffix == ".pptx":
            raw = extract_pptx_text(data)
        else:
            return ""
    except Exception as err:
        print(f"WARNING: Could not parse Office file {url}: {err}", file=sys.stderr)
        return ""
    return _truncate_text(raw, MAX_EXTRACT_CHARS_PER_FILE)


def build_reference_attachment(urls: list[str]) -> str:
    """Download Office reference files and return markdown-style text to append to the user message."""
    blocks: list[str] = []
    total_budget = MAX_REFERENCE_CONTENT_TOTAL_CHARS

    for url in urls[:50]:
        suffix = _suffix_from_url(url)
        if suffix not in OFFICE_SUFFIXES:
            continue
        data = fetch_url_bytes(url)
        if not data:
            continue
        body = extract_office_text(url, data, suffix)
        if not body:
            continue
        header = f"### Reference file ({suffix}): {url}\n"
        room = total_budget - len(header)
        if room <= 0:
            blocks.append(
                "### Reference files\n"
                f"[Omitted remaining Office attachments: budget {MAX_REFERENCE_CONTENT_TOTAL_CHARS} chars exceeded]"
            )
            break
        if len(body) > room:
            body = _truncate_text(body, room)
        blocks.append(header + body)
        total_budget -= len(header) + len(body)

    if not blocks:
        return ""
    return "\n\n".join(["## Extracted reference file contents (Office documents only)", *blocks])


def format_reference_files_section(urls: list[str]) -> str:
    if not urls:
        return ""
    url_list = "\n".join(f"  - {u}" for u in urls[:50])
    office_blob = build_reference_attachment(urls)
    parts = [f"\n\nReference files:\n{url_list}"]
    if office_blob:
        parts.append("\n\n" + office_blob)
    return "".join(parts)


def load_gdpval() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install the datasets package: pip install datasets", file=sys.stderr)
        sys.exit(1)

    dataset_name = "openai/gdpval"
    try:
        ds = load_dataset(dataset_name, split="test")
    except ValueError as err:
        # Some versions of this dataset expose only a "train" split.
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


def make_task(model: str, client: OpenAI):
    def task(input: dict) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skilled professional completing workplace tasks. "
                    "Provide detailed, accurate, and complete responses. "
                    "When asked to create a document or file, describe its full contents "
                    "in a structured way that could be used to recreate it exactly."
                ),
            },
            {"role": "user", "content": input["prompt"]},
        ]

        ref_urls = input.get("reference_file_urls") or []
        if ref_urls:
            messages[1]["content"] += format_reference_files_section(ref_urls)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content

    return task


def make_rubric_scorer(judge_model: str, client: OpenAI):
    """LLM-as-judge scorer that evaluates output against each rubric criterion."""

    def rubric_scorer(input: dict, output: str, expected: Any = None) -> float:
        rubric_items: list[dict] = json.loads(input["rubric_json"])
        total_points = sum(item["score"] for item in rubric_items)

        if total_points == 0 or not rubric_items:
            return 0.0

        earned_points = 0

        for item in rubric_items:
            criterion = item["criterion"]
            item_score = item["score"]

            response = client.chat.completions.create(
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
                            f"RESPONSE:\n{output[:3000]}\n\n"
                            f"CRITERION: {criterion}\n\n"
                            "Does the response satisfy this criterion? Answer YES or NO only."
                        ),
                    },
                ],
                max_tokens=5,
                temperature=0,
            )

            answer = response.choices[0].message.content.strip().upper()
            if "YES" in answer:
                earned_points += item_score

        return earned_points / total_points

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
        help="Experiment name (defaults to --model value)",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent eval tasks",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (useful for testing)",
    )
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Loading gdpval dataset...", flush=True)
    data = load_gdpval()
    if args.limit:
        data = data[: args.limit]
    print(f"Loaded {len(data)} rows", flush=True)

    Eval(
        args.project,
        experiment_name=args.experiment or args.model,
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


if __name__ == "__main__":
    main()
