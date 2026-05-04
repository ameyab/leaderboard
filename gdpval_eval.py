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
import re
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
GENERATABLE_SUFFIXES = frozenset({".xlsx", ".docx", ".pptx", ".pdf"})
MAX_OFFICE_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_EXTRACT_CHARS_PER_FILE = 250_000
MAX_REFERENCE_CONTENT_TOTAL_CHARS = 500_000
MAX_GENERATED_FILES = 4
MAX_GENERATED_CONTENT_CHARS_PER_FILE = 80_000
FETCH_TIMEOUT_SEC = 90
USER_AGENT = "gdpval-eval/1.0 (reference fetcher)"
DEFAULT_ARTIFACT_OUTPUT_DIR = "generated_artifacts"


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


def infer_target_suffixes(input_data: dict) -> list[str]:
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


def sanitize_filename(filename: str, fallback_suffix: str) -> str:
    candidate = (filename or "").strip()
    candidate = candidate.replace("\\", "_").replace("/", "_")
    candidate = re.sub(r"[^A-Za-z0-9._ -]+", "_", candidate).strip(" .")
    if not candidate:
        candidate = f"deliverable{fallback_suffix}"
    suffix = os.path.splitext(candidate)[1].lower()
    if suffix in {".xls", ".xlsm"}:
        candidate = os.path.splitext(candidate)[0] + ".xlsx"
    elif suffix == ".ppt":
        candidate = os.path.splitext(candidate)[0] + ".pptx"
    elif suffix not in GENERATABLE_SUFFIXES:
        candidate = os.path.splitext(candidate)[0] + fallback_suffix
    return candidate


def _string_rows(raw_rows: Any) -> list[list[str]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[list[str]] = []
    for row in raw_rows:
        if isinstance(row, list):
            rows.append([str(cell) if cell is not None else "" for cell in row])
    return rows


def build_docx(path: str, spec: dict) -> None:
    from docx import Document

    doc = Document()
    title = str(spec.get("title") or "").strip()
    if title:
        doc.add_heading(title, level=1)

    paragraphs = spec.get("paragraphs")
    if isinstance(paragraphs, list):
        for paragraph in paragraphs:
            text = str(paragraph).strip()
            if text:
                doc.add_paragraph(text)
    else:
        body = str(spec.get("content") or spec.get("text") or "").strip()
        if body:
            for paragraph in body.splitlines():
                if paragraph.strip():
                    doc.add_paragraph(paragraph.strip())

    tables = spec.get("tables")
    if isinstance(tables, list):
        for table_spec in tables:
            rows = _string_rows((table_spec or {}).get("rows"))
            if not rows:
                continue
            tbl = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
            for r_idx, row in enumerate(rows):
                for c_idx, value in enumerate(row):
                    tbl.cell(r_idx, c_idx).text = value

    if len(doc.paragraphs) == 0 and len(doc.tables) == 0:
        doc.add_paragraph("(empty deliverable)")
    doc.save(path)


def build_xlsx(path: str, spec: dict) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    sheets = spec.get("sheets")
    if isinstance(sheets, list):
        for idx, sheet_spec in enumerate(sheets, start=1):
            name = str((sheet_spec or {}).get("name") or f"Sheet{idx}").strip()[:31] or f"Sheet{idx}"
            ws = wb.create_sheet(title=name)
            for row in _string_rows((sheet_spec or {}).get("rows")):
                ws.append(row)
    else:
        ws = wb.create_sheet(title="Sheet1")
        raw = str(spec.get("content") or spec.get("text") or "").strip()
        if raw:
            for line in raw.splitlines():
                ws.append([cell.strip() for cell in line.split("\t")])

    if len(wb.sheetnames) == 0:
        ws = wb.create_sheet(title="Sheet1")
        ws["A1"] = "(empty deliverable)"
    wb.save(path)
    wb.close()


def build_pptx(path: str, spec: dict) -> None:
    from pptx import Presentation

    prs = Presentation()
    slides = spec.get("slides")
    if isinstance(slides, list) and slides:
        for slide_spec in slides:
            layout = prs.slide_layouts[1]
            slide = prs.slides.add_slide(layout)
            title = str((slide_spec or {}).get("title") or "").strip()
            body_lines = slide_spec.get("bullets")
            if not isinstance(body_lines, list):
                fallback = str((slide_spec or {}).get("content") or "").strip()
                body_lines = [line.strip() for line in fallback.splitlines() if line.strip()]
            slide.shapes.title.text = title or "Slide"
            text_frame = slide.shapes.placeholders[1].text_frame
            text_frame.clear()
            if body_lines:
                text_frame.text = str(body_lines[0])
                for line in body_lines[1:]:
                    p = text_frame.add_paragraph()
                    p.text = str(line)
            else:
                text_frame.text = "(empty)"
    else:
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = str(spec.get("title") or "Slide 1")
        text_frame = slide.shapes.placeholders[1].text_frame
        text_frame.text = str(spec.get("content") or spec.get("text") or "(empty)")

    prs.save(path)


def build_pdf(path: str, spec: dict) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    lines: list[str] = []
    title = str(spec.get("title") or "").strip()
    if title:
        lines.append(title)
        lines.append("")
    paragraphs = spec.get("paragraphs")
    if isinstance(paragraphs, list):
        for paragraph in paragraphs:
            lines.extend(str(paragraph).splitlines())
            lines.append("")
    else:
        body = str(spec.get("content") or spec.get("text") or "").strip()
        lines.extend(body.splitlines() or ["(empty deliverable)"])

    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 50
    for raw_line in lines:
        line = raw_line.rstrip()
        chunks = [line[i : i + 110] for i in range(0, len(line), 110)] or [""]
        for chunk in chunks:
            c.drawString(50, y, chunk)
            y -= 14
            if y < 50:
                c.showPage()
                y = height - 50
    c.save()


def _extract_generated_content(path: str, suffix: str, spec: dict) -> str:
    if suffix in OFFICE_SUFFIXES:
        with open(path, "rb") as f:
            data = f.read()
        return extract_office_text(path, data, suffix)
    if suffix == ".pdf":
        lines = []
        if spec.get("title"):
            lines.append(str(spec.get("title")))
        paragraphs = spec.get("paragraphs")
        if isinstance(paragraphs, list):
            lines.extend(str(p) for p in paragraphs)
        else:
            lines.append(str(spec.get("content") or spec.get("text") or ""))
        return _truncate_text("\n".join(lines).strip() or "(empty pdf)", MAX_GENERATED_CONTENT_CHARS_PER_FILE)
    return ""


def build_artifacts(task_id: str, payload: dict, allowed_suffixes: list[str]) -> list[dict]:
    task_dir = os.path.join(DEFAULT_ARTIFACT_OUTPUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return []

    out: list[dict] = []
    for artifact in artifacts[:MAX_GENERATED_FILES]:
        if not isinstance(artifact, dict):
            continue
        requested_type = str(artifact.get("file_type") or "").strip().lower()
        filename = str(artifact.get("filename") or "").strip()
        suffix = os.path.splitext(filename)[1].lower()
        if requested_type and not requested_type.startswith("."):
            requested_type = "." + requested_type
        if requested_type in {".xls", ".xlsm"}:
            requested_type = ".xlsx"
        if requested_type == ".ppt":
            requested_type = ".pptx"
        if suffix in {".xls", ".xlsm"}:
            suffix = ".xlsx"
        elif suffix == ".ppt":
            suffix = ".pptx"
        target_suffix = suffix or requested_type
        if target_suffix not in GENERATABLE_SUFFIXES:
            target_suffix = allowed_suffixes[0] if allowed_suffixes else ".docx"
        if target_suffix not in allowed_suffixes:
            continue

        safe_name = sanitize_filename(filename or f"deliverable{target_suffix}", target_suffix)
        path = os.path.join(task_dir, safe_name)

        try:
            if target_suffix == ".docx":
                build_docx(path, artifact)
            elif target_suffix == ".xlsx":
                build_xlsx(path, artifact)
            elif target_suffix == ".pptx":
                build_pptx(path, artifact)
            elif target_suffix == ".pdf":
                build_pdf(path, artifact)
            else:
                continue
            out.append(
                {
                    "filename": safe_name,
                    "suffix": target_suffix,
                    "path": path,
                    "content": _extract_generated_content(path, target_suffix, artifact),
                }
            )
        except Exception as err:
            print(f"WARNING: Could not create artifact {safe_name}: {err}", file=sys.stderr)
    return out


def format_artifact_output(narrative: str, artifacts: list[dict]) -> str:
    parts = [f"NARRATIVE RESPONSE:\n{(narrative or '').strip()}"]
    if not artifacts:
        parts.append("\nGENERATED FILES:\n(none)")
        return "\n\n".join(parts)

    lines = ["\nGENERATED FILES:"]
    for idx, artifact in enumerate(artifacts, start=1):
        lines.append(f"{idx}. {artifact['filename']} ({artifact['suffix']})")
        lines.append(f"   path: {artifact['path']}")
    parts.append("\n".join(lines))

    content_blocks: list[str] = []
    for artifact in artifacts:
        body = _truncate_text(artifact.get("content") or "", MAX_GENERATED_CONTENT_CHARS_PER_FILE)
        if not body:
            continue
        content_blocks.append(
            f"### Extracted content: {artifact['filename']}\n"
            f"{body}"
        )
    if content_blocks:
        parts.append("GENERATED FILE CONTENTS:\n\n" + "\n\n".join(content_blocks))
    return "\n\n".join(parts)


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
        if ref_urls:
            messages[1]["content"] += format_reference_files_section(ref_urls)

        response = client.chat.completions.create(model=model, messages=messages, response_format={"type": "json_object"})
        raw = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"narrative_response": raw, "artifacts": []}

        artifacts = build_artifacts(str(input.get("task_id", "unknown-task")), payload, allowed_suffixes)
        narrative = str(payload.get("narrative_response") or "")
        return format_artifact_output(narrative, artifacts)

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
                            f"RESPONSE:\n{output[:12000]}\n\n"
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
