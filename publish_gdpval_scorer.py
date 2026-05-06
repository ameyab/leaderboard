#!/usr/bin/env python3
"""
Publishable Braintrust scorer definition for GDPVal rubric scoring.

Usage:
  # Publish/update scorer in Braintrust
  bt functions push publish_gdpval_scorer.py --if-exists replace

Environment:
  OPENAI_API_KEY                Required by the scorer at runtime.
  BRAINTRUST_PROJECT            Optional, defaults to "gdpval".
  GDPVAL_SCORER_NAME            Optional, defaults to "gdpval-rubric-scorer".
  GDPVAL_SCORER_SLUG            Optional, defaults to "gdpval-rubric-scorer".
  GDPVAL_SCORER_JUDGE_MODEL     Optional, defaults to "gpt-4.1-mini".
"""

from __future__ import annotations

import json
import os
from typing import Any

from braintrust import projects, wrap_openai
from openai import OpenAI
from pydantic import BaseModel, Field

from benchmark_utils import (
    DEFAULT_JUDGE_MAX_TOKENS,
    create_chat_completion_with_retries,
    normalize_rubric_score,
    output_to_scoring_text,
)

PROJECT_NAME = os.getenv("BRAINTRUST_PROJECT", "gdpval")
SCORER_NAME = os.getenv("GDPVAL_SCORER_NAME", "gdpval-rubric-scorer")
SCORER_SLUG = os.getenv("GDPVAL_SCORER_SLUG", "gdpval-rubric-scorer")
JUDGE_MODEL = os.getenv("GDPVAL_SCORER_JUDGE_MODEL", "gpt-4.1-mini")


class RubricScorerInput(BaseModel):
    output: str = Field(..., description="Model response under evaluation")
    input: dict[str, Any] = Field(..., description="Eval input payload containing prompt and rubric_json")
    expected: Any | None = Field(default=None, description="Optional expected value")


def _extract_rubric_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    if "rubric_json" in input_payload:
        return input_payload
    nested = input_payload.get("input")
    if isinstance(nested, dict) and "rubric_json" in nested:
        return nested
    return input_payload

def gdpval_rubric_scorer(output: str, input: dict[str, Any], expected: Any = None) -> float:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run gdpval_rubric_scorer")

    payload = _extract_rubric_payload(input)
    rubric_items = json.loads(payload["rubric_json"])
    if not rubric_items:
        return 0.0

    prompt = payload.get("prompt", "")
    client = wrap_openai(OpenAI(api_key=api_key))
    earned_points = 0.0
    output_text = _output_to_scoring_text(output)

    for item in rubric_items:
        criterion = item["criterion"]
        item_score = float(item["score"])

        response = create_chat_completion_with_retries(
            client,
            model=JUDGE_MODEL,
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
                        f"TASK:\n{prompt[:2000]}\n\n"
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


project = projects.create(PROJECT_NAME)
project.scorers.create(
    handler=gdpval_rubric_scorer,
    name=SCORER_NAME,
    slug=SCORER_SLUG,
    description="GDPVal rubric scorer using criterion-by-criterion LLM judgment.",
    if_exists="replace",
    parameters=RubricScorerInput,
    metadata={"dataset": "openai/gdpval", "judge_model": JUDGE_MODEL},
    tags=["gdpval", "rubric", "llm-judge"],
)
