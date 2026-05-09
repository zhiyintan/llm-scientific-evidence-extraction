#!/usr/bin/env python3
"""Shared helpers for notebook and batch evaluation."""

from __future__ import annotations

import json
import os
import re
from statistics import mean, median
from typing import Any

import threading

from openai import OpenAI

LLM_CLIENT: OpenAI | None = None
LLM_MODEL: str | None = None
_INIT_LOCK = threading.Lock()

_LLM_BASE_URL = os.environ.get("EVAL_LLM_BASE_URL", "http://localhost:8100/v1")
_LLM_API_KEY = os.environ.get("EVAL_LLM_API_KEY", "EMPTY")


def strip_citations(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def parse_floatish(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = strip_citations(str(value)).lower()
    if not text or text in {"n/a", "na", "none", "null", "-", "not reported"}:
        return None

    text = text.replace(",", "")
    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)", text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def normalize_paper_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def normalize_c_answers(gt_answers: Any) -> list[dict[str, Any]]:
    if not isinstance(gt_answers, list):
        return gt_answers if isinstance(gt_answers, list) else []
    normalized: list[dict[str, Any]] = []
    for item in gt_answers:
        if not isinstance(item, dict):
            continue
        paper_id = item.get("paper_id")
        answers = item.get("answer", [])
        if not isinstance(answers, list):
            answers = [answers]
        for ans in answers:
            merged = {"paper_id": paper_id}
            if isinstance(ans, dict):
                merged.update(ans)
            else:
                merged["answer"] = ans
            normalized.append(merged)
    return normalized


def normalize_pred_c_answers(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict) and "answers" in raw:
        raw = raw["answers"]
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        paper_id = item.get("paper_id")
        answers = item.get("answer", [])
        if not isinstance(answers, list):
            answers = [answers]
        for ans in answers:
            merged = {"paper_id": paper_id}
            if isinstance(ans, dict):
                merged.update(ans)
            else:
                merged["answer"] = ans
            normalized.append(merged)
    return normalized


def get_row_field_value(row: dict[str, Any], field: str) -> Any:
    aliases = {
        "measurement_scale_or_unit": ["measurement_scale_or_unit", "unit", "scale"],
        "unit": ["unit", "measurement_scale_or_unit"],
        "scale": ["scale", "measurement_scale_or_unit"],
    }
    for key in aliases.get(field, [field]):
        if key in row:
            val = row.get(key)
            if val is not None and val != "":
                return val
    return row.get(field, row.get("answer", ""))


def _normalize_text(text: Any) -> str:
    return strip_citations(str(text)).lower()


def _ensure_client() -> OpenAI:
    """Lazy-init a shared OpenAI client + resolve the judge model id.

    Resolution order (first non-empty wins — NO silent fallback to a hardcoded
    placeholder):
      1. EVAL_LLM_MODEL env var (explicit user override; highest priority).
      2. GET /v1/models on the configured base_url (auto-detect).

    If neither produces a usable model id, raise RuntimeError immediately —
    better to fail loudly than to silently send every judge call to a
    non-existent model and record spurious no_match verdicts.
    """
    global LLM_CLIENT, LLM_MODEL
    with _INIT_LOCK:
        if LLM_CLIENT is None:
            LLM_CLIENT = OpenAI(base_url=_LLM_BASE_URL, api_key=_LLM_API_KEY)
        if not LLM_MODEL:
            explicit = os.environ.get("EVAL_LLM_MODEL", "").strip()
            if explicit:
                LLM_MODEL = explicit
            else:
                try:
                    LLM_MODEL = LLM_CLIENT.models.list().data[0].id
                except Exception as exc:
                    raise RuntimeError(
                        f"Eval judge model could not be resolved. "
                        f"Auto-detect via {_LLM_BASE_URL}/models failed ({exc}) and "
                        f"EVAL_LLM_MODEL env var is not set. "
                        f"Set EVAL_LLM_MODEL explicitly to the served model id "
                        f"(e.g., 'google/gemma-4-31B-it') to avoid silent 404s."
                    ) from exc
    return LLM_CLIENT


KIND_TO_LABEL = {"exact": "EXACT", "over_detail": "OVER_DETAIL", "less_detail": "LESS_DETAIL", "no_match": "NO"}
_KIND_TO_MQ = {"exact": 1.0, "over_detail": 0.75, "less_detail": 0.5, "no_match": 0.0}

_LLM_JUDGE_PROMPT = """You are evaluating whether a model prediction matches a ground-truth answer.

Field context: {ctx}

Compare prediction vs ground truth based on core identity, not surface wording.

Definitions:
- Core identity: the minimal information needed to unambiguously pick out the entity/concept the ground truth refers to within a single study.
- Supporting detail: extra modifiers, units, abbreviations, age ranges, grade ranges, specific addresses, cultivars, parenthetical notes, statistics — extra description that does not change the entity identity.

Return one of these "kind" values:
- "exact"       : prediction and ground truth express the same core identity with no meaningful loss or addition.
- "over_detail" : prediction covers all core identity of the ground truth AND adds extra supporting detail beyond the ground truth.
- "less_detail" : prediction is less rich than the ground truth, BUT what prediction says is specific enough to unambiguously pick out the same entity the ground truth refers to within the study, with no conflict.
- "no_match"    : prediction misses essential identifying info, over-generalizes to a broader category that could refer to a different entity, or contradicts the ground truth.

Key distinction between less_detail and no_match:
- less_detail: ground truth's extra content is demographic/descriptive decoration (grade, age range, specific address, cultivar, subspecies, unit) that further describes the same entity the prediction names. Replacing ground truth with the prediction does not change which entity is meant.
- no_match  : prediction is a broader category such that, if you replaced ground truth with the prediction, a reader could not identify the same entity anymore.

Examples:
- ctx=study_population, pred="students", gt="students in grades 1 (7-8), 4 (10-11), 7 (13-14)" -> less_detail
- ctx=study_population, pred="children", gt="Children aged 5-18 years" -> less_detail
- ctx=study_population, pred="people", gt="elderly aged 65 and older" -> no_match
- ctx=geolocation, pred="Central Kalimantan", gt="Central Kalimantan, Indonesia" -> less_detail
- ctx=geolocation, pred="Asia", gt="Indonesia" -> no_match
- ctx=variable, pred="plant water content", gt="plant water content (% of dry mass)" -> less_detail
- ctx=variable, pred="NDVI", gt="Normalized Difference Vegetation Index (NDVI)" -> exact
- ctx=variable, pred="yield", gt="grain yield (t/ha)" -> no_match

Return JSON only with keys:
- kind: one of "exact", "over_detail", "less_detail", "no_match"
- match: boolean (true for exact/over_detail/less_detail; false for no_match)

Prediction: {pred_text}
Ground truth: {gt_text}"""


def llm_match_3tier(pred_text: str, gt_text: str, ctx: str) -> tuple[bool, float, str]:
    """Return (is_match, match_quality, kind). kind is one of
    'exact' / 'over_detail' / 'less_detail' / 'no_match'.

    Retry behaviour:
      - Transient errors (connection/timeout/empty content/bad JSON) are
        retried up to 3 times.
      - Unrecoverable errors (HTTP 400/401/403/404 — wrong model / auth /
        request) RAISE immediately. Silent fallback to no_match would
        undercount F1 without any signal to the operator; fail loud instead.
      - If all 3 retries fail with transient errors, RAISE as well. The
        caller (evaluate_file in batch_eval.py) catches per-file so one
        bad file doesn't sink an entire batch, but the failure surfaces.
    """
    pred_text = strip_citations(pred_text)
    gt_text = strip_citations(gt_text)

    if not pred_text or not gt_text or pred_text == "—" or gt_text == "—":
        return False, 0.0, "no_match"

    if _normalize_text(pred_text) == _normalize_text(gt_text):
        return True, 1.0, "exact"

    client = _ensure_client()
    prompt = _LLM_JUDGE_PROMPT.format(ctx=ctx, pred_text=pred_text, gt_text=gt_text)

    # Import lazily to avoid pulling openai internals at module import time.
    try:
        from openai import APIStatusError, APIConnectionError, APITimeoutError
    except ImportError:
        APIStatusError = APIConnectionError = APITimeoutError = tuple()

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
                messages=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content
            content = (raw or "").strip()
            if not content:
                raise ValueError("empty LLM content")
            payload = json.loads(content)
            kind = str(payload.get("kind", "no_match")).lower()
            if kind not in _KIND_TO_MQ:
                kind = "no_match"
            return (kind != "no_match"), _KIND_TO_MQ[kind], kind
        except Exception as exc:
            # HTTP 4xx (bad request / model not found / auth) is NOT going to
            # be fixed by retrying; fail loudly so the operator notices.
            status = getattr(exc, "status_code", None)
            if status in (400, 401, 403, 404):
                raise RuntimeError(
                    f"LLM judge fatal error (HTTP {status}) — likely wrong "
                    f"EVAL_LLM_MODEL/base_url/api_key. model={LLM_MODEL!r} "
                    f"base_url={_LLM_BASE_URL!r} err={exc}"
                ) from exc
            last_err = exc
            continue

    raise RuntimeError(
        f"LLM judge failed after 3 transient retries. "
        f"ctx={ctx!r} model={LLM_MODEL!r} base_url={_LLM_BASE_URL!r} "
        f"pred={pred_text!r} gt={gt_text!r} last_err={last_err}"
    )


def llm_match(pred_text: str, gt_text: str, ctx: str) -> tuple[bool, float]:
    """Back-compat 2-tuple wrapper around llm_match_3tier."""
    is_match, mq, _kind = llm_match_3tier(pred_text, gt_text, ctx)
    return is_match, mq


def get_oc_answers(question_payload: dict[str, Any]) -> dict[str, Any]:
    answers = question_payload.get("answers", {})
    return answers if isinstance(answers, dict) else {}


def extract_scalar_from_union(pred_ans: Any, gt_field: str, agg_mode: str) -> float | None:
    del gt_field
    if isinstance(pred_ans, dict):
        direct = parse_floatish(pred_ans.get("final_answer"))
        if direct is not None:
            return direct
        values = pred_ans.get("final_list")
    else:
        values = pred_ans

    if not isinstance(values, list):
        return None

    nums = [parse_floatish(v) for v in values]
    nums = [v for v in nums if v is not None]
    if agg_mode == "count":
        return float(len(nums if nums else values))
    if not nums:
        return None
    if agg_mode == "avg":
        return float(mean(nums))
    if agg_mode == "median":
        return float(median(nums))
    return parse_floatish(values[0]) if values else None


def compute_scalar_score(pred_val: Any, gt_val: Any) -> float:
    pred_num = parse_floatish(pred_val)
    gt_num = parse_floatish(gt_val)
    if pred_num is None or gt_num is None:
        return 0.0
    return 1.0 if round(pred_num, 4) == round(gt_num, 4) else 0.0
