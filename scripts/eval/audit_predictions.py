#!/usr/bin/env python3
"""Notebook-friendly audit helpers for structured prediction evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_run_eval() -> Any:
    module_path = Path(__file__).with_name("run_eval.py")
    spec = importlib.util.spec_from_file_location("audit_run_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_run_eval"] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load_run_eval()

LLM_CLIENT = None
LLM_MODEL = None


def _ensure_llm_binding() -> None:
    if LLM_CLIENT is not None:
        run_eval.LLM_CLIENT = LLM_CLIENT
    if LLM_MODEL:
        run_eval.LLM_MODEL = LLM_MODEL


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _format_row(row: dict[str, Any], fields: str | list[str]) -> str:
    if isinstance(fields, list):
        parts = [f"paper_id={run_eval.normalize_paper_id(row.get('paper_id'))}"]
        for field in fields:
            parts.append(f"{field}={run_eval.get_row_field_value(row, field)}")
        return "; ".join(parts)
    return (
        f"paper_id={run_eval.normalize_paper_id(row.get('paper_id'))}; "
        f"{fields}={run_eval.get_row_field_value(row, fields)}"
    )


def _prediction_rows(prediction: Any) -> list[dict[str, Any]]:
    rows = run_eval.normalize_pred_c_answers(prediction)
    if rows:
        return rows
    if isinstance(prediction, dict):
        answers = prediction.get("answers")
        if isinstance(answers, list):
            normalized = []
            for item in answers:
                if isinstance(item, dict):
                    normalized.append(item)
            if normalized:
                return normalized
    return []


def _ground_truth_rows(gt_answers: Any) -> list[dict[str, Any]]:
    if isinstance(gt_answers, list):
        rows = run_eval.normalize_c_answers(gt_answers)
        return rows if rows else [item for item in gt_answers if isinstance(item, dict)]
    return []


def _rows_match(pred_row: dict[str, Any], gt_row: dict[str, Any], fields: str | list[str]) -> tuple[bool, float]:
    pred_pid = run_eval.normalize_paper_id(pred_row.get("paper_id"))
    gt_pid = run_eval.normalize_paper_id(gt_row.get("paper_id"))
    if pred_pid is not None and gt_pid is not None and pred_pid != gt_pid:
        return False, 0.0

    _ensure_llm_binding()

    if isinstance(fields, list):
        total = 0.0
        for field in fields:
            pred_val = run_eval.get_row_field_value(pred_row, field)
            gt_val = run_eval.get_row_field_value(gt_row, field)
            pred_text = run_eval.strip_citations(str(pred_val))
            gt_text = run_eval.strip_citations(str(gt_val))
            if field == "sample_size":
                pred_num = run_eval.parse_floatish(pred_text)
                gt_num = run_eval.parse_floatish(gt_text)
                score = 1.0 if pred_num is not None and gt_num is not None and round(pred_num, 4) == round(gt_num, 4) else 0.0
            else:
                matched, score = run_eval.llm_match(pred_text, gt_text, field)
                if not matched:
                    score = 0.0
            total += score
        avg = total / len(fields) if fields else 0.0
        return avg > 0.0, round(avg, 4)

    pred_val = run_eval.get_row_field_value(pred_row, fields)
    gt_val = run_eval.get_row_field_value(gt_row, fields)
    pred_text = run_eval.strip_citations(str(pred_val))
    gt_text = run_eval.strip_citations(str(gt_val))
    if fields == "sample_size":
        pred_num = run_eval.parse_floatish(pred_text)
        gt_num = run_eval.parse_floatish(gt_text)
        matched = pred_num is not None and gt_num is not None and round(pred_num, 4) == round(gt_num, 4)
        return matched, 1.0 if matched else 0.0
    matched, score = run_eval.llm_match(pred_text, gt_text, fields)
    return matched, round(score if matched else 0.0, 4)


def audit_q(q_id: str, prediction: Any, gt_answers: Any, fields: str | list[str]) -> dict[str, Any]:
    del q_id
    pred_rows = _prediction_rows(prediction)
    gt_rows = _ground_truth_rows(gt_answers)
    matched: list[dict[str, Any]] = []
    missed_gt: list[str] = []
    spurious_pred: list[str] = []
    audit_rows: list[dict[str, Any]] = []

    unmatched_gt = set(range(len(gt_rows)))
    used_pred = set()

    for pred_index, pred_row in enumerate(pred_rows):
        best_gt = None
        best_score = -1.0
        for gt_index in unmatched_gt:
            is_match, score = _rows_match(pred_row, gt_rows[gt_index], fields)
            if is_match and score > best_score:
                best_gt = gt_index
                best_score = score
        pred_text = _format_row(pred_row, fields)
        if best_gt is None:
            spurious_pred.append(pred_text)
            audit_rows.append(
                {
                    "status": "spurious_pred",
                    "pred_index": pred_index,
                    "gt_index": None,
                    "pred_text": pred_text,
                    "gt_text": None,
                }
            )
            continue

        used_pred.add(pred_index)
        unmatched_gt.remove(best_gt)
        gt_text = _format_row(gt_rows[best_gt], fields)
        matched.append(
            {
                "prediction": pred_text,
                "ground_truth": gt_text,
                "score": round(best_score, 4),
                "pred_cite": run_eval.normalize_paper_id(pred_row.get("paper_id")),
                "gt_id": run_eval.normalize_paper_id(gt_rows[best_gt].get("paper_id")),
            }
        )
        audit_rows.append(
            {
                "status": "matched",
                "pred_index": pred_index,
                "gt_index": best_gt,
                "pred_text": pred_text,
                "gt_text": gt_text,
            }
        )

    for gt_index in sorted(unmatched_gt):
        gt_text = _format_row(gt_rows[gt_index], fields)
        missed_gt.append(gt_text)
        audit_rows.append(
            {
                "status": "missed_gt",
                "pred_index": None,
                "gt_index": gt_index,
                "pred_text": None,
                "gt_text": gt_text,
            }
        )

    pred_count = len(pred_rows)
    gt_count = len(gt_rows)
    matched_count = len(matched)
    precision = matched_count / pred_count if pred_count else 0.0
    recall = matched_count / gt_count if gt_count else 0.0
    return {
        "pred_count": pred_count,
        "gt_count": gt_count,
        "matched_count": matched_count,
        "missed_count": len(missed_gt),
        "spurious_count": len(spurious_pred),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "matched": matched,
        "missed_gt": missed_gt,
        "spurious_pred": spurious_pred,
        "audit_rows": audit_rows,
    }
