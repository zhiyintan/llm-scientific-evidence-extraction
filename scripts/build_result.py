#!/usr/bin/env python3
"""
Build `result/` from raw predictions + eval CSVs.

Output layout:
  result/<model>/<mode>/<qid>/<domain>/<run>/{metrics.json, prediction.json}

Sources:
  - Raw predictions:   data/results/predictions/<pred_dir>/<mode>/<domain>/<model_dir>/predictions.json
  - Eval aggregates:   scripts/eval/results/<eval_dir>/all_metrics.csv
  - Eval breakdowns:   scripts/eval/results/<eval_dir>/*__<mode>__<domain>__<model_dir>__predictions.log

The mapping of (model, run, kind) → (pred_dir, eval_dir) is declared in the
SOURCE table below; patch layers (rescue_*, reagg_object_*) are applied
afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
PRED_ROOT = REPO / "data" / "results" / "predictions"
EVAL_ROOT = REPO / "scripts" / "eval" / "results"
OUT_ROOT = REPO / "result"

DOMAINS = ["agriculture", "biodiversity", "hotel", "myopia", "social"]


class EvalSrc(NamedTuple):
    eval_dir: str
    pred_filter: str | None  # substring of CSV pred_file column to include; None = all


# (model, run, kind) -> pred directory name under data/results/predictions/
PREDICTIONS_DIR: dict[tuple[str, int, str], str] = {
    ("gemma4_31b", 1, "method"): "method_gemma4_all_20260410",
    ("gemma4_31b", 1, "object"): "object_gemma4_20260413",
    ("gemma4_31b", 2, "method"): "run2_gemma4_method_20260419",
    ("gemma4_31b", 2, "object"): "run2_gemma4_20260419",
    ("gemma4_31b", 3, "method"): "run3_gemma4_method_20260419",
    ("gemma4_31b", 3, "object"): "run3_gemma4_20260419",
    ("qwen3_5_9b", 1, "method"): "run1_qwen35_9b_method_20260419",
    ("qwen3_5_9b", 1, "object"): "run1_qwen35_9b_20260419",
    ("qwen3_5_9b", 2, "method"): "run2_qwen35_9b_method_20260419",
    ("qwen3_5_9b", 2, "object"): "run2_qwen35_9b_20260419",
    ("qwen3_5_9b", 3, "method"): "run3_qwen35_9b_method_20260419",
    ("qwen3_5_9b", 3, "object"): "run3_qwen35_9b_20260419",
    ("qwen3_5_122b", 1, "method"): "method_qwen35_all_20260410",
    ("qwen3_5_122b", 1, "object"): "object_qwen35_20260413",
    ("qwen3_5_122b", 2, "method"): "run2_qwen35_122b_method_20260419",
    ("qwen3_5_122b", 2, "object"): "run2_qwen35_122b_20260419",
    ("qwen3_5_122b", 3, "method"): "run3_qwen35_122b_method_20260419",
    ("qwen3_5_122b", 3, "object"): "run3_qwen35_122b_20260419",
    ("minimax_m25", 1, "method"): "method_minimax_20260410",
    ("minimax_m25", 1, "object"): "object_minimax_20260413",
    ("minimax_m25", 2, "method"): "run2_minimax_method_20260419",
    ("minimax_m25", 2, "object"): "run2_minimax_20260419",
    ("minimax_m25", 3, "method"): "run3_minimax_method_20260419",
    ("minimax_m25", 3, "object"): "run3_minimax_20260419",
    ("gpt5_4", 1, "method"): "gpt54_method_20260420",
    ("gpt5_4", 1, "object"): "gpt54_20260420",
}

# (model, run, kind) -> EvalSrc per mode. `None` key means same dir for both modes.
# `pred_filter` is the substring required in the CSV pred_file column; None disables filtering.
EVAL_SOURCES: dict[tuple[str, int, str], dict[str | None, EvalSrc]] = {
    ("gemma4_31b", 1, "method"): {None: EvalSrc("20260414_091231_method", "method_gemma4_all_20260410")},
    ("gemma4_31b", 1, "object"): {None: EvalSrc("20260412_193947", "object_gemma4_20260409")},
    ("gemma4_31b", 2, "method"): {None: EvalSrc("run2_gemma4_method_20260419", None)},
    ("gemma4_31b", 2, "object"): {None: EvalSrc("run2_gemma4_object_20260419", None)},
    ("gemma4_31b", 3, "method"): {None: EvalSrc("run3_gemma4_method_20260419", None)},
    ("gemma4_31b", 3, "object"): {None: EvalSrc("run3_gemma4_object_20260419", None)},
    ("qwen3_5_9b", 1, "method"): {None: EvalSrc("run1_qwen35_9b_method_20260419", None)},
    ("qwen3_5_9b", 1, "object"): {None: EvalSrc("run1_qwen35_9b_object_20260419", None)},
    ("qwen3_5_9b", 2, "method"): {None: EvalSrc("run2_qwen35_9b_method_20260419", None)},
    ("qwen3_5_9b", 2, "object"): {None: EvalSrc("run2_qwen35_9b_object_20260419", None)},
    ("qwen3_5_9b", 3, "method"): {None: EvalSrc("run3_qwen35_9b_method_20260419", None)},
    ("qwen3_5_9b", 3, "object"): {None: EvalSrc("run3_qwen35_9b_object_20260419", None)},
    ("qwen3_5_122b", 1, "method"): {None: EvalSrc("20260414_091231_method", "method_qwen35_all_20260410")},
    ("qwen3_5_122b", 1, "object"): {None: EvalSrc("20260412_193947", "object_qwen35_20260410")},
    ("qwen3_5_122b", 2, "method"): {None: EvalSrc("run2_qwen35_122b_method_20260419", None)},
    ("qwen3_5_122b", 2, "object"): {None: EvalSrc("run2_qwen35_122b_object_20260419", None)},
    ("qwen3_5_122b", 3, "method"): {None: EvalSrc("run3_qwen35_122b_method_20260419", None)},
    ("qwen3_5_122b", 3, "object"): {None: EvalSrc("run3_qwen35_122b_object_20260419", None)},
    ("minimax_m25", 1, "method"): {None: EvalSrc("20260414_091231_method", "method_minimax_20260410")},
    ("minimax_m25", 1, "object"): {
        "global": EvalSrc("20260415_133101", "object_minimax_20260413"),
        "per_paper": EvalSrc("20260415_200934", "object_minimax_20260413"),
    },
    ("minimax_m25", 2, "method"): {None: EvalSrc("run2_minimax_method_20260419", None)},
    ("minimax_m25", 2, "object"): {None: EvalSrc("run2_minimax_object_20260419", None)},
    ("minimax_m25", 3, "method"): {None: EvalSrc("run3_minimax_method_20260419", None)},
    ("minimax_m25", 3, "object"): {None: EvalSrc("run3_minimax_object_20260419", None)},
    ("gpt5_4", 1, "method"): {None: EvalSrc("gpt54_method_20260420", None)},
    ("gpt5_4", 1, "object"): {None: EvalSrc("gpt54_object_20260420", None)},
}

# Rescue patches: each rescue dir's rows override matching (model, run=1, kind, mode, domain, qid, step) records.
# The CSV's `experiment_folder` column identifies which base experiment is being patched; map back to (model, run, kind).
RESCUE_EXPERIMENT_TO_MODEL_RUN_KIND: dict[str, tuple[str, int, str]] = {
    "method_minimax_20260410": ("minimax_m25", 1, "method"),
    "method_qwen35_all_20260410": ("qwen3_5_122b", 1, "method"),
    "method_gemma4_all_20260410": ("gemma4_31b", 1, "method"),
    "object_minimax_20260413": ("minimax_m25", 1, "object"),
    "object_qwen35_20260413": ("qwen3_5_122b", 1, "object"),
    "object_gemma4_20260413": ("gemma4_31b", 1, "object"),
}

RESCUE_DIRS = ["rescue_r1_method_20260420", "rescue_r1_object_20260420"]

# Reagg overlays for per_paper object evaluations.
REAGG_DIR = "reagg_object_20260420"
REAGG_LABEL_TO_MODEL_RUN: dict[str, tuple[str, int]] = {
    "gemma4_run1": ("gemma4_31b", 1),
    "gemma4_run2": ("gemma4_31b", 2),
    "gemma4_run3": ("gemma4_31b", 3),
    "qwen122b_run1": ("qwen3_5_122b", 1),
    "qwen122b_run2": ("qwen3_5_122b", 2),
    "qwen122b_run3": ("qwen3_5_122b", 3),
    "gpt54": ("gpt5_4", 1),
}

# Fix overlays: post-hoc re-evaluations whose rows fully override matching entries.
# Each entry is (eval_dir_name, affected_kind) — it replaces any (model, 1, kind) rows
# it covers, identified by the experiment_folder→(model,run,kind) rescue map.
FIX_OVERLAYS: list[tuple[str, str]] = [
    # MC_M2.4 re-evaluation after GT was added (covers gemma4/minimax/qwen35_122b × 5 domains × 2 modes × 3 steps).
    ("20260417_160720", "method"),
    # Gemma4 run1 method per_paper social M2.6 fix (original was 0-pred).
    ("20260417_m26fix", "method"),
]

# Substring of the CSV pred_file column that uniquely identifies the model's prediction files.
MODEL_PRED_SUBSTRING: dict[str, str] = {
    "gemma4_31b": "google_gemma_4_31B_it",
    "qwen3_5_9b": "Qwen_Qwen3.5_9B",
    "qwen3_5_122b": "Qwen_Qwen3.5_122B_A10B_FP8",
    "minimax_m25": "MiniMaxAI_MiniMax_M2.5",
    "gpt5_4": "gpt_5.4_2026_03_05",
}

METRIC_FIELDS = [
    "pred_count", "gt_count", "paper_count",
    "matched_strict", "precision_strict", "recall_strict", "f1_strict",
    "matched_medium", "precision_medium", "recall_medium", "f1_medium",
    "matched_lenient", "precision_lenient", "recall_lenient", "f1_lenient",
    "avg_semantic_precision_lenient", "note",
]


def csv_mode_from_pred_file(pred_file: str) -> str:
    """'global_noimg'/'per_paper' segment in pred_file → 'global'/'per_paper'."""
    parts = pred_file.split("/")
    for seg in parts:
        if seg == "global_noimg":
            return "global"
        if seg == "per_paper":
            return "per_paper"
    raise ValueError(f"Cannot derive mode from {pred_file!r}")


def build_predictions_path(model: str, run: int, kind: str, mode: str, domain: str) -> Path:
    pred_dir = PREDICTIONS_DIR[(model, run, kind)]
    mode_seg = "global_noimg" if mode == "global" else "per_paper"
    pred_root = PRED_ROOT / pred_dir / mode_seg / domain
    if not pred_root.is_dir():
        return Path()
    # exactly one model_dir per domain
    model_sub = MODEL_PRED_SUBSTRING[model]
    candidates = [p for p in pred_root.iterdir() if p.is_dir() and model_sub in p.name]
    if len(candidates) != 1:
        return Path()
    return candidates[0] / "predictions.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def step_record(row: dict[str, str]) -> dict[str, str]:
    """Extract the step-level metrics block from a CSV row."""
    rec: dict[str, str] = {"domain": row["domain"]}
    for f in METRIC_FIELDS:
        rec[f] = row.get(f, "")
    return rec


# ---- Log parsing for per-paper breakdown ----

SEP_LINE = "=" * 80
QID_HEADER_RE = re.compile(r"^(M[12C]?_?[0-9.]+|O[12C]?_?[0-9.]+|MC_[A-Z0-9_.]+|OC_[A-Z0-9_.]+)\b")


def _parse_paper_breakdown(lines: list[str], start: int) -> tuple[list[dict[str, str | None]], int]:
    """Parse `paper | judge | pred | gt` table starting at `start`. Return (rows, next_index)."""
    # Expect header row at `start`: "paper | judge | pred | gt"
    if start >= len(lines) or "paper" not in lines[start] or "judge" not in lines[start]:
        return [], start
    i = start + 2  # skip header + separator row
    rows: list[dict[str, str | None]] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("=") or line.startswith("[Step"):
            break
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            break
        paper, judge, pred, gt = parts[0], parts[1], parts[2], parts[3]
        pred_v: str | None = None if pred == "—" else pred
        gt_v: str | None = None if gt == "—" else gt
        rows.append({"paper": paper, "judge": judge, "pred": pred_v, "gt": gt_v})
        i += 1
    return rows, i


def _parse_finallist_breakdown(lines: list[str], start: int) -> tuple[list[dict[str, str]], int]:
    """Parse `pred | gt | status` table (final_list step)."""
    if start >= len(lines) or "pred" not in lines[start] or "status" not in lines[start]:
        return [], start
    i = start + 2
    rows: list[dict[str, str]] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("=") or line.startswith("[Step"):
            break
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            break
        rows.append({"pred": parts[0], "gt": parts[1], "status": parts[2]})
        i += 1
    return rows, i


def parse_log_breakdowns(log_path: Path) -> dict[tuple[str, str], list[dict]]:
    """Parse a predictions.log. Returns {(qid, step): breakdown_rows}."""
    if not log_path.is_file():
        return {}
    with log_path.open(encoding="utf-8") as f:
        lines = f.read().splitlines()
    out: dict[tuple[str, str], list[dict]] = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() == SEP_LINE:
            i += 1
            if i >= len(lines):
                break
            header = lines[i].strip()
            i += 1
            m = QID_HEADER_RE.match(header)
            if not m:
                continue
            qid = m.group(1)
            if not qid.startswith("MC_") and not qid.startswith("OC_"):
                # normal qid: next lines are metric summary, then breakdown header
                while i < len(lines) and not lines[i].lstrip().startswith("paper"):
                    if lines[i].strip() == SEP_LINE:
                        break
                    i += 1
                if i < len(lines) and lines[i].lstrip().startswith("paper"):
                    br, i = _parse_paper_breakdown(lines, i)
                    out[(qid, "question")] = br
            else:
                # MC/OC: has [Step 1] evidences, [Step 2] final_list, [Step 3] final_answer
                while i < len(lines) and lines[i].strip() != SEP_LINE:
                    line = lines[i]
                    if line.startswith("[Step 1]"):
                        # evidences: breakdown is paper-level
                        j = i + 1
                        while j < len(lines) and not lines[j].lstrip().startswith("paper"):
                            if lines[j].strip() == SEP_LINE:
                                break
                            j += 1
                        if j < len(lines) and lines[j].lstrip().startswith("paper"):
                            br, j = _parse_paper_breakdown(lines, j)
                            out[(qid, "evidences")] = br
                        i = j
                    elif line.startswith("[Step 2]"):
                        # final_list: pred | gt | status
                        j = i + 1
                        while j < len(lines) and not lines[j].lstrip().startswith("pred"):
                            if lines[j].strip() == SEP_LINE:
                                break
                            j += 1
                        if j < len(lines) and lines[j].lstrip().startswith("pred"):
                            br, j = _parse_finallist_breakdown(lines, j)
                            out[(qid, "final_list")] = br
                        i = j
                    elif line.startswith("[Step 3]"):
                        i += 1
                    else:
                        i += 1
        else:
            i += 1
    return out


# ---- Core build ----

class Records:
    """In-memory (model, mode, qid, domain, run) -> record store."""

    def __init__(self) -> None:
        self.data: dict[tuple, dict] = {}

    def get_or_create(self, model: str, mode: str, qid: str, domain: str, run: int) -> dict:
        key = (model, mode, qid, domain, run)
        rec = self.data.get(key)
        if rec is None:
            rec = {
                "model": model,
                "run": f"run{run}",
                "mode": mode,
                "qid": qid,
                "domain": domain,
                "pred_file": "",
                "eval_dir": "",
                "steps": {},
            }
            self.data[key] = rec
        return rec


def derive_eval_source(model: str, run: int, kind: str, mode: str) -> EvalSrc:
    sources = EVAL_SOURCES[(model, run, kind)]
    if mode in sources:
        return sources[mode]
    return sources[None]


def ingest_csv_rows(
    records: Records,
    rows: list[dict[str, str]],
    model: str,
    run: int,
    eval_dir: str,
    kind_filter: str | None,
    pred_filter: str | None,
    mode_filter: str | None,
    eval_dir_label: str | None = None,
) -> int:
    """Add rows to records store. Returns number of rows ingested."""
    pred_root_str = str(PRED_ROOT) + "/"
    model_sub = MODEL_PRED_SUBSTRING[model]
    n = 0
    for row in rows:
        pf = row["pred_file"]
        if model_sub not in pf:
            continue
        if pred_filter and pred_filter not in pf:
            continue
        try:
            mode = csv_mode_from_pred_file(pf)
        except ValueError:
            continue
        if mode_filter and mode != mode_filter:
            continue
        qid = row["question_id"]
        if kind_filter == "method" and not qid.startswith("M"):
            continue
        if kind_filter == "object" and not qid.startswith("O"):
            continue
        step = row["step"]
        domain = row["domain"]
        rec = records.get_or_create(model, mode, qid, domain, run)
        # Normalize pred_file to pred_root-relative path, rewritten to current predictions dir.
        kind = "method" if qid.startswith("M") else "object"
        pred_path = build_predictions_path(model, run, kind, mode, domain)
        rec["pred_file"] = str(pred_path) if pred_path != Path() else pf.replace(pred_root_str, "")
        label = eval_dir_label or eval_dir
        if rec["eval_dir"] and label not in rec["eval_dir"].split("+"):
            rec["eval_dir"] = f"{rec['eval_dir']}+{label}"
        else:
            rec["eval_dir"] = label
        rec["steps"][step] = step_record(row)
        n += 1
    return n


def attach_breakdowns_for_eval(records: Records, model: str, run: int, eval_dir: str) -> None:
    """Parse .log files in `eval_dir` and attach breakdowns to matching records."""
    eval_path = EVAL_ROOT / eval_dir
    if not eval_path.is_dir():
        return
    model_sub = MODEL_PRED_SUBSTRING[model]
    for log in sorted(eval_path.glob("*.log")):
        name = log.name
        if model_sub not in name:
            continue
        # Filename pattern: <exp>__<mode>__<domain>__<model_dir>__predictions.log
        # Some eval dirs strip <exp>__ prefix — split from the known suffix.
        stem = name[:-len("__predictions.log")] if name.endswith("__predictions.log") else log.stem
        parts = stem.split("__")
        # Find the mode_seg index
        mode_idx = None
        for idx, seg in enumerate(parts):
            if seg in ("global_noimg", "per_paper"):
                mode_idx = idx
                break
        if mode_idx is None or mode_idx + 1 >= len(parts):
            continue
        mode_seg = parts[mode_idx]
        domain = parts[mode_idx + 1]
        mode = "global" if mode_seg == "global_noimg" else "per_paper"
        breakdowns = parse_log_breakdowns(log)
        if not breakdowns:
            continue
        for (qid, step), br in breakdowns.items():
            key = (model, mode, qid, domain, run)
            rec = records.data.get(key)
            if rec is None:
                continue
            step_rec = rec["steps"].get(step)
            if step_rec is None:
                continue
            step_rec["breakdown"] = br


def apply_rescue_patches(records: Records) -> None:
    for rescue_name in RESCUE_DIRS:
        csv_path = EVAL_ROOT / rescue_name / "all_metrics.csv"
        if not csv_path.is_file():
            continue
        rows = load_csv(csv_path)
        for row in rows:
            exp = row["experiment_folder"]
            if exp not in RESCUE_EXPERIMENT_TO_MODEL_RUN_KIND:
                continue
            model, run, kind = RESCUE_EXPERIMENT_TO_MODEL_RUN_KIND[exp]
            # Single-row ingest via the normal path, tagging eval_dir with rescue name.
            ingest_csv_rows(
                records, [row], model, run,
                eval_dir=rescue_name, kind_filter=kind, pred_filter=None,
                mode_filter=None, eval_dir_label=f"rescue:{rescue_name}",
            )
        # Attach any breakdowns from rescue logs too.
        for (model, run, _kind) in set(RESCUE_EXPERIMENT_TO_MODEL_RUN_KIND.values()):
            attach_breakdowns_for_eval(records, model, run, rescue_name)


def apply_fix_overlays(records: Records) -> None:
    """Apply post-hoc fix overlays (e.g., MC_M2.4 re-eval after GT backfill).

    For every (model, mode, qid, domain, run) record touched by the fix, all
    existing steps are discarded before re-ingesting — a fix supersedes the
    stale GT-missing or zero-pred placeholder the base CSV wrote.
    """
    for fix_dir, kind in FIX_OVERLAYS:
        csv_path = EVAL_ROOT / fix_dir / "all_metrics.csv"
        if not csv_path.is_file():
            continue
        rows = load_csv(csv_path)
        affected_models: set[tuple[str, int, str]] = set()
        cleared: set[tuple] = set()
        for row in rows:
            exp = row["experiment_folder"]
            if exp not in RESCUE_EXPERIMENT_TO_MODEL_RUN_KIND:
                continue
            model, run, rkind = RESCUE_EXPERIMENT_TO_MODEL_RUN_KIND[exp]
            if rkind != kind:
                continue
            affected_models.add((model, run, rkind))
            try:
                mode = csv_mode_from_pred_file(row["pred_file"])
            except ValueError:
                continue
            qid = row["question_id"]
            key = (model, mode, qid, row["domain"], run)
            if key not in cleared:
                rec = records.data.get(key)
                if rec is not None:
                    rec["steps"].clear()
                cleared.add(key)
            ingest_csv_rows(
                records, [row], model, run,
                eval_dir=fix_dir, kind_filter=kind, pred_filter=None,
                mode_filter=None, eval_dir_label=f"fix:{fix_dir}",
            )
        for (model, run, _kind) in affected_models:
            attach_breakdowns_for_eval(records, model, run, fix_dir)


def apply_reagg_overlays(records: Records) -> None:
    base = EVAL_ROOT / REAGG_DIR
    if not base.is_dir():
        return
    for label, (model, run) in REAGG_LABEL_TO_MODEL_RUN.items():
        sub = base / label
        csv_path = sub / "all_metrics.csv"
        if not csv_path.is_file():
            continue
        rows = load_csv(csv_path)
        # reagg is per_paper object only — drop any matching records first to avoid stale breakdowns.
        to_drop = [k for k in records.data
                   if k[0] == model and k[1] == "per_paper" and k[4] == run and k[2].startswith("O")]
        for k in to_drop:
            records.data.pop(k, None)
        label_full = f"{REAGG_DIR}/{label}"
        ingest_csv_rows(
            records, rows, model, run,
            eval_dir=label_full, kind_filter="object", pred_filter=None,
            mode_filter="per_paper", eval_dir_label=label_full,
        )
        # Re-parse breakdowns within the reagg subdir.
        attach_breakdowns_for_eval(records, model, run, f"{REAGG_DIR}/{label}")


def load_prediction_payload(pred_file: Path, qid: str) -> dict | None:
    if not pred_file.is_file():
        return None
    try:
        with pred_file.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if qid in data:
        return data[qid]
    return None


def write_out(records: Records, out_root: Path) -> int:
    n = 0
    for (model, mode, qid, domain, run), rec in records.data.items():
        dest = out_root / model / mode / qid / domain / f"run{run}"
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        # Prediction payload: load this qid from the predictions.json in the matching pred dir.
        pred_root_is_abs = rec["pred_file"].startswith("/")
        pred_path = Path(rec["pred_file"]) if pred_root_is_abs else (PRED_ROOT / rec["pred_file"])
        payload = load_prediction_payload(pred_path, qid)
        if payload is not None:
            with (dest / "prediction.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        n += 1
    return n


def build(out_root: Path) -> Records:
    records = Records()
    # Base pass
    for (model, run, kind), sources in EVAL_SOURCES.items():
        # Collect all (mode_key, EvalSrc) to process; mode_key=None means both modes from the same dir
        seen_dirs: set[tuple[str, str | None]] = set()
        for mode_key, src in sources.items():
            csv_path = EVAL_ROOT / src.eval_dir / "all_metrics.csv"
            if not csv_path.is_file():
                print(f"  missing CSV: {csv_path}", file=sys.stderr)
                continue
            rows = load_csv(csv_path)
            n = ingest_csv_rows(
                records, rows, model, run,
                eval_dir=src.eval_dir, kind_filter=kind,
                pred_filter=src.pred_filter,
                mode_filter=mode_key,
            )
            if (src.eval_dir, mode_key) not in seen_dirs:
                attach_breakdowns_for_eval(records, model, run, src.eval_dir)
                seen_dirs.add((src.eval_dir, mode_key))
            print(f"  {model} run{run} {kind} mode={mode_key or 'both'} from {src.eval_dir}: {n} rows")

    # Patch layers (apply in escalating recency order)
    apply_rescue_patches(records)
    apply_fix_overlays(records)
    apply_reagg_overlays(records)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_ROOT),
                    help="Output directory (default: %(default)s)")
    ap.add_argument("--clean", action="store_true",
                    help="Remove existing output directory before building")
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.clean and out_root.exists():
        print(f"Removing {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    records = build(out_root)
    n = write_out(records, out_root)
    print(f"\nWrote {n} record dirs under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
