#!/usr/bin/env python3
"""Batch evaluator for predictions.json files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def _load_run_eval() -> Any:
    module_path = Path(__file__).with_name("run_eval.py")
    spec = importlib.util.spec_from_file_location("batch_run_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["batch_run_eval"] = module
    spec.loader.exec_module(module)
    return module


def _apply_vllm_port_from_argv() -> None:
    """Pre-parse --port so EVAL_LLM_BASE_URL is set before run_eval is imported."""
    port: int | None = None
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                pass
            break
        if arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError:
                pass
            break
    if port is not None:
        os.environ["EVAL_LLM_BASE_URL"] = f"http://localhost:{port}/v1"
    elif "EVAL_LLM_BASE_URL" not in os.environ:
        os.environ["EVAL_LLM_BASE_URL"] = "http://localhost:8100/v1"


_apply_vllm_port_from_argv()
run_eval = _load_run_eval()

FIELD_MAP = {
    "O1.1": ("object", "geolocation"),
    "O1.2": ("object", "sample_size"),
    "O1.3": ("object", "study_population"),
    "O2.1": ("object", ["study_population", "sample_size"]),
    "O2.2": ("object", ["study_population", "geolocation"]),
    "O2.3": ("object", ["study_population", "geolocation", "sample_size"]),
    "M1.1": ("method", "statistical_method"),
    "M1.2": ("method", "variable"),
    "M1.3": ("method", "independent_variable"),
    "M1.4": ("method", "dependent_variable"),
    "M2.1": ("method", ["variable", "unit"]),
    "M2.2": ("method", ["independent_variable", "unit"]),
    "M2.3": ("method", ["dependent_variable", "unit"]),
    "M2.4": ("method", ["independent_variable", "dependent_variable"]),
    "M2.5": ("method", ["independent_variable", "dependent_variable", "statistical_method"]),
    "M2.6": ("method", ["independent_variable", "dependent_variable", "statistical_method", "conditions", "effect_size"]),
}

COMP_CONFIG = {
    "OC_O1.1": {"gt_type": "object", "evidence_fields": "geolocation", "agg_mode": "count"},
    "OC_O1.2.1": {"gt_type": "object", "evidence_fields": "sample_size", "agg_mode": "count"},
    "OC_O1.2.2": {"gt_type": "object", "evidence_fields": "sample_size", "agg_mode": "avg"},
    "OC_O1.2.3": {"gt_type": "object", "evidence_fields": "sample_size", "agg_mode": "median"},
    "MC_M1.1": {"gt_type": "method", "evidence_fields": "statistical_method", "agg_mode": "count"},
    "MC_M1.2": {"gt_type": "method", "evidence_fields": "variable", "agg_mode": "count"},
    "MC_M1.3": {"gt_type": "method", "evidence_fields": "independent_variable", "agg_mode": "count"},
    "MC_M1.4": {"gt_type": "method", "evidence_fields": "dependent_variable", "agg_mode": "count"},
    "MC_M2.4": {"gt_type": "method", "evidence_fields": ["independent_variable", "dependent_variable"], "agg_mode": "count"},
    "MC_M2.6": {"gt_type": "method", "evidence_fields": ["independent_variable", "dependent_variable", "statistical_method", "conditions", "effect_size"], "agg_mode": "count"},
}

PM_TO_GT = {
    "biodiversity": "1_biodiversity",
    "hotel": "2_engineering",
    "myopia": "3_myopia",
    "health": "3_myopia",
    "agriculture": "4_agriculture",
    "water": "4_agriculture",
    "social": "5_social",
}


def make_pbar(iterable, total: int | None = None, desc: str = "", leave: bool = False, position: int = 0):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        position=position,
        dynamic_ncols=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate predictions against ground truth.")
    parser.add_argument("--pred-root", default="/mlde/ss/data/results/predictions")
    parser.add_argument("--pred-file", action="append", default=[], help="Specific predictions.json to evaluate.")
    parser.add_argument("--folders", nargs="*", default=None, help="Top-level experiment folders under pred-root.")
    parser.add_argument("--pred-glob", default="**/predictions.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--questions", nargs="*", default=None)
    parser.add_argument("--paper-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--max-file-workers", type=int, default=1,
                        help="Number of prediction files to evaluate in parallel (default: 1 = sequential)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list-folders", action="store_true")
    parser.add_argument("--port", type=int, default=8100,
                        help="vLLM backend port for the eval LLM judge (default: 8100). "
                             "Sets EVAL_LLM_BASE_URL to http://localhost:<port>/v1.")
    return parser.parse_args()


def flatten_preds(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {item["question_id"]: item.get("answers", item) for item in raw if isinstance(item, dict) and "question_id" in item}
    return {}


def load_gt_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return {item["question_id"]: item for item in json.load(f)}


def infer_domain(pred_file: Path, pred_root: Path) -> str | None:
    try:
        rel_parts = pred_file.relative_to(pred_root).parts
    except ValueError:
        rel_parts = pred_file.parts
    # Exact-match pass (old layout, where a path part is literally the domain key).
    for part in rel_parts:
        if part in PM_TO_GT:
            return part
    # Prefix-match pass (new flat layout, e.g., "hotel_qwen_qwen3.5_9b_per_paper").
    for part in rel_parts:
        for key in PM_TO_GT:
            if part.startswith(key + "_") or part == key:
                return key
    return None


def infer_run_kind(pred_file: Path, pred_root: Path) -> str:
    try:
        first = pred_file.relative_to(pred_root).parts[0]
    except ValueError:
        first = pred_file.parts[-4] if len(pred_file.parts) >= 4 else pred_file.parent.name
    if first.startswith("object_"):
        return "object"
    if first.startswith("method_"):
        return "method"
    return "mixed"


def sanitize_relpath(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return "__".join(rel.parts[:-1] + (rel.stem,))


def list_experiment_folders(pred_root: Path) -> list[Path]:
    return sorted(
        path for path in pred_root.iterdir()
        if path.is_dir() and path.name.startswith(("object_", "method_"))
    )


def choose_experiment_folders(pred_root: Path, requested: list[str] | None, list_only: bool) -> list[Path]:
    folders = list_experiment_folders(pred_root)
    if not folders:
        raise SystemExit(f"No object_/method_ folders found under {pred_root}")

    print("Available prediction folders:")
    for idx, folder in enumerate(folders, start=1):
        print(f"  [{idx}] {folder.name}")

    if list_only:
        raise SystemExit(0)

    if requested is None:
        raw = input("Select folders by number or name, comma-separated; or 'all': ").strip()
        requested = [part.strip() for part in raw.split(",") if part.strip()] if raw else ["all"]

    if not requested or any(item.lower() == "all" for item in requested):
        return folders

    by_name = {folder.name: folder for folder in folders}
    selected: list[Path] = []
    seen: set[str] = set()
    for item in requested:
        if item.isdigit():
            idx = int(item)
            if idx < 1 or idx > len(folders):
                raise SystemExit(f"Invalid folder index: {item}")
            folder = folders[idx - 1]
        else:
            folder = by_name.get(item)
            if folder is None:
                raise SystemExit(f"Unknown folder: {item}")
        if folder.name not in seen:
            selected.append(folder)
            seen.add(folder.name)
    return selected


def get_gt_rows(qid: str, obj_data: dict[str, Any], mth_data: dict[str, Any], filter_paper_ids: set[int] | None) -> list[dict[str, Any]]:
    gt_type, _ = FIELD_MAP.get(qid, ("object", None))
    src = obj_data if gt_type == "object" else mth_data
    if qid not in src:
        return []
    answers = src[qid].get("answers", [])
    if isinstance(answers, dict):
        return [answers]
    rows = run_eval.normalize_c_answers(answers)
    if filter_paper_ids:
        rows = [r for r in rows if run_eval.normalize_paper_id(r.get("paper_id")) in filter_paper_ids]
    return rows


def get_pred_rows(qid: str, preds: dict[str, Any], filter_paper_ids: set[int] | None) -> list[dict[str, Any]]:
    raw = preds.get(qid)
    if raw is None:
        return []
    rows = run_eval.normalize_pred_c_answers(raw)
    if filter_paper_ids:
        rows = [r for r in rows if run_eval.normalize_paper_id(r.get("paper_id")) in filter_paper_ids]
    return rows


def _field_value_to_text(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}={run_eval.strip_citations(str(v))}" for k, v in value.items())
    if isinstance(value, list):
        return "; ".join(_field_value_to_text(v) for v in value)
    return run_eval.strip_citations(str(value))


def fmt_val(row: dict[str, Any], fields: str | list[str]) -> str:
    if isinstance(fields, list):
        return "; ".join(f"{field}={run_eval.strip_citations(str(run_eval.get_row_field_value(row, field)))}" for field in fields)
    return run_eval.strip_citations(str(run_eval.get_row_field_value(row, fields)))


def run_judge(pred_row: dict[str, Any], gt_row: dict[str, Any], fields: str | list[str]) -> tuple[bool, float, str]:
    if isinstance(fields, list):
        pred_text = "; ".join(f"{field}={run_eval.strip_citations(str(run_eval.get_row_field_value(pred_row, field)))}" for field in fields)
        gt_text = "; ".join(f"{field}={run_eval.strip_citations(str(run_eval.get_row_field_value(gt_row, field)))}" for field in fields)
        ctx = ", ".join(fields)
    else:
        pred_text = run_eval.strip_citations(str(run_eval.get_row_field_value(pred_row, fields)))
        gt_text = run_eval.strip_citations(str(run_eval.get_row_field_value(gt_row, fields)))
        if fields == "sample_size":
            pred_num = run_eval.parse_floatish(pred_text)
            gt_num = run_eval.parse_floatish(gt_text)
            if pred_num is not None and gt_num is not None:
                matched = round(pred_num, 4) == round(gt_num, 4)
                return matched, 1.0 if matched else 0.0, "EXACT" if matched else "NO"
        ctx = fields
    is_match, match_quality, kind = run_eval.llm_match_3tier(pred_text, gt_text, ctx)
    return is_match, match_quality, run_eval.KIND_TO_LABEL[kind]


# ── Strictness tiers ────────────────────────────────────────────────
# strict : only EXACT counts
# medium : EXACT + OVER_DETAIL
# lenient: EXACT + OVER_DETAIL + LESS_DETAIL
TIER_LABELS: dict[str, set[str]] = {
    "strict":  {"EXACT"},
    "medium":  {"EXACT", "OVER_DETAIL"},
    "lenient": {"EXACT", "OVER_DETAIL", "LESS_DETAIL"},
}
TIER_ORDER: tuple[str, ...] = ("strict", "medium", "lenient")
_LABEL_TIER: dict[str, int] = {"EXACT": 0, "OVER_DETAIL": 1, "LESS_DETAIL": 2, "NO": 3}


def _greedy_match_global(
    candidates: list[tuple[int, int]],
    judge_cache: dict[tuple[int, int], tuple[bool, float, str]],
    allowed_labels: set[str],
) -> tuple[set[int], set[int], dict[int, int], float, int]:
    ranked = sorted(
        candidates,
        key=lambda pair: (
            _LABEL_TIER.get(judge_cache[pair][2], 3),
            -judge_cache[pair][1],
            pair[0],
            pair[1],
        ),
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    match_map: dict[int, int] = {}
    prec_sum = 0.0
    matched = 0
    for pi, gi in ranked:
        _is_m, prec, lbl = judge_cache[(pi, gi)]
        if lbl not in allowed_labels:
            continue
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        match_map[pi] = gi
        prec_sum += prec
        matched += 1
    return used_p, used_g, match_map, prec_sum, matched


def _greedy_match_in_paper(
    pair_cands: list[tuple[int, int]],
    judge_cache: dict[tuple[Any, int, int], tuple[bool, float, str]],
    pid: Any,
    allowed_labels: set[str],
) -> tuple[set[int], set[int], dict[int, int]]:
    ranked = sorted(
        pair_cands,
        key=lambda pair: (
            _LABEL_TIER.get(judge_cache[(pid, pair[0], pair[1])][2], 3),
            -judge_cache[(pid, pair[0], pair[1])][1],
            pair[0],
            pair[1],
        ),
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    match_map: dict[int, int] = {}
    for pi, gi in ranked:
        _is_m, _prec, lbl = judge_cache[(pid, pi, gi)]
        if lbl not in allowed_labels:
            continue
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        match_map[pi] = gi
    return used_p, used_g, match_map


def _paper_avg_prf1(
    pred_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    used_pred: set[int],
) -> tuple[float, float, float, int]:
    pred_by_paper: dict[Any, list[int]] = {}
    for pi, pr in enumerate(pred_rows):
        pid = run_eval.normalize_paper_id(pr.get("paper_id"))
        pred_by_paper.setdefault(pid, []).append(pi)
    gt_by_paper: dict[Any, list[int]] = {}
    for gi, gr in enumerate(gt_rows):
        pid = run_eval.normalize_paper_id(gr.get("paper_id"))
        gt_by_paper.setdefault(pid, []).append(gi)
    matched_by_paper: dict[Any, int] = {}
    for pi in used_pred:
        pid = run_eval.normalize_paper_id(pred_rows[pi].get("paper_id"))
        matched_by_paper[pid] = matched_by_paper.get(pid, 0) + 1

    all_pids = set(pred_by_paper.keys()) | set(gt_by_paper.keys())
    ps, rs, fs = [], [], []
    for pid in all_pids:
        n_pred_p = len(pred_by_paper.get(pid, []))
        n_gt_p = len(gt_by_paper.get(pid, []))
        n_matched_p = matched_by_paper.get(pid, 0)
        p, r, f = prf1(n_matched_p, n_pred_p, n_gt_p)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    n = len(all_pids)
    return (
        sum(ps) / n if n else 0.0,
        sum(rs) / n if n else 0.0,
        sum(fs) / n if n else 0.0,
        n,
    )


def pid_sort_key(pid: Any) -> tuple[int, Any]:
    if pid is None:
        return (1, 0)
    try:
        return (0, int(pid))
    except Exception:
        return (0, str(pid))


def prf1(matched: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = min(max(widths[col], len(str(row.get(col, "")))), 120)
    sep = "-+-".join("-" * widths[col] for col in columns)
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    body = []
    for row in rows:
        body.append(" | ".join(str(row.get(col, ""))[: widths[col]].ljust(widths[col]) for col in columns))
    return "\n".join([header, sep] + body)


def comp_evidences_table(
    pred_ev: list[Any],
    gt_ev: list[Any],
    ev_fields: str | list[str],
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int]:
    """Per-paper × per-item matching with 3-tier strictness.
    Returns (display_rows_using_lenient_match, tier_stats, n_papers).
    tier_stats[tier] = {matched_items, n_pred_items, n_gt_items,
                        macro_p, macro_r, macro_f1, n_papers}
    """
    fields = ev_fields
    scalar_key = ev_fields if isinstance(ev_fields, str) else "value"

    def to_items(ev_item: Any) -> list[dict[str, Any]]:
        if not ev_item:
            return []
        ans = ev_item.get("answer", [])
        if not isinstance(ans, list):
            ans = [ans]
        out: list[dict[str, Any]] = []
        for a in ans:
            row: dict[str, Any] = dict(a) if isinstance(a, dict) else {scalar_key: a}
            out.append(row)
        return out

    per_paper: dict[Any, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for ev_item in pred_ev or []:
        pid = run_eval.normalize_paper_id(ev_item.get("paper_id"))
        per_paper.setdefault(pid, ([], []))[0].extend(to_items(ev_item))
    for ev_item in gt_ev or []:
        pid = run_eval.normalize_paper_id(ev_item.get("paper_id"))
        per_paper.setdefault(pid, ([], []))[1].extend(to_items(ev_item))

    candidates_by_paper: dict[Any, list[tuple[int, int]]] = {}
    all_candidates: list[tuple[Any, int, int]] = []
    for pid, (pr_items, gt_items) in per_paper.items():
        pairs = [(pi, gi) for pi in range(len(pr_items)) for gi in range(len(gt_items))]
        candidates_by_paper[pid] = pairs
        for pi, gi in pairs:
            all_candidates.append((pid, pi, gi))

    def _judge(args: tuple[Any, int, int]) -> tuple[Any, int, int, bool, float, str]:
        pid, pi, gi = args
        pr = per_paper[pid][0][pi]
        gr = per_paper[pid][1][gi]
        is_m, prec, lbl = run_judge(pr, gr, fields)
        return pid, pi, gi, is_m, prec, lbl

    judge_cache: dict[tuple[Any, int, int], tuple[bool, float, str]] = {}
    if all_candidates:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for pid, pi, gi, is_m, prec, lbl in ex.map(_judge, all_candidates):
                judge_cache[(pid, pi, gi)] = (is_m, prec, lbl)

    # Per-tier macro metrics (single LLM pass; all tiers reuse judge_cache)
    tier_stats: dict[str, dict[str, Any]] = {}
    for tier_name in TIER_ORDER:
        allowed = TIER_LABELS[tier_name]
        total_matched = 0
        total_np = 0
        total_ng = 0
        paper_prf: list[tuple[float, float, float]] = []
        for pid, (pr_items, gt_items) in per_paper.items():
            np_, ng_ = len(pr_items), len(gt_items)
            _up, _ug, mmap = _greedy_match_in_paper(
                candidates_by_paper[pid], judge_cache, pid, allowed
            )
            m = len(mmap)
            total_matched += m
            total_np += np_
            total_ng += ng_
            p, r, f = prf1(m, np_, ng_)
            paper_prf.append((p, r, f))
        n = len(paper_prf)
        macro_p = sum(x[0] for x in paper_prf) / n if n else 0.0
        macro_r = sum(x[1] for x in paper_prf) / n if n else 0.0
        macro_f1 = sum(x[2] for x in paper_prf) / n if n else 0.0
        tier_stats[tier_name] = {
            "matched_items": total_matched,
            "n_pred_items": total_np,
            "n_gt_items": total_ng,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f1": macro_f1,
            "n_papers": n,
        }

    # Build display rows using lenient matching (shows the most pairs)
    rows: list[dict[str, Any]] = []
    for pid in sorted(per_paper.keys(), key=pid_sort_key):
        pr_items, gt_items = per_paper[pid]
        used_p, used_g, match_map = _greedy_match_in_paper(
            candidates_by_paper[pid], judge_cache, pid, TIER_LABELS["lenient"]
        )
        for pi, gi in match_map.items():
            pr = pr_items[pi]
            gr = gt_items[gi]
            _, _, lbl = judge_cache[(pid, pi, gi)]
            rows.append({
                "paper": pid,
                "judge": lbl,
                "pred": fmt_val(pr, fields),
                "gt": fmt_val(gr, fields),
            })
        for pi, pr in enumerate(pr_items):
            if pi not in used_p:
                rows.append({
                    "paper": pid,
                    "judge": "EXTRA",
                    "pred": fmt_val(pr, fields),
                    "gt": "—",
                })
        for gi, gr in enumerate(gt_items):
            if gi not in used_g:
                rows.append({
                    "paper": pid,
                    "judge": "MISSING",
                    "pred": "—",
                    "gt": fmt_val(gr, fields),
                })

    return rows, tier_stats, len(per_paper)


def normalize_list_val(value: Any) -> str | None:
    if isinstance(value, dict):
        parts = [
            f"{k}={run_eval.strip_citations(str(v))}"
            for k, v in value.items()
            if k != "paper_id"
        ]
        text = "; ".join(parts)
    else:
        text = run_eval.strip_citations(str(value)).strip().lower()
    text = text.strip().lower()
    return text or None


def comp_list_table(
    pred_list: list[Any],
    gt_list: list[Any],
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """final_list comparison with 3-tier strictness.
    Returns (display_rows_using_lenient_match, tier_stats).
    tier_stats[tier] = {matched, n_pred, n_gt, precision, recall, f1}
    """
    pred_items = [v for v in (normalize_list_val(x) for x in (pred_list or [])) if v]
    gt_items = [v for v in (normalize_list_val(x) for x in (gt_list or [])) if v]

    pred_set = set(pred_items)
    gt_set = set(gt_items)
    exact_common = pred_set & gt_set

    unmatched_pred = [v for v in pred_items if v not in exact_common]
    unmatched_gt = [v for v in gt_items if v not in exact_common]

    fuzzy_results: list[tuple[str, str, bool, float, str]] = []
    if unmatched_pred and unmatched_gt:
        pairs = [(pv, gv) for pv in unmatched_pred for gv in unmatched_gt]

        def judge_pair(args: tuple[str, str]) -> tuple[str, str, bool, float, str]:
            pv, gv = args
            is_m, mq, kind = run_eval.llm_match_3tier(pv, gv, "value")
            return pv, gv, is_m, mq, kind

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fuzzy_results = list(ex.map(judge_pair, pairs))

    def _tier_match(allowed: set[str]) -> dict[str, str]:
        pred_to_gt: dict[str, str] = {v: v for v in exact_common}
        used_p: set[str] = set(exact_common)
        used_g: set[str] = set(exact_common)
        ranked = sorted(
            fuzzy_results,
            key=lambda t: (
                _LABEL_TIER.get(run_eval.KIND_TO_LABEL.get(t[4], "NO"), 3),
                -t[3],
                t[0],
                t[1],
            ),
        )
        for pv, gv, _ism, _prec, kind in ranked:
            lbl = run_eval.KIND_TO_LABEL.get(kind, "NO")
            if lbl not in allowed:
                continue
            if pv in used_p or gv in used_g:
                continue
            pred_to_gt[pv] = gv
            used_p.add(pv)
            used_g.add(gv)
        return pred_to_gt

    np_total = len(pred_items)
    ng_total = len(gt_items)
    tier_stats: dict[str, dict[str, Any]] = {}
    for tier_name in TIER_ORDER:
        p2g = _tier_match(TIER_LABELS[tier_name])
        matched = len(p2g)
        p, r, f = prf1(matched, np_total, ng_total)
        tier_stats[tier_name] = {
            "matched": matched,
            "n_pred": np_total,
            "n_gt": ng_total,
            "precision": p,
            "recall": r,
            "f1": f,
        }

    # Build display rows using lenient matching
    lenient_p2g = _tier_match(TIER_LABELS["lenient"])
    fuzzy_kind: dict[tuple[str, str], str] = {}
    for pv, gv, _ism, _prec, kind in fuzzy_results:
        fuzzy_kind[(pv, gv)] = kind

    rows: list[dict[str, Any]] = []
    for value in sorted(exact_common):
        rows.append({"pred": value, "gt": value, "status": "EXACT"})
    fuzzy_lenient_items = [(pv, gv) for pv, gv in lenient_p2g.items() if pv not in exact_common]
    for pv, gv in sorted(fuzzy_lenient_items):
        kind = fuzzy_kind.get((pv, gv), "no_match")
        lbl = run_eval.KIND_TO_LABEL.get(kind, "NO")
        rows.append({"pred": pv, "gt": gv, "status": lbl})
    matched_lenient_gt = {gv for _pv, gv in lenient_p2g.items()}
    for value in sorted(pred_set - set(lenient_p2g.keys())):
        rows.append({"pred": value, "gt": "—", "status": "EXTRA"})
    for value in sorted(gt_set - matched_lenient_gt):
        rows.append({"pred": "—", "gt": value, "status": "MISSING"})

    return rows, tier_stats


def evaluate_standard_question(
    qid: str,
    preds: dict[str, Any],
    obj_data: dict[str, Any],
    mth_data: dict[str, Any],
    filter_paper_ids: set[int] | None,
    max_workers: int,
) -> tuple[dict[str, Any], list[str]]:
    _, fields = FIELD_MAP[qid]
    pred_rows = get_pred_rows(qid, preds, filter_paper_ids)
    gt_rows = get_gt_rows(qid, obj_data, mth_data, filter_paper_ids)

    candidates = []
    for pred_idx, pred_row in enumerate(pred_rows):
        for gt_idx, gt_row in enumerate(gt_rows):
            pred_pid = run_eval.normalize_paper_id(pred_row.get("paper_id"))
            gt_pid = run_eval.normalize_paper_id(gt_row.get("paper_id"))
            if pred_pid is not None and gt_pid is not None and pred_pid != gt_pid:
                continue
            candidates.append((pred_idx, gt_idx))

    judge_cache: dict[tuple[int, int], tuple[bool, float, str]] = {}

    def judge_pair(args: tuple[int, int]) -> tuple[int, int, bool, float, str]:
        pred_idx, gt_idx = args
        is_match, precision, label = run_judge(pred_rows[pred_idx], gt_rows[gt_idx], fields)
        return pred_idx, gt_idx, is_match, precision, label

    if candidates:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for pred_idx, gt_idx, is_match, precision, label in ex.map(judge_pair, candidates):
                judge_cache[(pred_idx, gt_idx)] = (is_match, precision, label)

    # Per-tier greedy match + paper-averaged P/R/F1
    tier_metrics: dict[str, dict[str, Any]] = {}
    for tier_name in TIER_ORDER:
        allowed = TIER_LABELS[tier_name]
        used_p_t, _ug_t, _mm_t, prec_sum_t, matched_t = _greedy_match_global(
            candidates, judge_cache, allowed
        )
        p, r, f, n_papers_t = _paper_avg_prf1(pred_rows, gt_rows, used_p_t)
        tier_metrics[tier_name] = {
            "matched": matched_t,
            "precision": p,
            "recall": r,
            "f1": f,
            "avg_semantic_precision": (prec_sum_t / matched_t) if matched_t else None,
            "n_papers": n_papers_t,
        }

    # Display rows use lenient matching (shows the most pairs)
    used_pred, used_gt, match_map, _ps_len, _m_len = _greedy_match_global(
        candidates, judge_cache, TIER_LABELS["lenient"]
    )
    rows = []
    for pred_idx, gt_idx in match_map.items():
        pred_row = pred_rows[pred_idx]
        gt_row = gt_rows[gt_idx]
        pid = run_eval.normalize_paper_id(pred_row.get("paper_id"))
        _is_match, _precision, label = judge_cache[(pred_idx, gt_idx)]
        rows.append({"paper": pid, "judge": label, "pred": fmt_val(pred_row, fields), "gt": fmt_val(gt_row, fields)})
    for pred_idx, pred_row in enumerate(pred_rows):
        if pred_idx not in used_pred:
            rows.append({"paper": run_eval.normalize_paper_id(pred_row.get("paper_id")), "judge": "EXTRA", "pred": fmt_val(pred_row, fields), "gt": "—"})
    for gt_idx, gt_row in enumerate(gt_rows):
        if gt_idx not in used_gt:
            rows.append({"paper": run_eval.normalize_paper_id(gt_row.get("paper_id")), "judge": "MISSING", "pred": "—", "gt": fmt_val(gt_row, fields)})
    rows.sort(key=lambda row: pid_sort_key(row["paper"]))

    n_papers = tier_metrics["lenient"]["n_papers"]
    summary = {
        "question_id": qid,
        "step": "question",
        "pred_count": len(pred_rows),
        "gt_count": len(gt_rows),
        "paper_count": n_papers,
        "note": "paper-avg",
    }
    for tier_name in TIER_ORDER:
        tm = tier_metrics[tier_name]
        summary[f"matched_{tier_name}"] = tm["matched"]
        summary[f"precision_{tier_name}"] = tm["precision"]
        summary[f"recall_{tier_name}"] = tm["recall"]
        summary[f"f1_{tier_name}"] = tm["f1"]
    summary["avg_semantic_precision_lenient"] = tier_metrics["lenient"]["avg_semantic_precision"]

    log_lines = ["=" * 80, f"{qid}  papers={n_papers}  pred={len(pred_rows)}  gt={len(gt_rows)}"]
    for tier_name in TIER_ORDER:
        tm = tier_metrics[tier_name]
        log_lines.append(
            f"  {tier_name:<7s}: matched={tm['matched']:>3d}  "
            f"P={tm['precision']:.4f}  R={tm['recall']:.4f}  F1={tm['f1']:.4f}  (paper-avg)"
        )
    log_lines.append(render_table(rows, ["paper", "judge", "pred", "gt"]))
    log_lines.append("")
    return summary, log_lines


def _empty_tier_summary(qid: str, step: str, note: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": qid,
        "step": step,
        "pred_count": 0,
        "gt_count": 0,
        "paper_count": 0,
        "avg_semantic_precision_lenient": None,
        "note": note,
    }
    for tier_name in TIER_ORDER:
        row[f"matched_{tier_name}"] = 0
        row[f"precision_{tier_name}"] = 0.0
        row[f"recall_{tier_name}"] = 0.0
        row[f"f1_{tier_name}"] = 0.0
    return row


def evaluate_comp_question(
    qid: str,
    preds: dict[str, Any],
    obj_data: dict[str, Any],
    mth_data: dict[str, Any],
    max_workers: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    cfg = COMP_CONFIG[qid]
    gt_src = obj_data if cfg["gt_type"] == "object" else mth_data
    log_lines = ["=" * 80, qid]

    if qid not in gt_src:
        return [_empty_tier_summary(qid, "question", "GT missing")], log_lines + ["GT missing", ""]
    if qid not in preds:
        return [_empty_tier_summary(qid, "question", "Prediction missing")], log_lines + ["Prediction missing", ""]

    gt_ans = gt_src[qid]["answers"]
    pred_ans = preds[qid] if isinstance(preds[qid], dict) else {}

    ev_rows, ev_tier_stats, ev_paper_count = comp_evidences_table(
        pred_ans.get("evidences"), gt_ans.get("evidences"), cfg["evidence_fields"], max_workers
    )

    list_rows, list_tier_stats = comp_list_table(
        pred_ans.get("final_list"), gt_ans.get("final_list"), max_workers
    )

    gt_val = run_eval.get_oc_answers(gt_src[qid]).get("final_answer")
    pred_val = run_eval.extract_scalar_from_union(pred_ans, "final_answer", cfg["agg_mode"])
    scalar_score = run_eval.compute_scalar_score(pred_val, gt_val)

    ev_lenient = ev_tier_stats["lenient"]
    list_lenient = list_tier_stats["lenient"]

    ev_summary: dict[str, Any] = {
        "question_id": qid,
        "step": "evidences",
        "pred_count": ev_lenient["n_pred_items"],
        "gt_count": ev_lenient["n_gt_items"],
        "paper_count": ev_paper_count,
        "avg_semantic_precision_lenient": None,
        "note": "paper-avg",
    }
    for tier_name in TIER_ORDER:
        ts = ev_tier_stats[tier_name]
        ev_summary[f"matched_{tier_name}"] = ts["matched_items"]
        ev_summary[f"precision_{tier_name}"] = ts["macro_p"]
        ev_summary[f"recall_{tier_name}"] = ts["macro_r"]
        ev_summary[f"f1_{tier_name}"] = ts["macro_f1"]

    list_summary: dict[str, Any] = {
        "question_id": qid,
        "step": "final_list",
        "pred_count": list_lenient["n_pred"],
        "gt_count": list_lenient["n_gt"],
        "paper_count": None,
        "avg_semantic_precision_lenient": None,
        "note": "",
    }
    for tier_name in TIER_ORDER:
        ts = list_tier_stats[tier_name]
        list_summary[f"matched_{tier_name}"] = ts["matched"]
        list_summary[f"precision_{tier_name}"] = ts["precision"]
        list_summary[f"recall_{tier_name}"] = ts["recall"]
        list_summary[f"f1_{tier_name}"] = ts["f1"]

    # final_answer is a scalar exact-or-not check — same score across all tiers.
    fa_summary: dict[str, Any] = {
        "question_id": qid,
        "step": "final_answer",
        "pred_count": 1,
        "gt_count": 1,
        "paper_count": None,
        "avg_semantic_precision_lenient": None,
        "note": "",
    }
    fa_matched = int(scalar_score >= 1.0)
    for tier_name in TIER_ORDER:
        fa_summary[f"matched_{tier_name}"] = fa_matched
        fa_summary[f"precision_{tier_name}"] = scalar_score
        fa_summary[f"recall_{tier_name}"] = scalar_score
        fa_summary[f"f1_{tier_name}"] = scalar_score

    summaries = [ev_summary, list_summary, fa_summary]

    log_lines.append("")
    log_lines.append(
        f"[Step 1] evidences  papers={ev_paper_count}  "
        f"items: pred={ev_lenient['n_pred_items']}  gt={ev_lenient['n_gt_items']}"
    )
    for tier_name in TIER_ORDER:
        ts = ev_tier_stats[tier_name]
        log_lines.append(
            f"  {tier_name:<7s}: matched={ts['matched_items']:>3d}  "
            f"P={ts['macro_p']:.4f}  R={ts['macro_r']:.4f}  F1={ts['macro_f1']:.4f}  (paper-avg)"
        )
    log_lines.append(render_table(ev_rows, ["paper", "judge", "pred", "gt"]))
    log_lines.append("")
    log_lines.append(
        f"[Step 2] final_list  pred={list_lenient['n_pred']}  gt={list_lenient['n_gt']}"
    )
    for tier_name in TIER_ORDER:
        ts = list_tier_stats[tier_name]
        log_lines.append(
            f"  {tier_name:<7s}: matched={ts['matched']:>3d}  "
            f"P={ts['precision']:.4f}  R={ts['recall']:.4f}  F1={ts['f1']:.4f}"
        )
    log_lines.append(render_table(list_rows, ["pred", "gt", "status"]))
    log_lines.append("")
    log_lines.append(f"[Step 3] final_answer  pred={pred_val}  gt={gt_val}  score={scalar_score:.4f}")
    log_lines.append("")
    return summaries, log_lines


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_prediction_files(args: argparse.Namespace, pred_root: Path) -> list[Path]:
    if args.pred_file:
        files = [Path(path).resolve() for path in args.pred_file]
    else:
        selected_folders = choose_experiment_folders(pred_root, args.folders, args.list_folders)
        files = []
        for folder in selected_folders:
            files.extend(sorted(folder.glob(args.pred_glob)))
    if args.limit is not None:
        files = files[: args.limit]
    return files


def evaluate_file(pred_file: Path, pred_root: Path, run_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    domain = infer_domain(pred_file, pred_root)
    if not domain or domain not in PM_TO_GT:
        raise RuntimeError(f"Cannot infer GT domain from {pred_file}")

    gt_dir = Path("/mlde/ss/data/ground_truth") / PM_TO_GT[domain]
    obj_gt_file = gt_dir / "object.json"
    mth_gt_file = gt_dir / "method.json"

    with pred_file.open(encoding="utf-8") as f:
        preds = flatten_preds(json.load(f))

    obj_data = load_gt_file(obj_gt_file)
    mth_data = load_gt_file(mth_gt_file)
    filter_paper_ids = set(args.paper_ids) if args.paper_ids else None

    question_ids = args.questions if args.questions else sorted(preds.keys())

    rel_name = sanitize_relpath(pred_file, pred_root)
    log_path = run_dir / f"{rel_name}.log"
    metrics_path = run_dir / f"{rel_name}.csv"

    metrics_rows = []
    log_lines = [
        f"pred_file: {pred_file}",
        f"obj_gt_file: {obj_gt_file}",
        f"mth_gt_file: {mth_gt_file}",
        f"domain: {domain}",
        f"run_kind: {infer_run_kind(pred_file, pred_root)}",
        "",
    ]

    qid_iter = make_pbar(question_ids, total=len(question_ids), desc=f"questions:{pred_file.parent.name}", leave=False, position=1)
    for qid in qid_iter:
        if qid in FIELD_MAP:
            summary, qlog = evaluate_standard_question(qid, preds, obj_data, mth_data, filter_paper_ids, args.max_workers)
            metrics_rows.append(summary)
            log_lines.extend(qlog)
        elif qid in COMP_CONFIG:
            summaries, qlog = evaluate_comp_question(qid, preds, obj_data, mth_data, args.max_workers)
            metrics_rows.extend(summaries)
            log_lines.extend(qlog)

    for row in metrics_rows:
        row.update(
            {
                "pred_file": str(pred_file),
                "obj_gt_file": str(obj_gt_file),
                "mth_gt_file": str(mth_gt_file),
                "domain": domain,
                "run_kind": infer_run_kind(pred_file, pred_root),
                "experiment_folder": pred_file.relative_to(pred_root).parts[0],
                "log_file": str(log_path),
            }
        )

    pred_total = sum(row["pred_count"] for row in metrics_rows)
    gt_total = sum(row["gt_count"] for row in metrics_rows)

    file_summary: dict[str, Any] = {
        "pred_file": str(pred_file),
        "obj_gt_file": str(obj_gt_file),
        "mth_gt_file": str(mth_gt_file),
        "domain": domain,
        "run_kind": infer_run_kind(pred_file, pred_root),
        "experiment_folder": pred_file.relative_to(pred_root).parts[0],
        "question_rows": len(metrics_rows),
        "pred_total": pred_total,
        "gt_total": gt_total,
    }
    for tier_name in TIER_ORDER:
        matched_total_t = sum(row[f"matched_{tier_name}"] for row in metrics_rows)
        file_summary[f"matched_total_{tier_name}"] = matched_total_t
        macro_p_t = sum(row[f"precision_{tier_name}"] for row in metrics_rows) / len(metrics_rows) if metrics_rows else 0.0
        macro_r_t = sum(row[f"recall_{tier_name}"] for row in metrics_rows) / len(metrics_rows) if metrics_rows else 0.0
        macro_f1_t = sum(row[f"f1_{tier_name}"] for row in metrics_rows) / len(metrics_rows) if metrics_rows else 0.0
        micro_p_t, micro_r_t, micro_f1_t = prf1(matched_total_t, pred_total, gt_total)
        file_summary[f"micro_precision_{tier_name}"] = micro_p_t
        file_summary[f"micro_recall_{tier_name}"] = micro_r_t
        file_summary[f"micro_f1_{tier_name}"] = micro_f1_t
        file_summary[f"macro_precision_{tier_name}"] = macro_p_t
        file_summary[f"macro_recall_{tier_name}"] = macro_r_t
        file_summary[f"macro_f1_{tier_name}"] = macro_f1_t
    file_summary["metrics_file"] = str(metrics_path)
    file_summary["log_file"] = str(log_path)

    metrics_fieldnames = [
        "pred_file",
        "obj_gt_file",
        "mth_gt_file",
        "domain",
        "run_kind",
        "experiment_folder",
        "question_id",
        "step",
        "pred_count",
        "gt_count",
        "paper_count",
        "matched_strict", "precision_strict", "recall_strict", "f1_strict",
        "matched_medium", "precision_medium", "recall_medium", "f1_medium",
        "matched_lenient", "precision_lenient", "recall_lenient", "f1_lenient",
        "avg_semantic_precision_lenient",
        "note",
        "log_file",
    ]
    write_csv(metrics_path, metrics_rows, metrics_fieldnames)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    return metrics_rows, file_summary


def main() -> int:
    args = parse_args()
    pred_root = Path(args.pred_root).resolve()
    pred_files = collect_prediction_files(args, pred_root)
    if not pred_files:
        raise SystemExit("No prediction files found.")

    run_dir = Path(args.output_dir).resolve() if args.output_dir else (Path(__file__).resolve().parent / "results" / datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    file_summaries = []
    file_iter = make_pbar(pred_files, total=len(pred_files), desc="prediction files", leave=True, position=0)

    def _eval_and_report(pred_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return evaluate_file(pred_file, pred_root, run_dir, args)

    def _fmt_done(pf: Path, fs: dict[str, Any]) -> str:
        return (
            f"[done] {pf} | rows={fs['question_rows']} | "
            f"matched[S/M/L]={fs['matched_total_strict']}/{fs['matched_total_medium']}/{fs['matched_total_lenient']} | "
            f"macro_f1[S/M/L]={fs['macro_f1_strict']:.4f}/{fs['macro_f1_medium']:.4f}/{fs['macro_f1_lenient']:.4f}"
        )

    failed_files: list[tuple[Path, str]] = []

    def _handle_future_result(pred_file: Path, fut) -> None:
        try:
            metrics_rows, file_summary = fut.result()
        except Exception as exc:
            # Never silently swallow judge/eval errors — surface them loudly so
            # the operator notices a bad run (e.g. wrong EVAL_LLM_MODEL, server
            # down). One bad file does not sink the whole batch, but the
            # failure is recorded and reported at the end.
            import traceback
            tqdm.write(f"[FAIL] {pred_file}  ->  {exc.__class__.__name__}: {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            failed_files.append((pred_file, f"{exc.__class__.__name__}: {exc}"))
            file_iter.update(1)
            return
        all_metrics.extend(metrics_rows)
        file_summaries.append(file_summary)
        file_iter.update(1)
        tqdm.write(_fmt_done(pred_file, file_summary))

    if args.max_file_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_file_workers) as file_ex:
            futures = {file_ex.submit(_eval_and_report, pf): pf for pf in pred_files}
            for fut in as_completed(futures):
                _handle_future_result(futures[fut], fut)
    else:
        for pred_file in file_iter:
            try:
                metrics_rows, file_summary = evaluate_file(pred_file, pred_root, run_dir, args)
            except Exception as exc:
                import traceback
                tqdm.write(f"[FAIL] {pred_file}  ->  {exc.__class__.__name__}: {exc}")
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                failed_files.append((pred_file, f"{exc.__class__.__name__}: {exc}"))
                continue
            all_metrics.extend(metrics_rows)
            file_summaries.append(file_summary)
            file_iter.set_postfix(
                rows=file_summary["question_rows"],
                matched_L=file_summary["matched_total_lenient"],
                macro_f1_M=f"{file_summary['macro_f1_medium']:.4f}",
            )
            tqdm.write(_fmt_done(pred_file, file_summary))

    batch_summary_fields = [
        "pred_file",
        "obj_gt_file",
        "mth_gt_file",
        "domain",
        "run_kind",
        "experiment_folder",
        "question_rows",
        "pred_total",
        "gt_total",
        "matched_total_strict", "matched_total_medium", "matched_total_lenient",
        "micro_precision_strict", "micro_recall_strict", "micro_f1_strict",
        "micro_precision_medium", "micro_recall_medium", "micro_f1_medium",
        "micro_precision_lenient", "micro_recall_lenient", "micro_f1_lenient",
        "macro_precision_strict", "macro_recall_strict", "macro_f1_strict",
        "macro_precision_medium", "macro_recall_medium", "macro_f1_medium",
        "macro_precision_lenient", "macro_recall_lenient", "macro_f1_lenient",
        "metrics_file",
        "log_file",
    ]
    write_csv(run_dir / "batch_summary.csv", file_summaries, batch_summary_fields)

    all_metrics_fields = [
        "pred_file",
        "obj_gt_file",
        "mth_gt_file",
        "domain",
        "run_kind",
        "experiment_folder",
        "question_id",
        "step",
        "pred_count",
        "gt_count",
        "paper_count",
        "matched_strict", "precision_strict", "recall_strict", "f1_strict",
        "matched_medium", "precision_medium", "recall_medium", "f1_medium",
        "matched_lenient", "precision_lenient", "recall_lenient", "f1_lenient",
        "avg_semantic_precision_lenient",
        "note",
        "log_file",
    ]
    write_csv(run_dir / "all_metrics.csv", all_metrics, all_metrics_fields)

    print(f"Results written to: {run_dir}")
    print(f"Batch summary: {run_dir / 'batch_summary.csv'}")
    print(f"All metrics: {run_dir / 'all_metrics.csv'}")

    if failed_files:
        fail_path = run_dir / "failed_files.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            for pf, err in failed_files:
                f.write(f"{pf}\t{err}\n")
        print(f"[WARN] {len(failed_files)}/{len(pred_files)} file(s) FAILED evaluation:")
        for pf, err in failed_files:
            print(f"  - {pf}  ->  {err}")
        print(f"Failed list: {fail_path}")
        return 1
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
