#!/usr/bin/env python3
"""
Aggregated QA Evaluation Pipeline

Supports two aggregation modes:
1. GLOBAL: Send all papers to LLM at once, get aggregated answer directly
2. MAP-REDUCE: Extract from each paper separately, then aggregate programmatically

Output format matches ground truth:
{
  "Q01": {"study_population": ["pop1", "pop2", ...]},
  "Q02": {"country": ["country1", "country2", ...]},
  ...
}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib

import json
import os
import re
from datetime import datetime, UTC
from collections import defaultdict
from pathlib import Path

from loguru import logger
from openai import OpenAI


def _sanitize_audit_name(name: str, max_len: int = 140) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", str(name)).strip("_")
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:12]
    prefix_len = max(32, max_len - len(digest) - 1)
    return f"{cleaned[:prefix_len].rstrip('_')}_{digest}"


def init_audit_dirs(base_dir: Path | None) -> dict[str, Path] | None:
    """Create global/per_paper/aggregation audit directories for one run."""
    if base_dir is None:
        return None
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_id
    dirs = {
        "run": run_dir,
        "global": run_dir / "global",
        "per_paper": run_dir / "per_paper",
        "aggregation": run_dir / "aggregation",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("LLM raw-output audit directory: {}", run_dir)
    return dirs


def save_audit_record(audit_dirs: dict[str, Path] | None, section: str, name: str, payload: dict) -> None:
    """Persist one raw-output audit record."""
    if not audit_dirs:
        return
    out = audit_dirs[section] / f"{_sanitize_audit_name(name)}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def detect_model(client: OpenAI) -> str:
    """Auto-detect model name from the vLLM server."""
    try:
        models = client.models.list()
        if models.data:
            name = models.data[0].id
            logger.info("Auto-detected model: {}", name)
            return name
    except Exception as e:
        logger.warning("Failed to auto-detect model: {}", e)
    return ""

from .qa import (
    read_markdown,
    iter_markdown_files,
    _interleave_markdown_and_images,
)


# ---------------------------------------------------------------------------
# Thinking model support (e.g. Qwen3.5)
# ---------------------------------------------------------------------------
_thinking_supported: dict[str, bool] = {}  # base_url -> bool; absent = not yet probed

def _build_extra_body(**extra_kw) -> dict:
    """Build extra_body dict, disabling thinking for thinking models."""
    body = {"repetition_penalty": 1.15}
    body.update(extra_kw)
    # Always include enable_thinking=False; _safe_create drops it per-endpoint if unsupported
    body["chat_template_kwargs"] = {"enable_thinking": False}
    return body

def _extract_content(response) -> str:
    """Extract text content from a chat completion response.

    Strips special markers like <|begin_of_box|> / <|end_of_box|>.
    Note: reasoning/thinking field is intentionally NOT used as fallback —
    returning thinking text as the answer produces garbage structured output.
    """
    msg = response.choices[0].message
    text = msg.content or ""
    # Strip special markers from GLM-4 style outputs
    text = re.sub(r"<\|begin_of_box\|>|<\|end_of_box\|>", "", text)
    return text.strip()

def _safe_create(client: "OpenAI", **kwargs):
    """Call chat.completions.create with auto-fallback.

    Handles two failure modes:
    1. Thinking params not supported → disables per-endpoint and retries.
    2. Context length exceeded → halves max_tokens and retries (up to 3 times).
    """
    base_url = str(client.base_url)
    # Drop enable_thinking param for endpoints known not to support it
    if _thinking_supported.get(base_url) is False:
        if "extra_body" in kwargs and isinstance(kwargs["extra_body"], dict):
            kwargs["extra_body"].pop("chat_template_kwargs", None)
    try:
        resp = client.chat.completions.create(**kwargs)
        if base_url not in _thinking_supported:
            has_thinking = "chat_template_kwargs" in (kwargs.get("extra_body") or {})
            if has_thinking:
                _thinking_supported[base_url] = True
                logger.info("Thinking params accepted by {} — thinking disabled via enable_thinking=False", base_url)
        return resp
    except Exception as e:
        err_text = str(e).lower()

        # --- Fallback 1: thinking params not supported by this endpoint ---
        thinking_keywords = ("chat_template_kwargs", "enable_thinking")
        if _thinking_supported.get(base_url) is not False and any(k in err_text for k in thinking_keywords):
            _thinking_supported[base_url] = False
            logger.warning("Thinking params not supported by {} — removing for this endpoint", base_url)
            if "extra_body" in kwargs and isinstance(kwargs["extra_body"], dict):
                kwargs["extra_body"].pop("chat_template_kwargs", None)
            return client.chat.completions.create(**kwargs)

        # --- Fallback 2: context length exceeded → reduce max_tokens ---
        ctx_keywords = ("context length", "maximum input length", "reduce the length", "max_model_len")
        if any(k in err_text for k in ctx_keywords):
            cur = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 16384
            token_key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
            for attempt in range(3):
                new_val = max(cur // 2, 512)
                if new_val == cur:
                    break
                logger.warning("Context length exceeded — reducing {} from {} to {} (attempt {})",
                              token_key, cur, new_val, attempt + 1)
                kwargs[token_key] = new_val
                cur = new_val
                try:
                    return client.chat.completions.create(**kwargs)
                except Exception as retry_e:
                    if any(k in str(retry_e).lower() for k in ctx_keywords):
                        continue
                    raise
            logger.error("Context length still exceeded after reducing to {} — skipping this request", cur)
            return None

        raise


def _repair_json_response(
    client: OpenAI,
    model: str,
    raw_text: str,
    question: dict,
    max_tokens: int,
) -> dict:
    """Ask the model to repair invalid JSON without changing the extracted content."""
    if not raw_text or not raw_text.strip():
        return {}

    q_id = question.get("id", "")
    q_text = question.get("question", "")
    repair_prompt = (
        "Your previous answer had invalid JSON formatting.\n"
        "Repair it into exactly one valid JSON object.\n\n"
        "Rules:\n"
        "- Preserve the extracted content exactly when possible.\n"
        "- Do not add explanations, markdown, or code fences.\n"
        "- Do not invent new evidence.\n"
        "- Return only valid JSON.\n\n"
        f"Question ID: {q_id}\n"
        f"Original extraction question:\n{q_text}\n\n"
        "Invalid output to repair:\n"
        f"{raw_text}"
    )

    token_param = {}
    if model.startswith("gpt-5") or model.startswith("gpt"):
        token_param = {
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    else:
        token_param = {
            "max_tokens": max_tokens,
            "extra_body": _build_extra_body(),
        }

    try:
        response = _safe_create(
            client,
            model=model,
            temperature=0,
            **token_param,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON repair tool. "
                        "Fix formatting only and return one valid JSON object."
                    ),
                },
                {"role": "user", "content": repair_prompt},
            ],
        )
        if response is None:
            return {}
        repaired_text = _extract_content(response)
        repaired = parse_json_response(repaired_text, check_complete=False)
        if repaired:
            logger.info("{}: JSON repair pass succeeded", q_id)
        else:
            logger.warning("{}: JSON repair pass returned no valid object", q_id)
        return repaired
    except Exception as e:
        logger.warning("{}: JSON repair pass failed: {}", q_id, e)
        return {}


# ---------------------------------------------------------------------------
# Meta Analysis Data Extraction Principles
# ---------------------------------------------------------------------------

EXTRACTION_PRINCIPLES = r"""# Meta-Analysis Data Extraction Principles

### 1. Verbatim Fidelity & Full Names
* **Verbatim Copy:** All extracted text must be copied exactly from the source. No paraphrasing.
* **Mandatory Full Names:** Do not use standalone abbreviations for variables or methods.
    * If an abbreviation is used, you **must** locate its full name in the text. Format: `Full Name (Abbreviation)`.
    * If the full name cannot be found, the data point must be excluded.
    * If the source provides only the full name, recording the abbreviation is optional.
* **Variable Descriptions:** Always use the full name for Independent Variable (IV) and Dependent Variable (DV); never use standalone abbreviations.

### 2. Study Population & Sample Size
* **Final Sample Only:** Extract only the final analytic sample size ($N$) used in the statistical tests.
* **Screening Logic:** Copy verbatim any description of exclusions, attrition, or eligibility leading to that final $N$.

### 3. Association Test Methods & Scope
* **Inclusion Criteria:** Include any method quantifying the relationship between two variables (e.g., $r$, $\beta$, $OR$, path coefficients).
* **Exclusion Criteria:** Exclude group-comparison tests (e.g., Independent Samples T-tests, ANOVA) and purely descriptive statistics.
* **Method Identification:** If only a symbol (e.g., $R^2$) is reported, you must locate the specific name of the statistical method used to produce it.

### 4. Variable Selection & Mediation
* **Mediation Exclusion:** Exclude mediators unless they are explicitly tested as independent primary predictors in their own right.
* **Role Definition:** Use the "Theoretical Starting Point" as the IV and the "Resultant Variable" as the DV.

### 5. Scale vs. Unit
* **Scale:** Record the scoring range or score boundaries (e.g., "scored 1–5", "BMI $\ge$ 30"). If not reported, use "not applicable".
* **Unit:** Record only the raw measurement unit (e.g., "mg/dL", "years"). If it is a scale score/index without a physical unit, use "not applicable".

### 6. No Significance Bias
* **Universal Extraction:** Extract all qualified effect sizes regardless of p-values, statistical significance, or whether they support the author's hypothesis.

### 7. Output Format
* **Methods:** `Full Method Name (Symbol)`.
    * *Note:* Different prefixes (e.g., "crude" vs "adjusted") are considered conditions, not different methods.
* **Effect Size:** `IV Full Name & DV Full Name: Symbol = Value`.
* **Conditions:** If multiple results exist for one pair, use: `IV Full Name & DV Full Name (Condition/Prefix): Symbol = Value`.
* **Separation:** Use a semicolon (`;`) to separate multiple entries in the same field.
"""


# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

def load_questions(config_path: Path) -> list[dict]:
    """Load questions from standardized_config.json."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return config.get("questions", [])



# ---------------------------------------------------------------------------
# Build document contexts
# ---------------------------------------------------------------------------

def build_doc_contexts(
    outputs_dir: Path,
    max_docs: int | None = None,
    max_chars: int | None = None,
    include_images: bool = True,
    gt_paper_index: dict | None = None,  # {"title_normalized": "1", ...} for matching
    paper_map: dict | None = None,  # {paper_id_int: folder_name} from paper_map.json
) -> list[dict]:
    """
    Build document contexts from markdown files.
    
    If paper_map is provided ({paper_id: folder_name}), uses it to filter and order papers.
    If gt_paper_index is provided, uses CSV order (paper_idx from GT).
    Only processes papers that exist in both GT and markdown directory.
    
    Returns list of {"file": str, "text": str, "blocks": list[dict], "paper_idx": int, "title": str}.
    """
    contexts = []
    
    def extract_title(text: str) -> str:
        """Extract title from first # header in markdown"""
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('# ') and not line.startswith('# ARTICLE'):
                return line[2:].strip()
        return ""
    
    def normalize_title(title: str) -> str:
        """Normalize title for fuzzy matching"""
        import re
        title = title.lower().strip()
        title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
        title = re.sub(r'\s+', ' ', title)
        return title
    
    # --- Paper map mode: use explicit folder mapping ---
    if paper_map:
        for paper_id in sorted(paper_map.keys()):
            folder_name = paper_map[paper_id]
            # Find the markdown file in this folder
            folder_path = outputs_dir / folder_name
            if not folder_path.exists():
                logger.warning("Folder not found for paper [{}]: {}", paper_id, folder_name)
                continue
            
            vlm_dir = folder_path / "vlm"
            if not vlm_dir.exists():
                logger.warning("No vlm/ dir in {} for paper [{}]", folder_name, paper_id)
                continue
            
            md_files = list(vlm_dir.glob("*.md"))
            if not md_files:
                logger.warning("No .md files in {}/vlm/ for paper [{}]", folder_name, paper_id)
                continue
            
            md_path = md_files[0]
            rel = str(md_path.relative_to(outputs_dir))
            text = read_markdown(md_path, max_chars)
            title = extract_title(text)
            
            if include_images:
                blocks = _interleave_markdown_and_images(
                    md_path=md_path,
                    text=text,
                    max_images=5,
                    max_image_bytes=1_000_000,
                )
            else:
                blocks = [{"type": "text", "text": text}]
            
            contexts.append({
                "file": rel,
                "text": text,
                "blocks": blocks,
                "paper_idx": paper_id,
                "title": title,
            })
            
            if max_docs and len(contexts) >= max_docs:
                break
        
        logger.info("Loaded {} documents from {} (paper_map mode)", len(contexts), outputs_dir)
        return contexts
    
    # Get all markdown files
    all_md_files = list(iter_markdown_files(outputs_dir, skip_suffixes=[]))
    
    # Build a lookup: normalized_title -> md_path
    md_by_title = {}
    for md_path in all_md_files:
        text = read_markdown(md_path, max_chars=5000)  # Read just enough for title
        title = extract_title(text)
        if title:
            title_norm = normalize_title(title)
            md_by_title[title_norm] = md_path
    
    if gt_paper_index:
        # CSV-driven: iterate through GT papers in order
        # gt_paper_index is {title_normalized: paper_idx_str}
        # Sort by paper_idx (the value) to get correct order
        sorted_gt = sorted(gt_paper_index.items(), key=lambda x: int(x[1]))
        
        for gt_title, paper_idx_str in sorted_gt:
            paper_idx = int(paper_idx_str)
            
            # Find matching markdown file
            matched_path = None
            
            # Try exact match
            if gt_title in md_by_title:
                matched_path = md_by_title[gt_title]
            else:
                # Try partial match
                for md_title, md_path in md_by_title.items():
                    if gt_title[:50] in md_title or md_title[:50] in gt_title:
                        matched_path = md_path
                        break
            
            if not matched_path:
                logger.warning("No markdown found for GT paper [{}]: {}...", paper_idx, gt_title[:50])
                continue
            
            # Read full content
            rel = str(matched_path.relative_to(outputs_dir))
            text = read_markdown(matched_path, max_chars)
            title = extract_title(text)
            
            if include_images:
                blocks = _interleave_markdown_and_images(
                    md_path=matched_path,
                    text=text,
                    max_images=5,
                    max_image_bytes=1_000_000,
                )
            else:
                blocks = [{"type": "text", "text": text}]
            
            contexts.append({
                "file": rel,
                "text": text,
                "blocks": blocks,
                "paper_idx": paper_idx,
                "title": title,
            })
            
            if max_docs and len(contexts) >= max_docs:
                break
    else:
        # No GT: use file order (sequential)
        for paper_idx, md_path in enumerate(sorted(all_md_files), start=1):
            rel = str(md_path.relative_to(outputs_dir))
            text = read_markdown(md_path, max_chars)
            title = extract_title(text)
            
            if include_images:
                blocks = _interleave_markdown_and_images(
                    md_path=md_path,
                    text=text,
                    max_images=5,
                    max_image_bytes=1_000_000,
                )
            else:
                blocks = [{"type": "text", "text": text}]
            
            contexts.append({
                "file": rel,
                "text": text,
                "blocks": blocks,
                "paper_idx": paper_idx,
                "title": title,
            })
            
            if max_docs and len(contexts) >= max_docs:
                break
    
    logger.info("Loaded {} documents from {}", len(contexts), outputs_dir)
    return contexts


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def is_complete_json(text: str) -> bool:
    """
    Quick check if text appears to be complete JSON (balanced braces/brackets).
    Returns False for obviously truncated JSON.
    """
    text = text.strip()
    if not text:
        return False
    
    # Count braces and brackets
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Must have balanced pairs
    if open_braces != close_braces:
        return False
    if open_brackets != close_brackets:
        return False
    
    # Must start and end properly
    if text.startswith('{') and not text.rstrip().endswith('}'):
        return False
    if text.startswith('[') and not text.rstrip().endswith(']'):
        return False
    
    return True


def _extract_json_body(text: str) -> str:
    """Extract the most likely JSON object/array payload from model output."""
    text = text.strip()
    if not text:
        return ""

    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    if bracket_match:
        return bracket_match.group(0)
    return text


def _common_json_fixes(text: str) -> str:
    """Apply lightweight repairs for common malformed-but-recoverable JSON."""
    text = text.replace("'", '"')
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r'}\s*,\s*}\s*,\s*{', '}, {', text)
    text = re.sub(r'(\]\s*}\s*,)\s*}\s*,\s*{', r'\1 {', text)
    text = re.sub(r'(\]\s*})\s*},', r'\1,', text)
    text = re.sub(r'^\{\s*\{', '{', text, count=1)
    # Fix invalid JSON escape sequences (e.g. LaTeX \Psi, \varPhi → \\Psi, \\varPhi)
    text = re.sub(r'\\([^"\\/bfnrtux\d\s])', r'\\\\\1', text)
    return text


def _try_parse_merged_objects(text: str) -> dict:
    """Merge multiple sibling JSON objects emitted back-to-back."""
    all_objects = re.findall(r'\{[^{}]*\}', text)
    if len(all_objects) <= 1:
        return {}

    merged = {}
    parsed_count = 0
    for obj_str in all_objects:
        try:
            obj = json.loads(obj_str)
            parsed_count += 1
            for key, value in obj.items():
                if key not in merged:
                    merged[key] = []
                if isinstance(value, list):
                    merged[key].extend(value)
                else:
                    merged[key].append(value)
        except json.JSONDecodeError:
            continue

    if parsed_count > 0:
        logger.info("Merged {} separate JSON objects into one with keys: {}", parsed_count, list(merged.keys()))
        return merged
    return {}


def parse_json_response(text: str, check_complete: bool = True) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks and truncation."""
    text = _extract_json_body(text)
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text_fixed = _common_json_fixes(text)
        try:
            return json.loads(text_fixed)
        except json.JSONDecodeError:
            pass

    # Check for completeness and try to repair if truncated
    if check_complete and not is_complete_json(text):
        logger.warning("Incomplete JSON detected (truncated). Total length: {} chars", len(text))
        logger.warning("Last 200 chars: ...{}", text[-200:] if len(text) > 200 else text)
        
        # Retry truncation repair on the cleaned text too, since many malformed
        # cases are recoverable after removing duplicate separators/braces.
        for candidate in (text_fixed, text):
            repaired = try_repair_truncated_json(candidate)
            if repaired:
                logger.info("Recovered partial JSON with {} top-level keys", len(repaired))
                return repaired
        return {}

    try:
        return json.loads(text_fixed)
    except json.JSONDecodeError:
        merged = _try_parse_merged_objects(text)
        if merged:
            return merged

        logger.warning("Failed to parse JSON: {}", text[:200])
        return {}


def try_repair_truncated_json(text: str) -> dict:
    """Attempt to repair truncated JSON by finding last valid structure or extracting items."""
    # Strategy 1: Find the last complete item in arrays/objects
    
    if not text.startswith('{'):
        return {}
    
    # Strategy 2: Try progressively shorter versions until one parses
    attempts = [
        # Close incomplete array and object
        lambda t: t.rsplit('",', 1)[0] + '"]}' if '",' in t else None,
        lambda t: t.rsplit('},', 1)[0] + '}]}' if '},' in t else None,
        # Find last complete key-value and close
        lambda t: re.sub(r',\s*"[^"]*":\s*\[[^\]]*$', ']}', t) + '}' if re.search(r',\s*"[^"]*":\s*\[[^\]]*$', t) else None,
        # Just close with brackets
        lambda t: t.rstrip(',') + ']}',
    ]
    
    for attempt_fn in attempts:
        try:
            fixed = attempt_fn(text)
            if fixed:
                result = json.loads(fixed)
                if result:
                    return result
        except (json.JSONDecodeError, TypeError):
            continue
    
    # Strategy 3: Extract complete key-value pairs
    try:
        pattern = r'"([^"]+)":\s*(\[[^\]]*\]|\{[^}]*\}|"[^"]*"|[\d.]+|true|false|null)'
        matches = re.findall(pattern, text)
        if matches:
            result = {}
            for key, value in matches:
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    continue
            if result:
                return result
    except Exception:
        pass
    
    # Strategy 4: NEW - Extract individual items from truncated arrays using regex
    # This handles cases like: {"study_population": ["Item 1 [1]", "Item 2 [2]", "Item 3 [
    try:
        # Find the array key and extract all complete quoted strings within it
        # Match pattern: "key": [ followed by quoted strings
        array_pattern = r'"([^"]+)":\s*\[\s*'
        array_match = re.search(array_pattern, text)
        
        if array_match:
            key = array_match.group(1)
            # Get everything after the opening bracket
            array_start = array_match.end()
            array_content = text[array_start:]
            
            # Extract all complete quoted strings (items that end properly with ")
            # Pattern matches: "....." followed by optional comma/whitespace
            # Handles items like: "Grid cells [1]", "Planning units [2]"
            item_pattern = r'"((?:[^"\\]|\\.)*)"\s*(?:,|$)'
            items = re.findall(item_pattern, array_content)
            
            if items:
                # Unescape any escaped characters
                cleaned_items = []
                for item in items:
                    try:
                        # Handle escaped quotes and other escapes
                        cleaned = item.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                        cleaned_items.append(cleaned)
                    except:
                        cleaned_items.append(item)
                
                if cleaned_items:
                    logger.info("Recovered {} items from truncated array using regex extraction", len(cleaned_items))
                    return {key: cleaned_items}
    except Exception as e:
        logger.debug("Regex extraction failed: {}", e)
    
    # Strategy 5: Even more aggressive - find ALL quoted strings with citation markers
    try:
        # Extract any string that looks like an extracted item (ends with [N] pattern)
        citation_pattern = r'"([^"]*\[\d+\][^"]*)"'
        items_with_citations = re.findall(citation_pattern, text)
        
        if items_with_citations and len(items_with_citations) >= 3:  # At least 3 items to be meaningful
            # Try to find the key from the beginning of the JSON
            key_match = re.search(r'"([^"]+)":\s*\[', text)
            key = key_match.group(1) if key_match else "items"
            
            logger.info("Recovered {} items with citations using aggressive regex", len(items_with_citations))
            return {key: items_with_citations}
    except Exception:
        pass
    
    return {}


# ---------------------------------------------------------------------------
# GLOBAL MODE: Ask all papers at once
# ---------------------------------------------------------------------------

def ask_global_question(
    client: OpenAI,
    model: str,
    docs: list[dict],
    question: dict,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    audit_dirs: dict[str, Path] | None = None,
) -> dict:
    """
    Ask a question across ALL documents at once.
    Returns the aggregated answer directly from LLM.
    Supports multimodal input (text + images).
    """
    q_id = question["id"]
    q_text = question["question"]
    
    is_minimax = "minimax" in (model or "").lower()
    base = (
        "You are a scientific literature analyst extracting aggregated data across multiple papers.\n"
        "Read ALL the papers below (including figures and tables) and extract the requested information.\n"
        "Output your answer as a single JSON object matching the required schema.\n\n"
        "## CRITICAL RULES\n"
        "1. NO DUPLICATES: Each data point appears ONLY ONCE unless the schema requires per-paper evidence.\n"
        "2. EXHAUSTIVE EXTRACTION: Extract ALL instances from EVERY paper [1] to [N].\n"
        "3. COMPLETE COVERAGE: Do not skip any paper.\n\n"
        "## CITATION REQUIREMENT\n"
        "For EVERY extracted value in `evidences`, append the paper index [N] inside the string.\n"
        "For C questions, return {answers:[{paper_id, answer:[...]}]}.\n"
        "For OC/MC computation questions, return {evidences, final_list, final_answer}.\n\n"
    )
    if is_minimax:
        # MiniMax is confused by long schema scaffolding — keep it terse.
        schema_block = (
            "## final_list rule (for OC/MC computation questions)\n"
            "- `final_list` is a LIST OF DICTS. Each entry: semantic fields "
            "(e.g., `country`, `sample_size`, `statistical_method`, `independent_variable`, "
            "`dependent_variable`) FIRST, `paper_id` LAST.\n"
            "- `paper_id` is a JSON ARRAY of integer paper indices (e.g., `[3]`, `[1, 2, 5]`). "
            "Use an array even for a single paper. Sort ascending, drop duplicates.\n"
            "- Do NOT put inline `[N]` citation markers inside the semantic-field values.\n"
            "- Example: `{\"country\": \"Brazil\", \"paper_id\": [1, 5]}`.\n\n"
        )
    else:
        schema_block = (
            "## OUTPUT SCHEMA BY QUESTION TYPE\n"
            "### (A) Extraction questions (O1.*, O2.*, M1.*, M2.*):\n"
            "```json\n"
            "{\"answers\": [ {\"paper_id\": 1, \"answer\": [ {...schema fields with [N] citations...} ]}, ... ]}\n"
            "```\n"
            "Top-level key `answers` is a LIST, ONE entry PER PAPER.\n\n"
            "### (B) Computation questions (OC_*, MC_*):\n"
            "The top-level JSON MUST have EXACTLY three keys — `evidences`, `final_list`, `final_answer`. "
            "Never return a bare list, a single flat dict, or only part of this structure.\n"
            "```json\n"
            "{\n"
            "  \"evidences\": [ {\"paper_id\": <int>, \"answer\": [ {...schema fields with [N] citations...} ]}, ... ],\n"
            "  \"final_list\": [ { <semantic fields first>, \"paper_id\": [<int>, ...] }, ... ],\n"
            "  \"final_answer\": <number>\n"
            "}\n"
            "```\n\n"
            "### final_list rules (for computation questions)\n"
            "- Each entry is a DICT. Semantic fields (`country`, `sample_size`, `statistical_method`, "
            "`independent_variable`, `dependent_variable`, etc.) come FIRST; `paper_id` comes LAST.\n"
            "- `paper_id` is always a JSON ARRAY of integer paper indices: `[3]`, `[1, 2, 5]`. "
            "Use an array even when a single paper contributes.\n"
            "- Merge cross-paper duplicates into ONE entry and UNION their `paper_id`.\n"
            "- Sort `paper_id` ascending; drop duplicates.\n"
            "- Do NOT put inline citation markers like `[1]` inside the semantic-field values — "
            "provenance belongs in `paper_id`.\n\n"
            "### Worked example (for 'count unique countries')\n"
            "```json\n"
            "{\n"
            "  \"evidences\": [\n"
            "    {\"paper_id\": 1, \"answer\": [{\"geolocation\": \"Brazil [1]\"}]},\n"
            "    {\"paper_id\": 5, \"answer\": [{\"geolocation\": \"Brazil [5]\"}]},\n"
            "    {\"paper_id\": 7, \"answer\": [{\"geolocation\": \"Kenya [7]\"}]}\n"
            "  ],\n"
            "  \"final_list\": [\n"
            "    {\"country\": \"Brazil\", \"paper_id\": [1, 5]},\n"
            "    {\"country\": \"Kenya\", \"paper_id\": [7]}\n"
            "  ],\n"
            "  \"final_answer\": 2\n"
            "}\n"
            "```\n\n"
        )
    system_prompt = base + schema_block + EXTRACTION_PRINCIPLES
    
    # Build multimodal content blocks
    user_content = []
    user_content.append({"type": "text", "text": f"The following are {len(docs)} scientific papers:\n"})
    
    total_chars = 0
    total_images = 0
    
    for doc in docs:
        paper_idx = doc.get("paper_idx", 1)
        # Add paper header with explicit index for citation
        user_content.append({"type": "text", "text": f"\n=== PAPER [{paper_idx}] ===\n"})
        
        # Add blocks (text and images)
        for block in doc.get("blocks", []):
            if block["type"] == "text":
                user_content.append({"type": "text", "text": block["text"]})
                total_chars += len(block["text"])
            elif block["type"] == "image_url":
                user_content.append(block)
                total_images += 1
    
    # Add question with citation reminder
    user_content.append({
        "type": "text", 
        "text": f"\n\nQuestion ({q_id}): {q_text}\n\n"
                "REMINDER: Append [N] to EACH value where N = paper number from the header.\n"
                "Example: 'Taiwan [1]', 'r=0.35 [2]', 'adults aged 18-65 [3]'\n"
                "Provide your answer as JSON only."
    })
    
    estimated_tokens = total_chars // 4
    logger.info("{}: Input ~{} chars (~{}K tokens estimate), {} images", q_id, total_chars, estimated_tokens // 1000, total_images)
    
    token_param = {}
    if model.startswith("gpt-5") or model.startswith("gpt"):
        # GPT models use max_completion_tokens and json_object response format
        token_param = {
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    else:
        # For local models (Qwen via vLLM)
        token_param = {
            "max_tokens": max_tokens,
            "extra_body": _build_extra_body(),
        }
    
    # Retry logic for JSON parsing failures
    max_retries = 5
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = _safe_create(client,
                model=model,
                temperature=temperature,
                **token_param,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            
            if response is None:
                return {}
            
            message = _extract_content(response)
            
            # Log token usage from API
            completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                completion_tokens = response.usage.completion_tokens
                logger.info("{}: prompt_tokens={}, completion_tokens={}, total={}", 
                           q_id, response.usage.prompt_tokens, 
                           completion_tokens, response.usage.total_tokens)
            
            # Detect if response was truncated due to max_tokens
            was_truncated = completion_tokens >= max_tokens - 10  # Allow small margin
            
            result = parse_json_response(message)
            save_audit_record(audit_dirs, "global", q_id, {
                "mode": "global",
                "question_id": q_id,
                "papers": [{"paper_id": doc.get("paper_idx"), "paper": doc.get("file")} for doc in docs],
                "raw_text": message,
                "parsed_answer": result,
                "was_truncated": was_truncated,
            })
            
            # Check if result is valid
            if result:
                if was_truncated:
                    logger.warning("{}: Response was truncated but got {} items from partial recovery", 
                                 q_id, len(result.get(list(result.keys())[0], [])) if result else 0)
                return result
            else:
                if was_truncated:
                    repaired = _repair_json_response(client, model, message, question, max_tokens)
                    if repaired:
                        return repaired
                    logger.warning("{}: Response truncated at {} tokens, no valid JSON recovered. Returning empty.", 
                                 q_id, completion_tokens)
                    return {}
                last_error = "Empty JSON result"
                logger.warning("{} attempt {}: {}", q_id, attempt + 1, last_error)
                repaired = _repair_json_response(client, model, message, question, max_tokens)
                if repaired:
                    return repaired
        
        except Exception as e:
            last_error = str(e)
            # On timeout, skip immediately — don't waste retries
            if "timed out" in last_error.lower() or "timeout" in last_error.lower():
                logger.warning("{}: timed out — skipping (no retry)", q_id)
                return {}
            logger.warning("{} attempt {} failed: {}", q_id, attempt + 1, last_error)

    logger.error("Failed {} after {} retries: {}", q_id, max_retries, last_error)
    return {}


def run_global_extraction(
    client: OpenAI,
    model: str,
    docs: list[dict],
    questions: list[dict],
    temperature: float,
    max_tokens: int,
    parallel: int = 3,
    audit_dirs: dict[str, Path] | None = None,
) -> dict:
    """Run global extraction for all questions."""
    results = {}

    logger.info("Running GLOBAL extraction for {} questions across {} papers (parallel={})", 
                len(questions), len(docs), parallel)
    
    def ask_q(q):
        return q["id"], ask_global_question(client, model, docs, q, temperature, max_tokens, audit_dirs=audit_dirs)
    
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_q = {executor.submit(ask_q, q): q["id"] for q in questions}
            for future in concurrent.futures.as_completed(future_to_q):
                q_id = future_to_q[future]
                try:
                    result_id, result_data = future.result()
                    results[result_id] = result_data
                    if isinstance(result_data, dict):
                        n_items = sum(len(v) if isinstance(v, list) else 1 for v in result_data.values())
                    elif isinstance(result_data, list):
                        n_items = len(result_data)
                    else:
                        n_items = 1
                    logger.info("Completed {}: {} items", result_id, n_items)
                except Exception as e:
                    logger.exception("Failed {}: {}", q_id, e)
                    results[q_id] = {}
    else:
        for q in questions:
            q_id, result = ask_q(q)
            results[q_id] = result
    
    return results


# ---------------------------------------------------------------------------
# MAP-REDUCE MODE: Extract per paper, then aggregate
# ---------------------------------------------------------------------------

def ask_paper_question(
    client: OpenAI,
    model: str,
    doc: dict,
    question: dict,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    audit_dirs: dict[str, Path] | None = None,
) -> dict:
    """Ask a question for a single paper. Supports multimodal (text + images)."""
    q_id = question["id"]
    q_text = question["question"]
    paper_idx = doc.get("paper_idx", 1)
    
    system_prompt = (
        "You are a scientific paper analyst extracting structured data from a single paper.\n"
        "Answer the question based ONLY on the provided paper.\n"
        "Output your answer as valid JSON matching the specified format.\n\n"
        "## CITATION REQUIREMENT\n"
        f"CRITICAL: Append [{paper_idx}] to EVERY extracted value.\n"
        f"Examples: 'Children aged 5-18 years [{paper_idx}]', 'r = 0.283 [{paper_idx}]'\n\n"
        f"{EXTRACTION_PRINCIPLES}"
    )
    
    # Build user content - use blocks for multimodal if available
    blocks = doc.get("blocks", [])
    
    if blocks and any(b.get("type") == "image_url" for b in blocks):
        # Multimodal: text + images
        user_content = [
            {"type": "text", "text": f"Paper [{paper_idx}]\n\n"},
        ]
        
        for block in blocks:
            if block.get("type") == "text":
                user_content.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image_url":
                user_content.append({
                    "type": "image_url",
                    "image_url": block["image_url"]
                })
        
        user_content.append({
            "type": "text",
            "text": f"\n\nQuestion ({q_id}): {q_text}\n\nRespond with only the JSON object that matches the Output Format specified in the question above."
        })
    else:
        # Text only
        user_content = (
            f"Paper [{paper_idx}]\n\n"
            f"{doc['text']}\n\n"
            f"Question ({q_id}): {q_text}\n\n"
            "Respond with only the JSON object that matches the Output Format specified in the question above."
        )
    
    token_param = {}
    if model.startswith("gpt"):
        token_param = {
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    else:
        # For local models (Qwen via vLLM)
        token_param = {
            "max_tokens": max_tokens,
            "extra_body": _build_extra_body(),
        }
    
    try:
        response = _safe_create(client,
            model=model,
            temperature=temperature,
            **token_param,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        
        if response is None:
            return {}
        
        message = _extract_content(response)
        
        # Log token usage
        if hasattr(response, 'usage') and response.usage:
            logger.debug("{} [{}]: prompt={}, completion={}, total={}", 
                        q_id, doc['file'][:20], response.usage.prompt_tokens, 
                        response.usage.completion_tokens, response.usage.total_tokens)
        
        result = parse_json_response(message)
        save_audit_record(audit_dirs, "per_paper", f"{q_id}__{doc['file']}", {
            "mode": "per_paper",
            "question_id": q_id,
            "paper_id": doc.get("paper_idx"),
            "paper": doc.get("file"),
            "raw_text": message,
            "parsed_answer": result,
        })
        if result:
            return result

        logger.warning("{} [{}]: invalid JSON, attempting repair pass",
                      q_id, doc['file'][:20])
        repaired = _repair_json_response(client, model, message, question, max_tokens)
        if repaired:
            return repaired
        return {}
    
    except Exception as e:
        logger.error("Error on {} for {}: {}", q_id, doc['file'], e)
        return {}


def _strip_citation(value) -> str:
    return re.sub(r'\s*\[\d+\]$', '', str(value)).strip()


def _resolve_paper_id(
    paper_file: str,
    default_idx: int,
    file_to_paper_idx: dict[str, int] | None,
) -> int:
    if not file_to_paper_idx:
        return default_idx
    if paper_file in file_to_paper_idx:
        return file_to_paper_idx[paper_file]
    for fname, pidx in file_to_paper_idx.items():
        if paper_file in fname or fname in paper_file:
            return pidx
    return default_idx


# Questions where single-paper prompts should say "Extract ALL" (may contain multiple items).
HIGH_DENSITY_QUESTIONS = {
    # Old T-series IDs (kept for backward compat)
    "T4.1", "T5.1", "T2.3", "T2.4", "T2.5", "T2.6",
    # Basic extraction (list-type)
    "O1.1", "O1.2", "O1.3",
    "M1.1", "M1.2", "M1.3", "M1.4",
    # Composite extraction (list-type)
    "O2.1", "O2.2", "O2.3",
    # Object computation (new IDs)
    "OC_O1.1", "OC_O1.2.1", "OC_O1.2.2", "OC_O1.2.3",
    # Backward compat for old IDs
    "OC.1", "OC.2", "OC.3", "OC.4",
    # Method composite (list-type, high-density)
    "M2.1", "M2.2", "M2.3", "M2.4", "M2.5", "M2.6",
    # Method computation (new IDs)
    "MC_M1.1", "MC_M1.2", "MC_M1.3", "MC_M1.4", "MC_M2.4", "MC_M2.6",
    # Backward compat for old IDs
    "MC.1", "MC.2", "MC.3", "MC.4", "MC.5",
}


def computation_post_process(
    aggregated: dict,
    per_paper_results: dict,
    file_to_paper_idx: dict | None = None,
) -> dict:
    """
    Post-process computation queries (OC_*, MC_*) after union aggregation.

    IMPORTANT: This reads from per_paper_results DIRECTLY, not from the union-
    aggregated data, because union aggregation destructures the multi-step JSON
    (takes first list, deduplicates items, drops context keys). We need the
    original per-paper answers to correctly aggregate across papers.
    """

    def _parse_number(s):
        """Extract a number from a string like '2014 QDS', '1,485', '15,018 cells'."""
        s = _strip_citation(str(s))
        s = re.sub(r'\s*(QDS|grid cells?|cells?|plots?|pixels?|sub-watershed areas?|'
                   r'individuals?|people|leaves?|hotels?|participants?|'
                   r'terrestrial cells|children|students|subjects?).*$', '', s, flags=re.IGNORECASE)
        s = s.replace(',', '').strip()
        range_match = re.match(r'(\d+)\s*[-–]\s*(\d+)', s)
        if range_match:
            return float(range_match.group(1))
        num_match = re.search(r'(\d+(?:\.\d+)?)', s)
        if num_match:
            return float(num_match.group(1))
        return None
    
    def _get_per_paper_answers(q_id):
        """Get list of (paper_file, answer_dict) for a question, sorted by file."""
        results = []
        for paper_file, answers in sorted(per_paper_results.items()):
            if q_id in answers and answers[q_id]:
                results.append((paper_file, answers[q_id]))
        return results

    def _pid_as_int(x):
        """Normalize a paper_id value to an int (for the paper_id:[int,...] schema)."""
        try:
            return int(str(x).strip())
        except (TypeError, ValueError, AttributeError):
            return None

    def _dedupe_preserve_order(values):
        seen = set()
        deduped = []
        for value in values:
            marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(value)
        return deduped

    def _extract_answer_items(ans, candidate_keys):
        if isinstance(ans, dict):
            # Per-paper wrapper: {"answers": [{"paper_id": "<id>", "answer": [{...}]}]}
            answers_wrap = ans.get("answers")
            if isinstance(answers_wrap, list):
                items = []
                for ent in answers_wrap:
                    if isinstance(ent, dict):
                        inner = ent.get("answer", [])
                        if isinstance(inner, list):
                            for item in inner:
                                if isinstance(item, dict):
                                    items.append(item)
                                elif isinstance(item, str) and item.strip():
                                    items.append({candidate_keys[0]: item})
                if items:
                    return items
            if isinstance(ans.get("answer"), list):
                return [item for item in ans["answer"] if isinstance(item, dict)]
            for key in candidate_keys:
                value = ans.get(key)
                if isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        return [item for item in value if isinstance(item, dict)]
                    return [{key: item} for item in value if item not in (None, "")]
                if isinstance(value, dict):
                    return [value]
                if isinstance(value, str) and value.strip():
                    # If ans itself has multiple candidate keys (i.e., ans IS the item), return [ans]
                    if sum(1 for k in candidate_keys if ans.get(k) not in (None, "")) > 1:
                        return [ans]
                    return [{key: value}]
            # Per-paper map-reduce format: dig into evidences[*].answer, then final_list
            evidences = ans.get("evidences")
            if isinstance(evidences, list):
                items = []
                for ev in evidences:
                    if isinstance(ev, dict):
                        inner = ev.get("answer", [])
                        if isinstance(inner, list):
                            for item in inner:
                                if isinstance(item, dict):
                                    items.append(item)
                                elif isinstance(item, str) and item.strip():
                                    items.append({candidate_keys[0]: item})
                if items:
                    return items
            # Last resort: final_list (strings wrapped with first candidate key)
            final_list = ans.get("final_list")
            if isinstance(final_list, list):
                return [
                    {candidate_keys[0]: item} if isinstance(item, str) else item
                    for item in final_list
                    if item not in (None, "")
                ]
        elif isinstance(ans, list):
            if ans and isinstance(ans[0], dict):
                return [item for item in ans if isinstance(item, dict)]
        return []

    # --- OC_O1.2.x: Compute count/avg/median from per-paper sample sizes ---
    # With LLM decomposition, each OC_O1.2.x per-paper answer contains the
    # extracted sample size for that paper. We collect and aggregate here.
    
    _need_sample_sizes = any(q in aggregated for q in ["OC_O1.2.1", "OC_O1.2.2", "OC_O1.2.3"])
    if _need_sample_sizes:
        def _extract_sample_size_candidate(ans):
            raw_val = None
            if isinstance(ans, dict):
                # 1. Direct top-level keys
                for key in [
                    "sample_size", "raw", "raw_sample_size", "total_sample_size",
                    "final_sample_size", "analytic_sample_size", "N",
                    "paper_level_sample_size",
                ]:
                    val = ans.get(key)
                    if val is None:
                        continue
                    if isinstance(val, list) and val:
                        entry = val[0]
                        raw_val = entry.get("raw", str(entry)) if isinstance(entry, dict) else str(entry)
                    elif isinstance(val, str):
                        if val.strip():
                            raw_val = val
                    elif isinstance(val, (int, float)):
                        raw_val = str(val)
                    if raw_val is not None:
                        break
                # 2. Dig into nested evidences/answer/answers structure (from per-paper map-reduce format)
                if raw_val is None:
                    for container_key in ["answers", "evidences", "answer"]:
                        container = ans.get(container_key)
                        if isinstance(container, list) and container:
                            first = container[0]
                            if isinstance(first, dict):
                                # answers[0]["answer"][0]["sample_size"] (new per-paper shape) or
                                # evidences[0]["answer"][0]["sample_size"] / answer[0]["sample_size"]
                                inner = first.get("answer", [first])
                                if isinstance(inner, list) and inner:
                                    inner_item = inner[0]
                                elif isinstance(inner, dict):
                                    inner_item = inner
                                else:
                                    inner_item = None
                                if isinstance(inner_item, dict):
                                    for key in ["sample_size", "N", "n", "raw"]:
                                        v = inner_item.get(key)
                                        if v and (isinstance(v, str) and v.strip() or isinstance(v, (int, float))):
                                            raw_val = str(v)
                                            break
                        if raw_val is not None:
                            break
                # 3. Fallback: scan values, but skip small integers (likely binary 0/1 counts)
                if raw_val is None:
                    for _, v in ans.items():
                        if isinstance(v, (int, float)) and v > 1:
                            raw_val = str(v)
                            break
                        if isinstance(v, str) and _parse_number(v) is not None and _parse_number(v) > 1:
                            raw_val = v
                            break
            elif isinstance(ans, str) and ans.strip():
                raw_val = ans

            if raw_val is None:
                return None, None
            parsed = _parse_number(raw_val)
            return raw_val, parsed

        # Merge sample-size evidence across OC_O1.2.x per paper.
        # OC_O1.2.2/3 are processed first — their per-paper answers contain the actual
        # sample size. OC_O1.2.1 per-paper returns a binary 0/1 (does this paper have
        # sample_size > 100?), so it's checked last as fallback only.
        sample_by_paper = {}
        for source_q in ["OC_O1.2.2", "OC_O1.2.3", "OC_O1.2.1"]:
            for default_idx, (paper_file, ans) in enumerate(_get_per_paper_answers(source_q), 1):
                paper_id = _resolve_paper_id(paper_file, default_idx, file_to_paper_idx)
                raw_val, parsed = _extract_sample_size_candidate(ans)
                existing = sample_by_paper.get(paper_id)
                candidate = {
                    "paper_id": paper_id,
                    "raw": raw_val if raw_val is not None else str(ans),
                    "parsed": parsed,
                }
                if existing is None:
                    sample_by_paper[paper_id] = candidate
                elif existing["parsed"] is None and parsed is not None:
                    sample_by_paper[paper_id] = candidate

        all_sample_sizes = [sample_by_paper[pid] for pid in sorted(sample_by_paper)]

        if not all_sample_sizes:
            logger.warning("OC_O1.2.x: no sample size data found from any source")
        else:
            logger.info("OC_O1.2.x: collected {} sample sizes", len(all_sample_sizes))
        
        _parsed_entries = [s for s in all_sample_sizes if s["parsed"] is not None]
        _normalized_values = [s["parsed"] for s in _parsed_entries]
        _ss_evidences = [
            {"paper_id": s["paper_id"], "answer": [{"sample_size": s["raw"]}]}
            for s in all_sample_sizes
        ]

        def _fmt_num(n):
            return str(int(n)) if float(n).is_integer() else str(n)

        # OC_O1.2.1: Count papers with sample_size > 100
        if "OC_O1.2.1" in aggregated:
            eligible = [(s["paper_id"], s["parsed"]) for s in _parsed_entries if s["parsed"] > 100]
            aggregated["OC_O1.2.1"] = {
                "evidences": _ss_evidences,
                "final_list": [
                    {"sample_size": _fmt_num(n), "paper_id": [pid]} for pid, n in eligible
                ],
                "final_answer": len(eligible),
            }
            logger.info("OC_O1.2.1 post-processed: {} papers with sample_size > 100 (from {} papers)", len(eligible), len(all_sample_sizes))

        # OC_O1.2.2: Average sample size
        if "OC_O1.2.2" in aggregated:
            items = [(s["paper_id"], s["parsed"]) for s in _parsed_entries]
            nums = [n for _, n in items]
            avg = round(sum(nums) / len(nums), 2) if nums else 0
            aggregated["OC_O1.2.2"] = {
                "evidences": _ss_evidences,
                "final_list": [
                    {"sample_size": _fmt_num(n), "paper_id": [pid]} for pid, n in items
                ],
                "final_answer": avg,
            }
            logger.info("OC_O1.2.2 post-processed: average={} from {} values", avg, len(nums))

        # OC_O1.2.3: Median sample size
        if "OC_O1.2.3" in aggregated:
            items = [(s["paper_id"], s["parsed"]) for s in _parsed_entries]
            if items:
                sorted_items = sorted(items, key=lambda t: t[1])
                sorted_nums = [n for _, n in sorted_items]
                n = len(sorted_nums)
                median = (sorted_nums[n // 2] + sorted_nums[(n - 1) // 2]) / 2
            else:
                sorted_items = []
                median = 0
            aggregated["OC_O1.2.3"] = {
                "evidences": _ss_evidences,
                "final_list": [
                    {"sample_size": _fmt_num(n), "paper_id": [pid]} for pid, n in sorted_items
                ],
                "final_answer": round(median, 2),
            }
            logger.info("OC_O1.2.3 post-processed: median={} from {} values", round(median, 2), len(items))

    # --- OC_O1.1: Count unique countries ---
    if "OC_O1.1" in aggregated:
        paper_answers = _get_per_paper_answers("OC_O1.1")

        def _extract_country_from_location(loc_str: str) -> str:
            """Fallback heuristic: last comma-separated token is usually the country."""
            parts = [p.strip() for p in loc_str.split(",") if p.strip()]
            return parts[-1] if parts else loc_str

        def _normalize_country_name(value: str) -> str | None:
            text = _strip_citation(value)
            if not text:
                return None
            text = re.sub(r"\s+", " ", text).strip()
            lowered = text.casefold()
            if lowered in {"not reported", "...", "global", "worldwide"}:
                return None
            if lowered.startswith("<") and lowered.endswith(">"):
                return None

            alias_map = {
                "usa": "United States",
                "u.s.a.": "United States",
                "u.s.": "United States",
                "united states of america": "United States",
                "uk": "United Kingdom",
                "u.k.": "United Kingdom",
                "the netherlands": "Netherlands",
                "republic of korea": "South Korea",
            }
            return alias_map.get(lowered, text)

        def _extract_location_from_ans(ans):
            """Fallback location extraction from per-paper answer (legacy heuristic)."""
            if not isinstance(ans, dict):
                return None
            for loc_key in ["geolocation", "location", "geographical_setting",
                            "paper_level_locations", "locations", "country", "countries"]:
                v = ans.get(loc_key)
                if isinstance(v, str) and v.strip():
                    return _strip_citation(v).strip()
                elif isinstance(v, list) and v:
                    first = v[0]
                    loc_str = _strip_citation(
                        str(first.get("geolocation", first.get("location", first)) if isinstance(first, dict) else first)
                    ).strip()
                    if loc_str:
                        return loc_str
            final_list = ans.get("final_list")
            if isinstance(final_list, list) and final_list:
                v = final_list[0]
                if isinstance(v, str) and v.strip():
                    return _strip_citation(v).strip()
            for container_key in ["evidences", "answer"]:
                container = ans.get(container_key)
                if isinstance(container, list) and container:
                    first = container[0]
                    if isinstance(first, dict):
                        inner = first.get("answer", [first])
                        if isinstance(inner, list) and inner:
                            inner_item = inner[0]
                        elif isinstance(inner, dict):
                            inner_item = inner
                        else:
                            inner_item = None
                        if isinstance(inner_item, dict):
                            for key in ["geolocation", "location", "country"]:
                                v = inner_item.get(key)
                                if isinstance(v, str) and v.strip():
                                    return _strip_citation(v).strip()
            return None

        existing = aggregated.get("OC_O1.1") if isinstance(aggregated.get("OC_O1.1"), dict) else {}
        existing_evidences = existing.get("evidences") if isinstance(existing, dict) else None

        evidences = existing_evidences if isinstance(existing_evidences, list) else []

        # country_key (casefold) -> {"display": str, "pids": set[int]}
        country_buckets: dict[str, dict] = {}

        def _add_country(name: str | None, pid) -> None:
            if not name:
                return
            pid_i = _pid_as_int(pid)
            key = name.casefold()
            bucket = country_buckets.setdefault(key, {"display": name, "pids": set()})
            if pid_i is not None:
                bucket["pids"].add(pid_i)

        # Re-derive from evidences (which carry paper_id)
        if evidences:
            for ev in evidences:
                if not isinstance(ev, dict):
                    continue
                ev_pid = ev.get("paper_id")
                for item in ev.get("answer") or []:
                    if not isinstance(item, dict):
                        continue
                    loc = item.get("geolocation") or item.get("location") or item.get("country")
                    if isinstance(loc, str):
                        country = _normalize_country_name(_extract_country_from_location(loc))
                        _add_country(country, ev_pid)
        else:
            for default_idx, (paper_file, ans) in enumerate(paper_answers, 1):
                paper_id = _resolve_paper_id(paper_file, default_idx, file_to_paper_idx)
                location = _extract_location_from_ans(ans)
                if location:
                    evidences.append({"paper_id": paper_id, "answer": [{"geolocation": location}]})
                    country = _normalize_country_name(_extract_country_from_location(location))
                    _add_country(country, paper_id)

        final_list = [
            {
                "country": bucket["display"],
                "paper_id": sorted(bucket["pids"]),
            }
            for _, bucket in sorted(country_buckets.items())
        ]

        aggregated["OC_O1.1"] = {
            "evidences": evidences,
            "final_list": final_list,
            "final_answer": len(final_list),
        }
        logger.info("OC_O1.1 post-processed: {} unique countries from {} papers",
                    len(final_list), len(paper_answers))

    # --- MC_M1.1: Count unique statistical methods ---
    if "MC_M1.1" in aggregated:
        paper_answers = _get_per_paper_answers("MC_M1.1")
        evidences = []
        buckets: dict[str, dict] = {}
        for default_idx, (paper_file, ans) in enumerate(paper_answers, 1):
            paper_id = _resolve_paper_id(paper_file, default_idx, file_to_paper_idx)
            pid_i = _pid_as_int(paper_id)
            answer_items = []
            for item in _extract_answer_items(ans, ["statistical_method", "methods", "method"]):
                method = _strip_citation(item.get("statistical_method", item.get("method", item.get("methods", ""))))
                if not method:
                    continue
                answer_items.append({"statistical_method": method})
                bucket = buckets.setdefault(method.casefold(), {"display": method, "pids": set()})
                if pid_i is not None:
                    bucket["pids"].add(pid_i)
            if answer_items:
                evidences.append({"paper_id": paper_id, "answer": _dedupe_preserve_order(answer_items)})

        final_list = [
            {"statistical_method": b["display"], "paper_id": sorted(b["pids"])}
            for _, b in sorted(buckets.items())
        ]

        aggregated["MC_M1.1"] = {
            "evidences": evidences,
            "final_list": final_list,
            "final_answer": len(final_list),
        }
        logger.info("MC_M1.1 post-processed: {} unique methods from {} papers", len(final_list), len(paper_answers))

    # --- MC_M1.2: Count unique variables ---
    if "MC_M1.2" in aggregated:
        paper_answers = _get_per_paper_answers("MC_M1.2")
        evidences = []
        buckets: dict[str, dict] = {}
        for default_idx, (paper_file, ans) in enumerate(paper_answers, 1):
            paper_id = _resolve_paper_id(paper_file, default_idx, file_to_paper_idx)
            pid_i = _pid_as_int(paper_id)
            answer_items = []
            for item in _extract_answer_items(ans, ["variable", "variables"]):
                variable = _strip_citation(item.get("variable", item.get("variables", "")))
                if not variable:
                    continue
                answer_items.append({"variable": variable})
                bucket = buckets.setdefault(variable.casefold(), {"display": variable, "pids": set()})
                if pid_i is not None:
                    bucket["pids"].add(pid_i)
            if answer_items:
                evidences.append({"paper_id": paper_id, "answer": _dedupe_preserve_order(answer_items)})

        final_list = [
            {"variable": b["display"], "paper_id": sorted(b["pids"])}
            for _, b in sorted(buckets.items())
        ]

        aggregated["MC_M1.2"] = {
            "evidences": evidences,
            "final_list": final_list,
            "final_answer": len(final_list),
        }
        logger.info("MC_M1.2 post-processed: {} unique variables from {} papers", len(final_list), len(paper_answers))

    # --- MC_M1.3: Count unique independent variables ---
    if "MC_M1.3" in aggregated:
        paper_answers = _get_per_paper_answers("MC_M1.3")
        evidences = []
        buckets: dict[str, dict] = {}
        for default_idx, (paper_file, ans) in enumerate(paper_answers, 1):
            paper_id = _resolve_paper_id(paper_file, default_idx, file_to_paper_idx)
            pid_i = _pid_as_int(paper_id)
            answer_items = []
            for item in _extract_answer_items(ans, ["independent_variable", "independent_variables", "variable", "variables"]):
                iv = _strip_citation(item.get("independent_variable", item.get("independent_variables", item.get("variable", item.get("variables", "")))))
                if not iv:
                    continue
                answer_items.append({"independent_variable": iv})
                bucket = buckets.setdefault(iv.casefold(), {"display": iv, "pids": set()})
                if pid_i is not None:
                    bucket["pids"].add(pid_i)
            if answer_items:
                evidences.append({"paper_id": paper_id, "answer": _dedupe_preserve_order(answer_items)})

        final_list = [
            {"independent_variable": b["display"], "paper_id": sorted(b["pids"])}
            for _, b in sorted(buckets.items())
        ]

        aggregated["MC_M1.3"] = {
            "evidences": evidences,
            "final_list": final_list,
            "final_answer": len(final_list),
        }
        logger.info("MC_M1.3 post-processed: {} unique IVs from {} papers", len(final_list), len(paper_answers))

    # --- MC_M1.4: Count unique dependent variables ---
    if "MC_M1.4" in aggregated:
        paper_answers = _get_per_paper_answers("MC_M1.4")
        evidences = []
        buckets: dict[str, dict] = {}
        for default_idx, (paper_file, ans) in enumerate(paper_answers, 1):
            paper_id = _resolve_paper_id(paper_file, default_idx, file_to_paper_idx)
            pid_i = _pid_as_int(paper_id)
            answer_items = []
            for item in _extract_answer_items(ans, ["dependent_variable", "dependent_variables", "variable", "variables"]):
                dv = _strip_citation(item.get("dependent_variable", item.get("dependent_variables", item.get("variable", item.get("variables", "")))))
                if not dv:
                    continue
                answer_items.append({"dependent_variable": dv})
                bucket = buckets.setdefault(dv.casefold(), {"display": dv, "pids": set()})
                if pid_i is not None:
                    bucket["pids"].add(pid_i)
            if answer_items:
                evidences.append({"paper_id": paper_id, "answer": _dedupe_preserve_order(answer_items)})

        final_list = [
            {"dependent_variable": b["display"], "paper_id": sorted(b["pids"])}
            for _, b in sorted(buckets.items())
        ]

        aggregated["MC_M1.4"] = {
            "evidences": evidences,
            "final_list": final_list,
            "final_answer": len(final_list),
        }
        logger.info("MC_M1.4 post-processed: {} unique DVs from {} papers", len(final_list), len(paper_answers))

    # --- MC_M2.4: Count unique IV-DV pairs (from M2.4 effect size data) ---
    if "MC_M2.4" in aggregated:
        paper_answers = _get_per_paper_answers("MC_M2.4")
        evidences = []
        pair_buckets: dict[tuple, dict] = {}
        for default_idx, (paper_file, ans) in enumerate(paper_answers, 1):
            paper_id = _resolve_paper_id(paper_file, default_idx, file_to_paper_idx)
            pid_i = _pid_as_int(paper_id)
            answer_items = []
            for item in _extract_answer_items(ans, ["pairs", "independent_variable", "dependent_variable"]):
                iv = _strip_citation(item.get("independent_variable", item.get("IV", "")))
                dv = _strip_citation(item.get("dependent_variable", item.get("DV", "")))
                if not iv or not dv:
                    continue
                answer_items.append({"independent_variable": iv, "dependent_variable": dv})
                key = (iv.casefold(), dv.casefold())
                bucket = pair_buckets.setdefault(key, {"iv": iv, "dv": dv, "pids": set()})
                if pid_i is not None:
                    bucket["pids"].add(pid_i)
            if answer_items:
                evidences.append({"paper_id": paper_id, "answer": _dedupe_preserve_order(answer_items)})

        final_list = [
            {
                "independent_variable": b["iv"],
                "dependent_variable": b["dv"],
                "paper_id": sorted(b["pids"]),
            }
            for _, b in sorted(pair_buckets.items())
        ]

        aggregated["MC_M2.4"] = {
            "evidences": evidences,
            "final_list": final_list,
            "final_answer": len(final_list),
        }
        logger.info("MC_M2.4 post-processed: {} unique IV-DV pairs from {} papers", len(final_list), len(paper_answers))

    # --- MC_M2.6: List effect size pairs with |effect_size| > 0.7 ---
    if "MC_M2.6" in aggregated:
        paper_answers = _get_per_paper_answers("MC_M2.6")
        evidences = []
        final_list = []
        for default_idx, (paper_file, ans) in enumerate(paper_answers, 1):
            paper_id = _resolve_paper_id(paper_file, default_idx, file_to_paper_idx)
            answer_items = []
            for item in _extract_answer_items(ans, ["effect_sizes", "entries", "independent_variable"]):
                iv = _strip_citation(item.get("independent_variable", item.get("IV", "")))
                dv = _strip_citation(item.get("dependent_variable", item.get("DV", "")))
                method = _strip_citation(item.get("statistical_method", ""))
                conditions = _strip_citation(item.get("conditions", ""))
                es = item.get("effect_size")
                try:
                    effect_size = float(es)
                except (TypeError, ValueError):
                    continue
                normalized = {
                    "independent_variable": iv,
                    "dependent_variable": dv,
                    "statistical_method": method,
                    "conditions": conditions,
                    "effect_size": effect_size,
                }
                answer_items.append(normalized)
                if abs(effect_size) > 0.7:
                    final_list.append({"paper_id": paper_id, **normalized})
            if answer_items:
                evidences.append({"paper_id": paper_id, "answer": _dedupe_preserve_order(answer_items)})

        final_list = _dedupe_preserve_order(final_list)
        aggregated["MC_M2.6"] = {
            "evidences": evidences,
            "final_list": final_list,
            "final_answer": len(final_list),
        }
        logger.info("MC_M2.6 post-processed: {} entries with |effect|>0.7 from {} total",
                    len(final_list), sum(len(e["answer"]) for e in evidences))
    
    return aggregated


def aggregate_results(
    client: OpenAI,
    model: str,
    per_paper_results: dict[str, dict],
    questions: list[dict],
    temperature: float = 0.1,
    parallel: int = 100,
    file_to_paper_idx: dict[str, int] | None = None,  # {filename: paper_idx}
    audit_dirs: dict[str, Path] | None = None,
) -> dict:
    """
    Use LLM to aggregate per-paper results into deduplicated lists.
    Processes questions in parallel.
    """
    logger.info("Aggregating results from {} papers using LLM (parallel={})...", len(per_paper_results), parallel)
    
    def aggregate_single_question(q):
        q_id = q["id"]
        q_text = q.get("original_question", q["question"])
        single_paper_q_text = q["question"]
        
        # Collect all answers for this question with paper_id
        # Sort papers by filename to ensure consistent order
        sorted_papers = sorted(per_paper_results.items(), key=lambda x: x[0])
        paper_answers = []
        for paper_file, answers in sorted_papers:
            if q_id in answers and answers[q_id]:
                paper_answers.append({
                    "paper_id": _resolve_paper_id(paper_file, 1, file_to_paper_idx),
                    "paper": paper_file,
                    "answer": answers[q_id],
                })
        
        if not paper_answers:
            return q_id, {}
        
        is_minimax = "minimax" in model.lower()
        is_oc = q_id.startswith("OC_")

        if not is_oc:
            # Regular C question — enforce C schema and one primary entry per paper
            system_prompt = """You are a data aggregation assistant.

Your task: combine single-paper answers into one final multi-paper result.

You will receive:
1. The ORIGINAL multi-paper question, which defines the real evaluation target.
2. The REWRITTEN single-paper question, which was only used to collect per-paper evidence.
3. The extracted answers from each paper.

CRITICAL RULES:
1. The ORIGINAL multi-paper question is the source of truth for aggregation, normalization, deduplication, and counting.
2. Treat the REWRITTEN single-paper question only as context for what each per-paper answer means.
3. Do not keep placeholders, templates, ellipses, or instruction text as extracted values.

OUTPUT SHAPE — here is a concrete shape example (the field name `color` and values `"red"`, `"blue"`, `"green"` are fictitious placeholders used only to illustrate the JSON structure; your real output must use the field names from the ORIGINAL question and real values from the per-paper answers below):

{
  "answers": [
    {"paper_id": 1, "answer": [{"color": "red"}]},
    {"paper_id": 2, "answer": [{"color": "blue"}]},
    {"paper_id": 3, "answer": [{"color": "red"}]}
  ]
}

- `answers` is the only top-level key. It is always a LIST of per-paper entries.
- `paper_id` is always an integer (e.g. 1, 2, 3), taken from the `paper_id` field of the per-paper inputs.
- `answer` is always a LIST whose items are dicts; each dict's keys are the field names required by the ORIGINAL question.
- Do NOT emit `evidences`, `final_list`, or `final_answer` — those belong to OC questions only, this is a C question.
- Do NOT wrap the output as a bare `{"paper_id": ..., "answer": [...]}` dict — always use the `{"answers": [...]}` envelope with all papers inside.
- Do NOT copy `color`/`red`/`blue`/`green` into your output; those are illustration-only tokens.

REDUCTION RULE — keep the primary answer per paper:
- When a paper's per-paper output enumerates multiple sub-items (e.g. all sub-regions, all variable names, all nested categories mentioned in the paper), DISTILL these down to the single most representative / top-level / primary entry that directly answers the ORIGINAL question for that paper.
- Prefer the broader/higher-level answer over its sub-components. Example: if a paper's per-paper answer lists "Indonesia, Sumatra, Borneo, Java", keep only "Indonesia". If it lists "quarter-degree square, biome, catchment, vegetation type", pick the single primary study unit.
- Only keep more than one item per paper when the paper truly covers multiple distinct primary entities that cannot be subsumed under a single one (e.g. a cross-country comparative study of two countries).
- Do not invent content; only select/merge from what the per-paper inputs actually contain.

Output one complete JSON object only."""
        else:
            # OC question — count/list aggregation schema
            base_rules = """You are a data aggregation assistant.

Your task: combine single-paper answers into one final multi-paper result.

You will receive:
1. The ORIGINAL multi-paper question, which defines the real evaluation target.
2. The REWRITTEN single-paper question, which was only used to collect per-paper evidence.
3. The extracted answers from each paper.

CRITICAL RULES:
1. The ORIGINAL multi-paper question is the source of truth for aggregation, normalization, deduplication, and counting.
2. Treat the REWRITTEN single-paper question only as context for what each per-paper answer means.
3. For `evidences`, preserve paper-level provenance. Keep the PRIMARY answer per paper — if a paper lists multiple sub-items, distill to the single most representative entry unless the paper truly reports distinct primary entities.
4. For `final_list` and `final_answer`, do cross-paper normalization and deduplication exactly as required by the ORIGINAL question.
5. Do not keep placeholders, templates, ellipses, or instruction text as extracted values.
6. If the ORIGINAL question says to normalize to countries / unique items / explicit mentions only, apply that rule when building `final_list` and `final_answer`.
"""
            if is_minimax:
                schema_block = """
## final_list rule
- `final_list` is a LIST OF DICTS. Each entry: semantic fields FIRST (e.g., `country`, `sample_size`, `statistical_method`, `independent_variable`, `dependent_variable`), `paper_id` LAST.
- `paper_id` is a JSON ARRAY of integer paper indices (e.g., `[3]`, `[1, 2, 5]`). Use an array even for a single paper. Sort ascending, drop duplicates.
- Merge cross-paper duplicates into ONE entry and UNION their paper_ids.
- Do NOT put inline `[N]` citation markers inside the semantic-field values.
- Example: `{"country": "Brazil", "paper_id": [1, 5]}`.

Use {"evidences", "final_list", "final_answer"} as the top-level keys. Do NOT emit `answers`.
Follow the Output Format in the ORIGINAL question exactly.
Output one complete JSON object only."""
            else:
                schema_block = """
## REQUIRED OUTPUT SHAPE
The top-level JSON MUST have EXACTLY three keys — `evidences`, `final_list`, `final_answer`.
Never return a bare list, a flat single dict, or only part of this structure.

```json
{
  "evidences": [ {"paper_id": <int>, "answer": [ {...schema fields...} ]}, ... ],
  "final_list": [ { <semantic fields first>, "paper_id": [<int>, ...] }, ... ],
  "final_answer": <number>
}
```

## final_list rules
- Each entry is a DICT. Semantic fields (`country`, `sample_size`, `statistical_method`, `independent_variable`, `dependent_variable`, etc.) come FIRST; `paper_id` comes LAST.
- `paper_id` is always a JSON ARRAY of integer paper indices: `[3]`, `[1, 2, 5]`. Use an array even when a single paper contributes.
- Merge cross-paper duplicates into ONE entry and UNION their `paper_id`.
- Sort `paper_id` ascending; drop duplicates.
- Do NOT put inline citation markers like `[1]` inside the semantic-field values — provenance belongs in `paper_id`.

## Worked example (for 'count unique countries')
```json
{
  "evidences": [
    {"paper_id": 1, "answer": [{"geolocation": "Brazil [1]"}]},
    {"paper_id": 5, "answer": [{"geolocation": "Brazil [5]"}]},
    {"paper_id": 7, "answer": [{"geolocation": "Kenya [7]"}]}
  ],
  "final_list": [
    {"country": "Brazil", "paper_id": [1, 5]},
    {"country": "Kenya", "paper_id": [7]}
  ],
  "final_answer": 2
}
```

Do NOT emit `answers` as a top-level key (that's for C-type extraction, not for OC/MC computation).
Follow the Output Format in the ORIGINAL question exactly.
Output one complete JSON object only."""
            system_prompt = base_rules + schema_block

        if is_minimax:
            system_prompt += """

ABSOLUTE REQUIREMENT — READ THE PER-PAPER ANSWERS BELOW AND USE THEIR REAL CONTENT.
- The Output Format shown in the ORIGINAL question is a schema, not an example. Placeholder tokens like `<id>`, `<number_and_unit>`, `<country>`, `<value>`, `...`, or any angle-bracketed/ellipsis field are STRUCTURE HINTS ONLY. NEVER write them into your output.
- Every value in your output MUST come from the actual per-paper answers provided in the user message. If a paper's answer is `sample_size: "11080 students"`, your output must contain `"11080"` (or `"11080 students"`), not `"<number_and_unit>"`.
- Do NOT copy, paraphrase, or echo the schema's placeholder strings under any circumstance.
- Before writing each field, ask: "which paper's real answer is this value coming from?" If you cannot name the source paper, the value is wrong — delete it.
- If a per-paper answer itself is a placeholder (e.g. `<id>`), drop that paper; do not forward the placeholder to your output."""

        answers_text = json.dumps(paper_answers, indent=2, ensure_ascii=False)

        if not is_oc:
            requirements = """Requirements:
1. Produce exactly one entry per paper inside the top-level `answers` list, preserving paper_id provenance.
2. Each paper's `answer` field is a list of dicts whose keys match the ORIGINAL question's schema fields.
3. Apply the REDUCTION RULE: if a paper's per-paper answer contains multiple sub-items that describe the same study at different granularities, keep only the single primary/top-level entry. Prefer the broadest correct answer (e.g. "Indonesia" over "Sumatra/Borneo/Java"). Keep multiple entries only when the paper truly reports distinct primary entities (e.g. an explicit cross-country comparative study).
4. Exclude placeholders or malformed values such as `<...>`, `...`, or instruction-like text.
5. Do NOT emit `evidences`, `final_list`, or `final_answer` — this is a C question, the only top-level key is `answers`.
6. Strip inline citation markers like `[1]`, `[2]`, `[3]` from the final answer strings (they exist only as provenance hints in the per-paper inputs; the ORIGINAL question never asks for them). For example, rewrite `"South Africa [1]"` as `"South Africa"` and `"quarter-degree square (QDS) [1]"` as `"quarter-degree square (QDS)"`.
7. Numeric normalization: when a field is plainly a count, size, or measurement (e.g. `sample_size`, `n`, `count`, `population_size`, anything whose per-paper values look like `"2014"`, `"1485 grid cells"`, `"131 cities"`, `"10,500 respondents"`, `"11,080 students"`), emit the value as a JSON number with the trailing unit/descriptor stripped. `"2014 QDS"` → `2014`, `"1,485 grid cells"` → `1485`, `"131 cities"` → `131`. If the per-paper value is `"not reported"`, an empty string, or otherwise non-numeric, keep the original string verbatim (do NOT invent a number)."""
        else:
            requirements = """Requirements:
1. Build `evidences` from the paper-level answers, preserving source paper identity. Keep the PRIMARY answer per paper — distill multi-item per-paper outputs to the single most representative entry unless the paper truly reports distinct primary entities.
2. Read the ORIGINAL multi-paper question carefully and derive `final_list` from the evidence items by following that question's step-by-step rules.
3. Deduplicate across papers when the ORIGINAL question asks for unique entities.
4. Exclude placeholders or malformed values such as `<...>`, `...`, or instruction-like text.
5. For OC questions, compute `final_answer` from the derived `final_list`, not from the raw count of evidence rows.
6. Strip inline citation markers like `[1]`, `[2]`, `[3]` from all answer strings in `evidences` and `final_list` (they exist only as provenance hints in the per-paper inputs; use the `paper_id`/structural fields for provenance, not inline markers).
7. Numeric normalization for count/size/measurement fields: when a field is plainly numeric (`sample_size`, `n`, `count`, `population_size`, …), emit it as a JSON number with trailing unit/descriptor stripped. `"2014 QDS"` → `2014`, `"1,485 grid cells"` → `1485`, `"131 cities"` → `131`. If the value is `"not reported"` or non-numeric, keep the original string."""

        user_content = f"""Original multi-paper question:
{q_text}

Rewritten single-paper extraction question:
{single_paper_q_text}

Answers from {len(paper_answers)} papers:
{answers_text}

Aggregate into a single JSON result.

{requirements}

Make sure your JSON is complete and properly closed."""
        
        # Retry logic for JSON parsing failures
        max_retries = 5
        last_error = None
        
        for attempt in range(max_retries):
            try:
                token_param = {"max_tokens": 40960, "extra_body": _build_extra_body()} if not model.startswith("gpt") else {"max_completion_tokens": 40960}
                
                response = _safe_create(client,
                    model=model,
                    temperature=temperature,
                    **token_param,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                
                if response is None:
                    logger.warning("Skipping aggregation for {} due to context length", q_id)
                    break

                full_response = _extract_content(response)
                
                # Log token usage for aggregation
                if hasattr(response, 'usage') and response.usage:
                    logger.debug("Aggregate {}: prompt={}, completion={}", 
                               q_id, response.usage.prompt_tokens, response.usage.completion_tokens)
                
                delimiter = "===AGGREGATED==="
                if delimiter in full_response:
                    json_part = full_response.split(delimiter)[-1].strip()
                else:
                    json_part = full_response
                
                result = parse_json_response(json_part)
                save_audit_record(audit_dirs, "aggregation", q_id, {
                    "mode": "aggregation",
                    "aggregation_type": "llm",
                    "question_id": q_id,
                    "input_paper_answers": paper_answers,
                    "raw_text": full_response,
                    "parsed_answer": result,
                })
                
                # Check if result is valid (not empty when we have answers)
                if result or not paper_answers:
                    logger.debug("Aggregated {}: {} items (attempt {})", q_id, len(result), attempt + 1)
                    return q_id, result
                else:
                    last_error = "Empty result despite having answers"
                    logger.warning("{} attempt {}: {}", q_id, attempt + 1, last_error)
                    
            except Exception as e:
                last_error = str(e)
                if "timed out" in last_error.lower() or "timeout" in last_error.lower():
                    logger.warning("{}: aggregation timed out — skipping (no retry)", q_id)
                    break
                logger.warning("{} attempt {} failed: {}", q_id, attempt + 1, last_error)

        logger.error("Failed to aggregate {} after {} retries: {}", q_id, max_retries, last_error)
        return q_id, {}
    
    # All questions go through LLM aggregation (with deduplication)
    aggregated = {}
    if questions:
        logger.info("Using LLM aggregation for all {} questions", len(questions))
        if parallel > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(aggregate_single_question, q): q["id"] for q in questions}
                for future in concurrent.futures.as_completed(futures):
                    q_id, result = future.result()
                    aggregated[q_id] = result
        else:
            for q in questions:
                q_id, result = aggregate_single_question(q)
                aggregated[q_id] = result
    
    # Post-process computation queries: compute final scalar from per-paper data
    aggregated = computation_post_process(aggregated, per_paper_results, file_to_paper_idx)
    
    return aggregated


def decompose_single_question(
    client: OpenAI,
    model: str,
    question: dict,
    temperature: float = 0.1,
) -> str:
    """
    Transform an aggregated question into a single-paper question.
    
    For OC/MC computation queries: uses LLM to extract the per-paper extraction
    step from the multi-step prompt (keeping Step 1, removing cross-paper 
    aggregation steps).
    
    For regular queries: uses deterministic rule-based text replacement.
    """
    import re
    
    q_id = question["id"]
    q_text = question["question"]
    
    # --- OC/MC computation queries: LLM-based decomposition ---
    is_computation = q_id.startswith("OC_") or q_id.startswith("MC_")
    if is_computation:
        # Per-paper Output Format: matches what ask_paper_question returns and the
        # aggregator expects — {"answers": [{"paper_id": "<id>", "answer": [ ... ]}]}.
        # Shapes that omit the `answers` wrapper (e.g. {"geolocation": "..."}) break
        # downstream aggregation, so all schemas are wrapped uniformly.
        # Schemas use CONCRETE EXAMPLE VALUES (not angle-bracket placeholders) because some
        # models (e.g., Qwen9b) echo literal <placeholder> strings as output. Concrete
        # example values like "Brazil" / "age (AGE)" / "0.35" force the model to recognize
        # these as shape examples to be replaced with the paper's actual values.
        output_schemas = {
            "OC_O1.1":   '{"answers": [{"paper_id": "1", "answer": [{"geolocation": "Beijing, China"}]}]}',
            "OC_O1.2.1": '{"answers": [{"paper_id": "1", "answer": [{"sample_size": "1,234 participants"}]}]}',
            "OC_O1.2.2": '{"answers": [{"paper_id": "1", "answer": [{"sample_size": "1,234 participants"}]}]}',
            "OC_O1.2.3": '{"answers": [{"paper_id": "1", "answer": [{"sample_size": "1,234 participants"}]}]}',
            "MC_M1.1":   '{"answers": [{"paper_id": "1", "answer": [{"statistical_method": "Pearson correlation"}, {"statistical_method": "Logistic regression"}]}]}',
            "MC_M1.2":   '{"answers": [{"paper_id": "1", "answer": [{"variable": "age"}, {"variable": "body mass index (BMI)"}]}]}',
            "MC_M1.3":   '{"answers": [{"paper_id": "1", "answer": [{"independent_variable": "age"}, {"independent_variable": "socioeconomic status (SES)"}]}]}',
            "MC_M1.4":   '{"answers": [{"paper_id": "1", "answer": [{"dependent_variable": "body mass index (BMI)"}, {"dependent_variable": "blood pressure"}]}]}',
            "MC_M2.4":   '{"answers": [{"paper_id": "1", "answer": [{"independent_variable": "age", "dependent_variable": "blood pressure"}, {"independent_variable": "gender", "dependent_variable": "blood pressure"}]}]}',
            "MC_M2.6":   '{"answers": [{"paper_id": "1", "answer": [{"independent_variable": "age", "dependent_variable": "blood pressure", "statistical_method": "Pearson correlation (r)", "conditions": "none", "effect_size": 0.35}]}]}',
        }

        schema = output_schemas.get(q_id, '{}')

        decompose_prompt = (
            "TASK: Rewrite a MULTI-paper meta-analysis question so it applies to ONE paper.\n\n"
            "YOUR OUTPUT = the REWRITTEN QUESTION as plain natural-language markdown text. "
            "It will later be fed to another LLM as a user prompt for single-paper extraction, "
            "so it must READ LIKE A QUESTION / INSTRUCTION — not a JSON answer.\n\n"
            "The ORIGINAL question below has multiple steps:\n"
            "  * Step 1 extracts raw data FROM EACH paper.\n"
            "  * Step 2 / 3 / 4 aggregate, count, average, filter, or deduplicate ACROSS papers.\n\n"
            "Rules for your rewrite:\n"
            "  1. Keep ONLY Step-1 extraction logic. DROP all cross-paper aggregation, counting, "
            "averaging, median, threshold filtering, deduplication, or 'final list' steps.\n"
            "  2. Replace 'all papers' / 'this collection' / 'across papers' with language about "
            "THIS single paper.\n"
            "  3. Keep ALL Step-1 extraction principles, rules, and exclusion criteria VERBATIM.\n"
            "  4. End the rewritten question with a section titled '## Output Format' that "
            "contains EXACTLY the JSON block below, inside a ```json fence, copied verbatim. "
            "(The values in the block are CONCRETE EXAMPLES so the extractor knows what shape "
            "to emit; the extractor will replace them with the actual paper's data.)\n\n"
            "REQUIRED Output Format block to include at the end of your rewrite:\n"
            "```json\n"
            f"{schema}\n"
            "```\n\n"
            "After the json fence, append ONE short sentence exactly like:\n"
            '  "Replace the example values above with the actual data extracted from the paper."\n\n'
            "IMPORTANT: Your response is the rewritten question (markdown). Do NOT output a JSON "
            "object as your response. The JSON block above is CONTENT that must appear inside "
            "your rewritten question, under its '## Output Format' block.\n\n"
            "## Original Question:\n"
            f"{q_text}\n\n"
            "## Rewritten Single-Paper Question (your response, ending with a '## Output Format' block):\n"
        )

        try:
            token_param = {}
            if model.startswith("gpt"):
                token_param = {"max_completion_tokens": 2048}
            else:
                # For open-source models (Qwen, DeepSeek, etc.), disable thinking mode
                # to avoid reasoning-leak into the decomposed question output.
                token_param = {"max_tokens": 4096, "extra_body": _build_extra_body()}

            response = _safe_create(
                client,
                model=model,
                temperature=temperature,
                **token_param,
                messages=[
                    {"role": "system", "content": (
                        "You rewrite a multi-paper research question into a SINGLE-paper version. "
                        "Your reply is the rewritten question as plain markdown text — never a JSON answer. "
                        "The rewritten question must end with a '## Output Format' section containing the "
                        "JSON schema provided in the user message (copied verbatim inside a ```json fence). "
                        "Do NOT include any reasoning, 'think' traces, or meta-commentary in your reply."
                    )},
                    {"role": "user", "content": decompose_prompt}
                ],
            )
            if response:
                result = _extract_content(response)
                if result:
                    # Strip reasoning-mode markers (Qwen, DeepSeek, etc. emit <think>...</think>)
                    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
                    # Some models leak a bare "<think>..." without a closing tag on truncation
                    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
                    result = result.strip()
                    # Validation: a proper rewritten single-paper question should be
                    # comparable-length to the original (not wildly longer from leaked
                    # reasoning). Reject if output is >3x original length.
                    orig_len = len(q_text)
                    has_output_fmt = "Output Format" in result
                    leaked_reasoning = (
                        len(result) > 3 * orig_len
                        or re.search(r"^\s*(Wait[,.]|Let me|Okay[,.]|I need to|Let's (write|reconsider)|One more check)",
                                     result, flags=re.MULTILINE)
                    )
                    if len(result) > 150 and has_output_fmt and not leaked_reasoning:
                        logger.debug("LLM-decomposed {}: {} -> {} (len={})", q_id, q_text[:40], result[:60], len(result))
                        return result
                    logger.warning(
                        "LLM decomposition for {} looked degenerate "
                        "(len={}, orig_len={}, Output Format present={}, reasoning_leak={})",
                        q_id, len(result), orig_len, has_output_fmt, bool(leaked_reasoning)
                    )
        except Exception as e:
            logger.error("LLM decomposition failed for {}: {}", q_id, e)

        # No fallback: for OC/MC, LLM decomposition is the ONLY supported path.
        # Raise so the caller can mark this question as broken and the user can rerun it.
        raise RuntimeError(
            f"LLM decomposition for {q_id} did not produce a valid rewritten question. "
            f"This question must be rerun; no rule-based fallback is used for OC/MC."
        )

    # --- Regular (non-OC/MC) queries: rule-based text replacement ---
    RULES = [
        (r'\bfrom these papers\b', 'from this paper'),
        (r'\bthese papers\b', 'this paper'),
        (r'\bacross all studies\b', 'from this paper'),
        (r'\ball papers\b', 'this paper'),
    ]
    
    result = q_text
    for pattern, replacement in RULES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    if q_id in HIGH_DENSITY_QUESTIONS:
        result = re.sub(r'\bExtract all\b', 'Extract ALL', result, flags=re.IGNORECASE)
        result = re.sub(r'\bList all\b', 'Extract ALL', result, flags=re.IGNORECASE)
        
        if 'extract ALL' in result.lower() and 'multiple' not in result.lower():
            if 'Output Format' in result:
                result = result.replace(
                    'Output Format',
                    'This paper may contain MULTIPLE items - extract ALL of them.\n\nOutput Format'
                )
    else:
        result = re.sub(r'\bExtract all\b', 'Extract the', result, flags=re.IGNORECASE)
        result = re.sub(r'\bList all\b', 'Extract the', result, flags=re.IGNORECASE)
        
        result = re.sub(
            r'"(\w+)":\s*\[\s*"<([^"]+)>"(?:,\s*\.\.\.)?s*\]',
            r'"\1": "<\2>"',
            result
        )
        result = re.sub(
            r'"(\w+)":\s*\[\s*(\{[^}]+\})(?:,\s*\.\.\.)?s*\]',
            r'"\1": \2',
            result
        )
    
    logger.debug("Rule-decomposed {}: {} -> {}", q_id, q_text[:40], result[:40])
    return result


def decompose_questions_for_single_paper(
    client: OpenAI,
    model: str,
    questions: list[dict],
    temperature: float = 0.1,
    parallel: int = 100,
) -> list[dict]:
    """
    Transform all aggregated questions into single-paper questions.
    Processes questions in parallel for speed.
    """
    logger.info("Decomposing {} questions into per-paper format (parallel={})...", len(questions), parallel)
    
    def decompose_q(q):
        rewritten_text = decompose_single_question(client, model, q, temperature)
        new_q = q.copy()
        new_q["question"] = rewritten_text
        new_q["original_question"] = q["question"]
        return q["id"], new_q
    
    results = {}
    failed = []
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(decompose_q, q): q["id"] for q in questions}
            for future in concurrent.futures.as_completed(futures):
                q_id = futures[future]
                try:
                    _, new_q = future.result()
                    results[q_id] = new_q
                except Exception as e:
                    # No fallback: drop the question so downstream per-paper extraction
                    # skips it entirely. User can identify the missing qids in the final
                    # predictions.json and rerun just those.
                    logger.error("DROP {} from per-paper run — decomposition failed: {}", q_id, e)
                    failed.append(q_id)
    else:
        for q in questions:
            try:
                _, new_q = decompose_q(q)
                results[q["id"]] = new_q
            except Exception as e:
                logger.error("DROP {} from per-paper run — decomposition failed: {}", q["id"], e)
                failed.append(q["id"])

    # Maintain original order; dropped questions are absent from the returned list.
    new_questions = [results[q["id"]] for q in questions if q["id"] in results]

    logger.info("Decomposed {}/{} questions; {} dropped: {}",
                len(new_questions), len(questions), len(failed), sorted(failed))
    return new_questions


def run_map_reduce_extraction(
    client: OpenAI,
    model: str,
    docs: list[dict],
    questions: list[dict],
    temperature: float,
    max_tokens: int,
    parallel_docs: int = 10,
    parallel_questions: int = 10,
    audit_dirs: dict[str, Path] | None = None,
) -> dict:
    """Run map-reduce extraction: per-paper then aggregate."""
    
    logger.info("Running MAP-REDUCE extraction: {} questions × {} papers", 
                len(questions), len(docs))
    
    # Step 1: Decompose aggregated questions into per-paper format
    per_paper_questions = decompose_questions_for_single_paper(
        client,
        model,
        questions,
        temperature,
        parallel=parallel_questions,
    )
    decomposed_questions_log = [
        {
            "question_id": q["id"],
            "original": q.get("original_question", q["question"]),
            "decomposed": q["question"],
        }
        for q in per_paper_questions
    ]
    
    per_paper_results = {}
    
    # Step 2: Process each paper with the decomposed questions
    def process_paper(doc):
        paper_results = {}
        for q in per_paper_questions:
            result = ask_paper_question(client, model, doc, q, temperature, max_tokens, audit_dirs=audit_dirs)
            paper_results[q["id"]] = result
        return doc["file"], paper_results
    
    if parallel_docs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_docs) as executor:
            future_to_doc = {executor.submit(process_paper, doc): doc["file"] for doc in docs}
            for future in concurrent.futures.as_completed(future_to_doc):
                try:
                    paper_file, results = future.result()
                    per_paper_results[paper_file] = results
                    logger.info("Completed paper: {}", paper_file)
                except Exception as e:
                    logger.exception("Failed paper: {}", e)
    else:
        for doc in docs:
            paper_file, results = process_paper(doc)
            per_paper_results[paper_file] = results
    
    # Aggregate using LLM
    logger.info("Aggregating results from {} papers...", len(per_paper_results))
    
    # Build file_to_paper_idx mapping for correct citation assignment
    file_to_paper_idx = {doc["file"]: doc.get("paper_idx", 1) for doc in docs}
    aggregated = aggregate_results(
        client,
        model,
        per_paper_results,
        questions,
        temperature,
        parallel=parallel_questions,
        file_to_paper_idx=file_to_paper_idx,
        audit_dirs=audit_dirs,
    )
    
    # Attach per_paper_results and decomposed questions so caller can save them
    aggregated["__per_paper_results__"] = per_paper_results
    aggregated["__decomposed_questions__"] = decomposed_questions_log

    return aggregated



# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    outputs_dir: Path,
    questions_config: Path,
    client: OpenAI,
    model: str,
    mode: str = "global",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_chars: int = 40000,
    max_docs: int | None = None,
    include_images: bool = True,
    parallel_questions: int = 3,
    parallel_docs: int = 2,
    output_dir: Path | None = None,
) -> dict:
    """
    Run the extraction pipeline.
    
    Extracts answers from papers and saves predictions.json.
    Evaluation should be done separately via scripts/eval/audit_predictions.py.
    
    Args:
        mode: "global" (all papers at once) or "map-reduce" (per-paper then aggregate)
    """
    
    # Load questions
    questions = load_questions(questions_config)
    logger.info("Loaded {} questions from {}", len(questions), questions_config)
    
    # Load documents
    docs = build_doc_contexts(outputs_dir, max_docs, max_chars, include_images)
    
    # Run extraction
    if mode == "global":
        predictions = run_global_extraction(
            client, model, docs, questions, temperature, max_tokens, parallel_questions
        )
    else:  # map-reduce
        predictions = run_map_reduce_extraction(
            client, model, docs, questions, temperature, max_tokens, parallel_docs, parallel_questions
        )
    
    # Output directory
    save_dir = output_dir or outputs_dir.parent / f"results_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_path = save_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Saved predictions to {}", pred_path)
    
    return predictions


# ---------------------------------------------------------------------------
# RETRY & MERGE UTILITIES (Integrated from scripts/)
# ---------------------------------------------------------------------------

def count_items(result: dict | list) -> int:
    """Count total items in a prediction result.
    
    Handles various formats:
    - {"key": [...items...]}: returns len(items)
    - [...items...]: returns len(items)
    - {}: returns 0
    
    Returns:
        Number of items in the result
    """
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                return len(v)
            elif isinstance(v, (int, float)):
                return 1 if v else 0
    elif isinstance(result, list):
        return len(result)
    return 0


def run_with_retry(
    client: OpenAI,
    model: str,
    docs: list[dict],
    question: dict,
    max_attempts: int = 3,
    success_threshold: int = 1,
    temperature: float = 0.1,
    max_tokens: int = 16384,
) -> dict:
    """Run a single question extraction with retry logic.
    
    If the initial extraction returns fewer items than success_threshold,
    retries up to max_attempts times and keeps the best result.
    
    Args:
        client: OpenAI client
        model: Model name
        docs: List of document contexts
        question: Question dict with 'id' and 'question' fields
        max_attempts: Maximum number of extraction attempts
        success_threshold: Minimum items needed to consider success
        temperature: Model temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Best result dict (with most items)
    """
    q_id = question["id"]
    best_count = 0
    best_result = {}
    
    for attempt in range(1, max_attempts + 1):
        logger.info("{} attempt {}/{}", q_id, attempt, max_attempts)
        
        try:
            result = ask_global_question(
                client=client,
                model=model,
                docs=docs,
                question=question,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            item_count = count_items(result)
            logger.info("{} attempt {}: {} items", q_id, attempt, item_count)
            
            if item_count > best_count:
                best_count = item_count
                best_result = result
            
            if item_count >= success_threshold:
                logger.info("{}: Success with {} items, stopping retries", q_id, item_count)
                break
                
        except Exception as e:
            err_msg = str(e)
            if "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                logger.warning("{}: timed out — skipping (no retry)", q_id)
                break
            logger.error("{} attempt {} failed: {}", q_id, attempt, e)
    
    logger.info("{} FINAL: {} items after {} attempts", q_id, best_count, max_attempts)
    return best_result


def merge_predictions(
    base: dict,
    *updates: dict,
    strategy: str = "best_count",
) -> dict:
    """Merge multiple prediction dicts, keeping best results.
    
    For each task_id, compares item counts and keeps the version
    with more items (or latest if counts are equal).
    
    Args:
        base: Base predictions dict
        *updates: One or more update dicts to merge in
        strategy: Merge strategy - "best_count" (default) or "latest"
        
    Returns:
        Merged predictions dict
    """
    merged = dict(base)  # Copy base
    
    for update in updates:
        if not update:
            continue
            
        for task_id, result in update.items():
            # Skip error results
            if isinstance(result, dict) and "error" in result:
                continue
                
            new_count = count_items(result)
            old_count = count_items(merged.get(task_id, {}))
            
            if strategy == "best_count":
                if new_count > old_count:
                    logger.debug("merge_predictions: {} updated {} -> {} items", 
                                task_id, old_count, new_count)
                    merged[task_id] = result
            elif strategy == "latest":
                merged[task_id] = result
                
    return merged


def build_ordered_docs_from_gt(
    docs_dir: Path,
    gt_path: Path,
    max_chars: int = 40000,
    include_images: bool = True,
    max_docs: int | None = None,
) -> list[dict]:
    """Build document contexts ordered by ground truth paper indices.
    
    Reads the ground truth JSON to extract paper titles and their citation
    indices, then builds document contexts in that exact order.
    
    This ensures [1], [2], etc. in extractions match the GT's paper ordering.
    
    Args:
        docs_dir: Path to MinerU markdown exports
        gt_path: Path to ground truth JSON (must have paper_index field)
        max_chars: Maximum characters per document
        include_images: Whether to include images
        max_docs: Maximum number of documents
        
    Returns:
        List of document contexts ordered by GT paper indices
    """
    # Load GT and extract paper index
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    # Build paper index from GT: title -> paper_idx
    gt_paper_index = {}
    
    def normalize_title(title: str) -> str:
        import re
        title = title.lower().strip()
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', ' ', title)
        return title
    
    # Try different GT formats
    if isinstance(gt_data, dict):
        for task_id, task_data in gt_data.items():
            if isinstance(task_data, dict):
                for key, items in task_data.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                paper_idx = item.get("paper_index")
                                title = item.get("title") or item.get("paper_title")
                                if paper_idx and title:
                                    gt_paper_index[normalize_title(title)] = str(paper_idx)
    
    if not gt_paper_index:
        logger.warning("Could not extract paper index from GT, using filesystem order")
        return build_doc_contexts(docs_dir, max_chars=max_chars, 
                                   include_images=include_images, max_docs=max_docs)
    
    logger.info("Found {} papers in GT index", len(gt_paper_index))
    
    # Use existing build_doc_contexts with gt_paper_index
    return build_doc_contexts(
        outputs_dir=docs_dir,
        max_chars=max_chars,
        include_images=include_images,
        max_docs=max_docs,
        gt_paper_index=gt_paper_index,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregated QA Extraction Pipeline"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        required=True,
        help="Directory containing MinerU markdown exports",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parent / "standardized_config.json",
        help="Path to standardized questions config",
    )
    parser.add_argument(
        "--mode",
        choices=["global", "map-reduce"],
        default="global",
        help="Extraction mode: global (all papers at once) or map-reduce (per-paper then aggregate)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (auto-detected from vLLM server if not specified)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-chars", type=int, default=40000)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--no-images", action="store_true", help="Disable image inclusion (images included by default)")
    parser.add_argument("--parallel-questions", type=int, default=10)
    parser.add_argument("--parallel-docs", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key, base_url=args.api_base)
    
    # Auto-detect model if not specified
    model = args.model
    if not model:
        model = detect_model(client)
        if not model:
            raise ValueError("Could not auto-detect model. Please specify --model.")
    
    predictions = run_pipeline(
        outputs_dir=args.outputs_dir,
        questions_config=args.questions,
        client=client,
        model=model,
        mode=args.mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_chars=args.max_chars,
        max_docs=args.max_docs,
        include_images=not args.no_images,
        parallel_questions=args.parallel_questions,
        parallel_docs=args.parallel_docs,
        output_dir=args.output_dir,
    )
    
    total_items = sum(
        len(v) if isinstance(v, list) else sum(len(vv) for vv in v.values() if isinstance(vv, list))
        for v in predictions.values() if v
    )
    print(f"\nExtraction complete: {len(predictions)} questions, {total_items} total items")
    print("Run scripts/eval/audit_predictions.py to evaluate.")


if __name__ == "__main__":
    main()
