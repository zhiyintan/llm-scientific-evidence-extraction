from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

from loguru import logger
from openai import OpenAI

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional at runtime
    Image = None


def iter_markdown_files(outputs_dir: Path, skip_suffixes: Sequence[str]) -> Iterable[Path]:
    suffixes = tuple(skip_suffixes)
    for md_path in sorted(outputs_dir.glob("**/*.md")):
        name = md_path.name
        if suffixes and any(name.endswith(suffix) for suffix in suffixes):
            logger.debug("Skip %s due to suffix filter", md_path)
            continue
        yield md_path


def read_markdown(md_path: Path, max_chars: int | None) -> str:
    text = md_path.read_text(encoding="utf-8")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n...\n"
    return text


def _to_data_url(path: Path, data: bytes) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)")


def _append_text_block(blocks: list[dict], chunk: str) -> None:
    if not chunk:
        return
    if blocks and blocks[-1].get("type") == "text":
        blocks[-1]["text"] += chunk
    else:
        blocks.append({"type": "text", "text": chunk})


def _normalize_image_target(target: str) -> str:
    cleaned = target.strip()
    if cleaned.startswith("<") and cleaned.endswith(">"):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return ""
    # Drop optional title e.g. path "title"
    if " " in cleaned:
        cleaned = cleaned.split()[0]
    return cleaned


def _prepare_image_bytes(image_path: Path, data: bytes, max_image_bytes: int | None) -> tuple[bytes, str]:
    """Downscale/compress images before sending them to the model."""
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime is None:
        mime = "image/jpeg"

    if Image is None:
        return data, mime

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            longest_edge = max(img.size)
            if longest_edge > 768:
                scale = 768 / float(longest_edge)
                new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            out = io.BytesIO()
            if has_alpha:
                img.save(out, format="PNG", optimize=True)
                prepared = out.getvalue()
                prepared_mime = "image/png"
            else:
                img = img.convert("RGB")
                img.save(out, format="JPEG", quality=60, optimize=True)
                prepared = out.getvalue()
                prepared_mime = "image/jpeg"

            if max_image_bytes is not None and len(prepared) > max_image_bytes:
                if max(img.size) > 512:
                    scale = 512 / float(max(img.size))
                    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                out = io.BytesIO()
                if has_alpha:
                    img.save(out, format="PNG", optimize=True)
                else:
                    img.save(out, format="JPEG", quality=45, optimize=True)
                prepared = out.getvalue()
            return prepared, prepared_mime
    except Exception as e:
        logger.warning("Failed to preprocess image %s: %s", image_path, e)
        return data, mime


def _build_image_block(md_path: Path, target: str, max_image_bytes: int | None) -> dict | None:
    normalized = _normalize_image_target(target)
    if not normalized:
        return None
    if "://" in normalized or normalized.startswith("data:"):
        logger.warning("Skip %s because %s is not a local file", md_path, normalized)
        return None
    image_path = Path(normalized)
    if not image_path.is_absolute():
        image_path = (md_path.parent / image_path).resolve()
    if not image_path.exists():
        logger.warning("Skip %s because referenced image %s does not exist", md_path, normalized)
        return None
    if not image_path.is_file():
        logger.warning("Skip %s because referenced image %s is not a file", md_path, normalized)
        return None
    data = image_path.read_bytes()
    data, mime = _prepare_image_bytes(image_path, data, max_image_bytes)
    if max_image_bytes is not None and len(data) > max_image_bytes:
        logger.warning("Skip %s because it is larger than %d bytes", image_path, max_image_bytes)
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
        },
    }


def _interleave_markdown_and_images(
    md_path: Path,
    text: str,
    max_images: int | None,
    max_image_bytes: int | None,
) -> list[dict]:
    blocks: list[dict] = []
    cursor = 0
    images_added = 0
    for match in IMAGE_PATTERN.finditer(text):
        start, end = match.span()
        _append_text_block(blocks, text[cursor:start])
        cursor = end
        if max_images is not None and images_added >= max_images:
            _append_text_block(blocks, match.group(0))
            continue
        target = match.group("target")
        image_block = _build_image_block(md_path, target, max_image_bytes)
        if image_block is None:
            _append_text_block(blocks, match.group(0))
            continue
        blocks.append(image_block)
        images_added += 1
    _append_text_block(blocks, text[cursor:])
    if not blocks:
        blocks.append({"type": "text", "text": text})
    return blocks


def build_context_blocks(
    outputs_dir: Path,
    max_docs: int | None,
    max_chars: int | None,
    include_images: bool,
    max_images: int | None,
    max_image_bytes: int | None,
    skip_suffixes: Sequence[str],
) -> list[dict]:
    blocks: list[dict] = []
    doc_count = 0
    for md_path in iter_markdown_files(outputs_dir, skip_suffixes=skip_suffixes):
        rel = md_path.relative_to(outputs_dir)
        text = read_markdown(md_path, max_chars)
        header = f"Document: {rel}"
        doc_text = f"{header}\n{text}"
        if include_images:
            blocks.extend(
                _interleave_markdown_and_images(
                    md_path=md_path,
                    text=doc_text,
                    max_images=max_images,
                    max_image_bytes=max_image_bytes,
                )
            )
        else:
            blocks.append({"type": "text", "text": doc_text})
        doc_count += 1
        if max_docs is not None and doc_count >= max_docs:
            break
    if not blocks:
        raise FileNotFoundError(f"No markdown files found under {outputs_dir}")
    return blocks


# ---------------------------------------------------------------------------
# Process single question (for parallel execution)
# ---------------------------------------------------------------------------

def process_single_question(
    question: str,
    context_blocks: list[dict],
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    """
    Process a single question using the naive approach:
    send all documents + question to the model in one request.
    
    Returns: (question, answer)
    """
    logger.info("[Q: {}] Processing with {} context blocks", question[:50], len(context_blocks))
    
    # Import extraction principles from pipeline
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
    
    # OpenAI's newer models use max_completion_tokens instead of max_tokens
    token_param = {}
    if model.startswith(("gpt")):
        token_param = {"max_completion_tokens": max_tokens}
    else:
        token_param = {"max_tokens": max_tokens}
    
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        **token_param,
        messages=[
            {
                "role": "system",
                "content": f"You are a scientific literature analyst extracting structured data from research papers.\n\n{EXTRACTION_PRINCIPLES}\n\nAnswer strictly based on the provided scientific papers. Output as valid JSON only.",
            },
            {
                "role": "user",
                "content": context_blocks + [{"type": "text", "text": f"Question: {question}"}],
            },
        ],
    )

    message = response.choices[0].message.content
    if isinstance(message, list):
        answer = "\n".join(block.get("text", "") for block in message if block.get("type") == "text")
    else:
        answer = message or ""
    
    logger.info("[Q: {}] Completed", question[:50])
    return (question, answer)


def process_topic(
    topic_name: str,
    outputs_dir: Path,
    questions: list[str],
    client: OpenAI,
    args: argparse.Namespace,
) -> dict:
    """
    Process all questions for a single topic using the naive approach.
    
    Returns: dict with topic results
    """
    logger.info("[Topic: {}] Loading documents from {}", topic_name, outputs_dir)
    
    try:
        context_blocks = build_context_blocks(
            outputs_dir=outputs_dir,
            max_docs=args.max_docs,
            max_chars=args.max_chars,
            include_images=not args.no_images,
            max_images=args.max_images,
            max_image_bytes=args.max_image_bytes,
            skip_suffixes=args.skip_suffixes,
        )
        num_docs = sum(1 for _ in iter_markdown_files(outputs_dir, args.skip_suffixes))
        logger.info("[Topic: {}] Loaded context with {} documents", topic_name, num_docs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[Topic: {}] Failed to load documents: {}", topic_name, exc)
        return {
            "topic": topic_name,
            "outputs_dir": str(outputs_dir),
            "error": str(exc),
            "questions": [],
        }

    # Process all questions for this topic in parallel
    results: list[tuple[str, str]] = []
    worker_count = min(args.parallel_questions, len(questions)) if len(questions) > 1 else 1
    
    logger.info("[Topic: {}] Processing {} questions with concurrency={}", topic_name, len(questions), worker_count)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_question = {
            executor.submit(
                process_single_question,
                question,
                context_blocks,
                client,
                args.model,
                args.temperature,
                args.max_tokens,
            ): question
            for question in questions
        }
        for future in concurrent.futures.as_completed(future_to_question):
            question = future_to_question[future]
            try:
                result = future.result()
                results.append(result)
                logger.info("[Topic: {}] Completed question: {}", topic_name, question[:50])
            except Exception as exc:  # noqa: BLE001
                logger.exception("[Topic: {}] Failed question '{}': {}", topic_name, question[:50], exc)
                results.append((question, f"ERROR: {exc}"))
    
    # Sort results by original question order
    question_order = {q: i for i, q in enumerate(questions)}
    results.sort(key=lambda x: question_order.get(x[0], 999))
    
    return {
        "topic": topic_name,
        "outputs_dir": str(outputs_dir),
        "num_documents": num_docs,
        "questions": [
            {"question": q, "answer": a}
            for q, a in results
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Naive QA: Send all documents + question to model in one request. "
            "Supports both single-topic mode (command-line) and multi-topic mode (JSON config). "
            "Use this as a baseline comparison against cot_qa.py."
        )
    )
    parser.add_argument(
        "questions",
        type=str,
        nargs="*",
        help="One or more questions to ask (single-topic mode). Use quotes for multi-word questions.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file with multiple topics and questions. If provided, overrides other input methods.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(__file__).parent / "outputs",
        help="Directory containing MinerU exports (single-topic mode only).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        help="OpenAI-compatible model to use (default Qwen3 VL on local vLLM).",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides OPENAI_API_KEY).")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of documents per request (default: unlimited).",
    )
    parser.add_argument("--max-chars", type=int, default=40_000, help="Character cap per document block.")
    parser.add_argument("--max-tokens", type=int, default=10240, help="Max tokens for the completion.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument(
        "--parallel-questions",
        type=int,
        default=5,
        help="Number of questions to process in parallel (default: 1 for sequential)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images per document (default: unlimited).",
    )
    parser.add_argument("--max-image-bytes", type=int, default=2_000_000, help="Skip images larger than this many bytes.")
    parser.add_argument("--no-images", action="store_true", help="Exclude images from the prompt.")
    parser.add_argument(
        "--skip-suffixes",
        nargs="*",
        default=(),
        metavar="SUFFIX",
        help="Skip markdown files whose names end with these suffixes (default: include everything).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save results to a JSON file (useful for multi-topic mode)",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Save console output to a text file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(api_key=api_key, base_url=args.api_base)

    # Open output file if specified
    output_file = None
    if args.output_txt:
        output_file = open(args.output_txt, "w", encoding="utf-8")
        logger.info("Console output will be saved to: {}", args.output_txt)

    def output(text: str = "") -> None:
        """Print to console and optionally to file."""
        print(text)
        if output_file:
            output_file.write(text + "\n")
            output_file.flush()

    try:
        # Determine mode: config file (multi-topic) or command-line (single-topic)
        if args.config:
            # Multi-topic mode: load config from JSON
            logger.info("Loading configuration from: {}", args.config)
            with open(args.config) as f:
                config = json.load(f)
            
            topics = config.get("topics", [])
            if not topics:
                raise ValueError("No topics found in config file")
            
            logger.info("Processing {} topic(s) using NAIVE approach (all docs in one request)", len(topics))
            
            # Process each topic
            all_results = []
            for topic_config in topics:
                topic_name = topic_config.get("name", "Unnamed Topic")
                outputs_dir = Path(topic_config.get("outputs_dir")).resolve()
                questions = topic_config.get("questions", [])
                
                if not outputs_dir.exists():
                    logger.error("[Topic: {}] Outputs directory does not exist: {}", topic_name, outputs_dir)
                    all_results.append({
                        "topic": topic_name,
                        "outputs_dir": str(outputs_dir),
                        "error": "Directory does not exist",
                        "questions": [],
                    })
                    continue
                
                if not questions:
                    logger.warning("[Topic: {}] No questions provided", topic_name)
                    continue
                
                topic_results = process_topic(
                    topic_name=topic_name,
                    outputs_dir=outputs_dir,
                    questions=questions,
                    client=client,
                    args=args,
                )
                all_results.append(topic_results)
            
            # Output results
            if args.output_json:
                logger.info("Saving results to: {}", args.output_json)
                with open(args.output_json, "w") as f:
                    json.dump({"method": "naive", "results": all_results}, f, indent=2, ensure_ascii=False)
                logger.info("Results saved")
            
            # Print results to console
            for topic_result in all_results:
                output(f"\n{'=' * 80}")
                output(f"TOPIC: {topic_result['topic']}")
                output(f"Directory: {topic_result['outputs_dir']}")
                if "error" in topic_result:
                    output(f"ERROR: {topic_result['error']}")
                else:
                    output(f"Documents: {topic_result.get('num_documents', 'N/A')}")
                output(f"{'=' * 80}")
                
                for i, qa in enumerate(topic_result.get("questions", []), 1):
                    output(f"\nQuestion {i}: {qa['question']}")
                    output(f"{'-' * 80}")
                    output(qa['answer'])
                    output()
        
        else:
            # Single-topic mode: use command-line arguments
            if not args.questions:
                raise ValueError("No questions provided. Use positional arguments or --config for JSON mode.")
            
            outputs_dir = args.outputs.resolve()
            if not outputs_dir.exists():
                raise FileNotFoundError(f"{outputs_dir} does not exist")
            
            logger.info("Single-topic mode: Processing {} question(s) using NAIVE approach", len(args.questions))
            
            topic_result = process_topic(
                topic_name="Single Topic",
                outputs_dir=outputs_dir,
                questions=args.questions,
                client=client,
                args=args,
            )
            
            # Output results
            if args.output_json:
                logger.info("Saving results to: {}", args.output_json)
                with open(args.output_json, "w") as f:
                    json.dump({"method": "naive", "results": [topic_result]}, f, indent=2, ensure_ascii=False)
            
            # Print results
            for i, qa in enumerate(topic_result["questions"], 1):
                output(f"\n{'=' * 80}")
                output(f"Question {i}/{len(topic_result['questions'])}: {qa['question']}")
                output(f"{'=' * 80}")
                output(qa['answer'])
                output()
    
    finally:
        if output_file:
            output_file.close()
            logger.info("Console output saved to: {}", args.output_txt)


if __name__ == "__main__":
    main()
