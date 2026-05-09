#!/usr/bin/env python3
"""
Run LLM extraction on a document corpus.

Usage:
    python src/run_extraction.py --domain social --mode global --model gpt-5.2
    python src/run_extraction.py --domain agriculture --mode per-paper --model qwen3-vl
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.pipeline import (
    load_questions,
    build_doc_contexts,
    run_global_extraction,
    run_map_reduce_extraction,
    detect_model,
    init_audit_dirs,
)
from openai import OpenAI
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM extraction on document corpus"
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["agriculture", "health", "social", "hotel", "biodiversity", "water", "myopia"],
        help="Domain to process",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["global", "per-paper"],
        help="Extraction mode: global (all docs at once) or per-paper (map-reduce)",
    )
    parser.add_argument(
        "--model", default=None, help="Model to use (auto-detected from vLLM server if not specified)"
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Directory containing markdown documents (auto-detected from domain if not provided)",
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Output directory for results"
    )
    parser.add_argument(
        "--config",
        default="data/queries/standardized_config.json",
        help="Path to query configuration file",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url", default=None, help="Custom API base URL (for local models)"
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds for OpenAI-compatible calls (default: 180)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for LLM generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16384, help="Max tokens for LLM response"
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to ground truth JSON for paper_id filtering/validation",
    )
    parser.add_argument(
        "--paper-map",
        default=None,
        help="Path to paper_map.json (maps paper_id -> folder_name per domain)",
    )
    parser.add_argument(
        "--paper-map-domain",
        default=None,
        help="Domain key in paper_map.json (e.g. 'hotel', 'biodiversity'). Defaults to --domain.",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Comma-separated question prefixes to run (e.g. 'O1,M1,O2,M2'). All if omitted.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image input (text-only). Use for models with limited multimodal context.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Max images per paper (default: 5). Set to 0 for text-only.",
    )
    parser.add_argument(
        "--save-raw-outputs",
        action="store_true",
        help="Save raw LLM outputs under <output-dir>/<run>/raw_llm_outputs/<run_id>/",
    )
    parser.add_argument(
        "--parallel-docs",
        type=int,
        default=10,
        help="Max concurrent papers in per-paper mode (default: 10)",
    )
    parser.add_argument(
        "--parallel-questions",
        type=int,
        default=10,
        help="Max concurrent question aggregation/decomposition workers in per-paper mode (default: 10)",
    )
    parser.add_argument(
        "--global-parallel",
        type=int,
        default=3,
        help="Max concurrent questions in global mode (default: 3)",
    )

    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)

    # Setup paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    # Auto-detect docs directory if not provided
    if args.docs_dir:
        docs_dir = Path(args.docs_dir)
    else:
        # Assume mineru_output structure
        docs_dir = Path(f"mineru_output/{args.domain}")
        if not docs_dir.exists():
            print(f"ERROR: Documents directory not found: {docs_dir}")
            print("Please provide --docs-dir or ensure mineru_output/{domain} exists")
            sys.exit(1)

    # Load GT paper IDs if provided
    gt_paper_ids = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)

            def collect_paper_ids(value):
                paper_ids = set()
                if isinstance(value, dict):
                    if "paper_id" in value and isinstance(value["paper_id"], int):
                        paper_ids.add(value["paper_id"])
                    for nested in value.values():
                        paper_ids.update(collect_paper_ids(nested))
                elif isinstance(value, list):
                    for item in value:
                        paper_ids.update(collect_paper_ids(item))
                return paper_ids

            gt_paper_ids = sorted(collect_paper_ids(gt_data))
            print(f"Loaded GT from {gt_path} with {len(gt_paper_ids)} paper IDs")
        else:
            print(f"WARNING: Ground truth file not found: {gt_path}")

    # Setup client
    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client_kwargs["timeout"] = args.request_timeout
    client = OpenAI(**client_kwargs)

    # Auto-detect model if not specified
    model = args.model
    if not model:
        model = detect_model(client)
        if not model:
            print("ERROR: Could not auto-detect model. Please specify --model")
            sys.exit(1)
    print(f"Using model: {model}")

    # Setup output (after model is known)
    model_safe = model.replace("-", "_").replace("/", "_")
    output_dir = (
        Path(args.output_dir)
        / f"{args.domain}_{model_safe}_{args.mode.replace('-', '_')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dirs = init_audit_dirs(output_dir / "raw_llm_outputs") if args.save_raw_outputs else None

    # Load config
    questions = load_questions(config_path)
    if args.questions:
        prefixes = tuple(p.strip() for p in args.questions.split(","))
        questions = [q for q in questions if q["id"].startswith(prefixes)]
    print(f"Loaded {len(questions)} questions from {config_path}")

    # Load paper_map if provided
    paper_map = None
    if args.paper_map:
        pm_path = Path(args.paper_map)
        if pm_path.exists():
            with open(pm_path, 'r') as f:
                all_maps = json.load(f)
            domain_key = args.paper_map_domain or args.domain
            if domain_key in all_maps:
                # Convert string keys to int keys
                paper_map = {int(k): v for k, v in all_maps[domain_key].items()}
                print(f"Loaded paper_map for '{domain_key}': {len(paper_map)} papers")
            else:
                print(f"WARNING: domain '{domain_key}' not found in paper_map. Available: {list(all_maps.keys())}")
        else:
            print(f"WARNING: paper_map file not found: {pm_path}")

    if gt_paper_ids is not None:
        if paper_map is None:
            print("ERROR: --ground-truth now requires --paper-map so paper_id values can be mapped to folders")
            sys.exit(1)

        missing_ids = [paper_id for paper_id in gt_paper_ids if paper_id not in paper_map]
        if missing_ids:
            print(f"ERROR: Missing paper_id values in paper_map: {missing_ids}")
            sys.exit(1)

        paper_map = {paper_id: paper_map[paper_id] for paper_id in gt_paper_ids}
        print(f"Filtered paper_map to {len(paper_map)} GT paper IDs")

    # Build document contexts ordered by paper_id
    print(f"Loading documents from {docs_dir}...")
    docs = build_doc_contexts(docs_dir, paper_map=paper_map,
                              include_images=not args.no_images)
    print(f"Loaded {len(docs)} documents")

    # Run extraction
    print(f"\nRunning {args.mode} extraction with {model}...")

    if args.mode == "global":
        results = run_global_extraction(
            client=client,
            model=model,
            docs=docs,
            questions=questions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            parallel=args.global_parallel,
            audit_dirs=audit_dirs,
        )
    else:  # per-paper
        results = run_map_reduce_extraction(
            client=client,
            model=model,
            docs=docs,
            questions=questions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            parallel_docs=args.parallel_docs,
            parallel_questions=args.parallel_questions,
            audit_dirs=audit_dirs,
        )

    # Save results
    # Extract and save per-paper intermediate results if available
    per_paper_data = results.pop("__per_paper_results__", None)
    decomposed_questions = results.pop("__decomposed_questions__", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "predictions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if per_paper_data:
        pp_file = output_dir / "per_paper_results.json"
        with open(pp_file, "w", encoding="utf-8") as f:
            json.dump(per_paper_data, f, indent=2, ensure_ascii=False)
        print(f"\nPer-paper results saved to {pp_file}")

    if decomposed_questions:
        dq_file = output_dir / "decomposed_questions.json"
        with open(dq_file, "w", encoding="utf-8") as f:
            json.dump(decomposed_questions, f, indent=2, ensure_ascii=False)
        print(f"Decomposed questions saved to {dq_file}")

    print(f"\nResults saved to {output_file}")
    if audit_dirs:
        print(f"Raw LLM outputs saved to {audit_dirs['run']}")


if __name__ == "__main__":
    main()
