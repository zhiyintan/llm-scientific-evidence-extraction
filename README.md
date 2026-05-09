# Diagnosing LLMs for Structured Scientific Evidence Extraction

This repository contains the code, query suite, and evaluation framework for the paper:

**Diagnosing the Limits of Large Language Models for Structured Scientific Evidence Extraction**

---

## Overview

Scientific evidence is typically reported in unstructured articles, while many downstream workflows require structured representations of study populations, variables, methods, and quantitative findings.

This project studies how well current large language models (LLMs) can construct schema-constrained evidence records from full-text scientific documents.

Instead of proposing a new extraction system, we provide a diagnostic evaluation framework that reveals where and how model performance breaks down as structural demands increase.

---

## What is included

- **Schema-grounded query suite**
  - Object-centric and method-centric extraction tasks
  - Increasing structural complexity (from single fields to multi-field tuples)

- **Derived statistical queries**
  - Multi-stage outputs: `evidences → final_list → final_answer`
  - Enables process-level analysis of reasoning and aggregation

- **Evaluation protocol**
  - Tuple-level precision / recall / F1 (macro-averaged)
  - Three matching levels: **Strict / Medium / Lenient**
  - Paper-level attribution constraints
  - Human-validated LLM-based semantic matching

- **Gold annotations**
  - Manually curated, paper-linked structured evidence
  - Cross-domain scientific corpora

---

## Key idea

Performance on scientific extraction is not limited by entity recognition alone.

As tasks require:
- binding multiple entities,
- attaching numerical values,
- preserving method-specific context,

model performance degrades sharply.

The framework isolates these failure modes by controlling tuple arity and aggregation complexity.

---

## Usage

### Run extraction

The extraction entrypoint is `src/run_extraction.py`.

By default it looks for parsed markdown documents under `mineru_output/{domain}`. This directory is not distributed with the repository due to copyright constraints. To run on your own corpus, pass `--docs-dir`.

Accepted input layouts:

- MinerU-style: `<docs-dir>/<paper_folder>/vlm/*.md` (optionally with local images referenced from the markdown)
- Plain markdown: any directory containing `*.md` files (the loader searches recursively)

Example:

```bash
python src/run_extraction.py \
  --domain social \
  --mode per-paper \
  --docs-dir /path/to/your/parsed_docs/social
```

Notes:

- Use `--model` to specify a model id; if omitted, the runner attempts to auto-detect from the endpoint.
- Use `--base-url` to point to a local OpenAI-compatible server.
- Use `--no-images` to force text-only inputs.

### Workflow

1. Run extraction queries on a set of papers  
2. Collect structured outputs in JSON format  
3. Evaluate against gold annotations using the provided protocol  

Details of queries and output formats are provided in the repository.

---

## Repository structure

```
.
├── data/                          # Configs and annotations
│   ├── paper_map.json             # Paper ID ↔ file/path mapping
│   ├── queries/                   # Standardized query configs (prompt/schema, etc.)
│   │   └── standardized_config.json
│   └── ground_truth/              # Human-verified gold annotations + domain subsets
│       ├── 1_biodiversity/        # Domain subset (biodiversity)
│       ├── 2_engineering/         # Domain subset (engineering / hotel energy)
│       ├── 3_myopia/              # Domain subset (health / myopia)
│       ├── 4_agriculture/         # Domain subset (agriculture / water)
│       └── 5_social/              # Domain subset (social science / well-being)
├── scripts/                       # Experiment and evaluation entrypoints
│   ├── build_result.py            # Build/aggregate result files
│   └── eval/                      # Evaluation and auditing tools
│       └── batch_eval.py
└── src/                           # Core code
    ├── run_extraction.py          # Main entrypoint for extraction runs
    └── extraction/                # Extraction pipeline and QA components
        ├── pipeline.py            # End-to-end pipeline wiring and execution
        └── qa.py                  # QA / extraction-related logic
```

Note: output directories such as `mineru_output/` and `results/` may be created when running extraction/evaluation, but are not distributed with this repository due to copyright constraints.

---

## Citation

If you use this repository, please cite:
TBD


---

## Note

This repository provides a **query-driven evaluation framework**, not a large-scale benchmark dataset.  
It is intended to support **fine-grained analysis and future benchmark design** for evidence-intensive scientific tasks.
