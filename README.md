# Diagnosing LLMs for Structured Scientific Evidence Extraction

This repository contains the code, query suite, and evaluation framework for the paper:

**Diagnosing the Limits of Large Language Models for Structured Scientific Evidence Extraction**

---

## Overview

Scientific evidence is typically reported in unstructured articles, while many downstream workflows require structured representations of study populations, variables, methods, and quantitative findings.

This project studies how well current large language models (LLMs) can construct **schema-constrained evidence records** from full-text scientific documents.

Instead of proposing a new extraction system, we provide a **diagnostic evaluation framework** that reveals where and how model performance breaks down as structural demands increase.

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

The framework isolates these failure modes by controlling **tuple arity** and **aggregation complexity**.

---

## Usage

1. Run extraction queries on a set of papers  
2. Collect structured outputs in JSON format  
3. Evaluate against gold annotations using the provided protocol  

Details of queries and output formats are provided in the repository.

---

## Citation

If you use this repository, please cite:
TBD


---

## Note

This repository provides a **query-driven evaluation framework**, not a large-scale benchmark dataset.  
It is intended to support **fine-grained analysis and future benchmark design** for evidence-intensive scientific tasks.
