# End-to-End Processing Pipeline

This document describes how raw PDFs are transformed into structured LLM outputs (topics, evidence quotes) suitable for the dashboard and analysis.

The pipeline is:

```
PDFs → OCR → cleaned text → chunks → LLM extraction → consolidation → post-processing
```

Each step produces a reviewable intermediate output. No step overwrites the previous one.

---

## Step 1 — Raw PDFs (input)

We start from the PDF files submitted to the consultation. These are treated as the ground-truth source documents.

All downstream analysis is constrained by what is present (and legible) in these PDFs. Errors here cannot be fixed later.

- Filter out non-submission PDFs before processing (see `data/ignore_ids.txt`).

---

## Step 2 — OCR: PDF → OCR JSON + per-page Markdown
**Script:** `scripts/01_ocr.py`

Each PDF is processed via the Mistral OCR API. For every input PDF we produce:
- an **OCR JSON** file (full API response),
- a **Markdown file** with per-page content and page markers.

### Outputs
- `<path>.json` — OCR response
- `<path>.md` — concatenated per-page markdown
- `process.log` — per-file logs
- `report.json` — summary of OCR results

---

## Step 3 — Page-level cleaning & filtering: OCR JSON → cleaned Markdown
**Script:** `scripts/02_clean.py`

OCR output is cleaned page by page, directly from the OCR JSON. This step:
- removes common OCR noise (images, footnotes, page numbers),
- normalises whitespace,
- drops pages below a minimum word count (default: 30).

Documents are bucketed as **reliable** (few pages dropped) or **flag** (many pages dropped).

### Outputs
- `_clean_md/reliable/*.clean.md`
- `_clean_md/flag/*.clean.md`
- Page-level transformation logs (`.jsonl`)
- Summary report: `_clean_logs/report.json`

---

## Step 4 — Chunking: cleaned Markdown → paragraph JSON
**Script:** `scripts/03_chunk.py`

Cleaned Markdown is converted into paragraph-level units suitable for LLM processing. This step:
- parses headings, paragraphs, and list items,
- assigns section context,
- computes heuristic signals (length, character ratios),
- selects paragraphs using a score threshold plus a top-k fallback,
- labels paragraphs as `chunk_quality = "ok"` or `"noisy"`.

### Outputs
- `<doc_id>.chunk.json` — paragraph text and metadata, section structure, selection flags, configuration and summary statistics.

---

## Step 5 — LLM extraction: paragraph JSON → structured outputs
**Script:** `scripts/04_extract.py`

Each paragraph with `chunk_quality="ok"` is processed independently by the LLM to extract:
- topic,
- topic source (`dfa` — from the predefined registry, or `new`),
- verbatim evidence quotes.

Paragraphs marked as noisy are skipped but still recorded.

### Outputs
- `<doc_id>.paragraph.json` — run metadata, per-paragraph extraction records, validation status, errors, latency, repair flags.

### Prompts
- `prompts/base_prompt_v2.txt` — main extraction prompt with DFA topic registry
- `prompts/dfa_context_v1.txt` — DFA topic registry context block
- `prompts/only_topic_no_registry_v3.txt` — ablation variant without registry

---

## Step 6 — Consolidation: LLM outputs + chunks → analysis-ready JSON
**Script:** `scripts/05_consolidate.py`

LLM outputs are re-joined with their original paragraph text and metadata. This step:
- finds the matching chunk file for each extraction file,
- re-attaches paragraph text and chunk metadata,
- writes a consolidated JSON file per document.

### Outputs
- One consolidated JSON per document.
- `_consolidation_failures.txt` if doc ID mismatches occur.

---

## Step 7 — Post-processing: topic merging and clustering
**Script:** `scripts/06_postprocess.py`

Topics extracted across documents are clustered and merged to resolve near-duplicates and overly specific labels. See the script for configuration.

---

## Utilities

- `scripts/utils/fix_topic_source.py` — repair `topic_source` tags (`dfa` vs `new`) if needed after extraction.
