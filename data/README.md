# Data

This directory contains all intermediate and final outputs of the processing pipeline, plus the document metadata used throughout.

## Files

### `dataset.csv`
Submission metadata for all consultation documents. Each row corresponds to one submission and includes identifiers, respondent category, and other metadata fields. This is the starting point for document selection and is used by the dashboard to display submission-level information.

### `ignore_ids.txt`
List of document IDs excluded from processing (one per line). These are submissions that were identified as non-relevant (e.g. full academic papers, annex-only files, duplicates) and filtered out before Step 2 (OCR).

---

## Directories

### `chunks/`
Paragraph-level chunk files produced by Step 4 (`scripts/03_chunk.py`). One `.chunk.json` file per document (4,324 documents). Each file contains the parsed paragraphs of the submission, together with section metadata, heuristic quality scores, and a `chunk_quality` flag (`"ok"` or `"noisy"`) that determines which paragraphs are sent to the LLM in the next step.

### `consolidated_outputs/`
Final analysis-ready files produced by the consolidation step (`scripts/05_consolidate.py`) and post-processing step (`scripts/06_postprocess.py`). Contains two subdirectories:

- **`consolidated_outputs_5-nano_1404/`** — Raw LLM extraction results. One `.paragraph.json` file per document, with each paragraph's extracted topic, topic source (`dfa` or `new`), and evidence quotes joined back to the original paragraph text.

- **`consolidated_outputs_5-nano_1404_postprocessed_topics/`** — Post-processed version of the above, with topic labels merged and deduplicated. This is the dataset used by the dashboard.

Both subdirectories contain 4,325 files. The model used for extraction was `gpt-5-nano` (run on 14 April 2026).