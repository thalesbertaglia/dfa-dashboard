# DFA Consultation Analysis

Code and data for the [DFA Dashboard](https://dfa-dashboard.thalesbertaglia.com).

This repository contains the full end-to-end pipeline for processing EU Digital Fairness Act (DFA) public consultation responses, from raw PDFs to structured topic extractions, together with an interactive Streamlit dashboard for exploring the results.

## Pipeline overview

```
PDFs → OCR → cleaned text → chunks → LLM extraction → consolidation → post-processing
```

See [docs/pipeline.md](docs/pipeline.md) for a detailed description of each step.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Copy and fill in API keys
cp .env.example .env
```

## Running the pipeline

Each script in `scripts/` corresponds to one pipeline step. Run them in order:

```bash
uv run python scripts/01_ocr.py        --input-dir data/pdf_data --output-dir data/ocr
uv run python scripts/02_clean.py      --input-dir data/ocr      --output-dir data/clean
uv run python scripts/03_chunk.py      --input-dir data/clean    --output-dir data/chunks
uv run python scripts/04_extract.py    --input-dir data/chunks   --output-dir data/extractions
uv run python scripts/05_consolidate.py --chunks-dir data/chunks --extractions-dir data/extractions --output-dir data/consolidated_outputs
uv run python scripts/06_postprocess.py --input-dir data/consolidated_outputs
```

Run any script with `--help` for the full list of options.

## Running the dashboard

```bash
uv run streamlit run app/app.py
```

By default the dashboard looks for consolidated outputs in `data/consolidated_outputs/`. Override with environment variables (see `.env.example`).

## Repository structure

```
scripts/
  01_ocr.py              # PDF → OCR JSON + Markdown (Mistral OCR API)
  02_clean.py            # OCR JSON → cleaned Markdown
  03_chunk.py            # Markdown → paragraph-level JSON
  04_extract.py          # Paragraphs → LLM topic + quote extraction
  05_consolidate.py      # LLM outputs + chunks → analysis-ready JSON
  06_postprocess.py      # Topic merging and clustering
  utils/
    fix_topic_source.py  # Repair topic_source tags post-extraction

prompts/
  only_topic_no_registry_v3.txt  # Prompt used in the deployed version of the dashboard

app/
  app.py                 # Streamlit dashboard

data/
  dataset.csv                                              # Submission metadata
  ignore_ids.txt                                           # Document IDs excluded from processing
  chunks/                                                  # Paragraph chunk files (Step 4 output, 4,324 docs)
  consolidated_outputs/
    consolidated_outputs_5-nano_1404/                      # Raw LLM extractions (Step 6 output)
    consolidated_outputs_5-nano_1404_postprocessed_topics/ # Post-processed topics (Step 7 output, used by dashboard)

docs/
  pipeline.md            # Detailed pipeline documentation
```

## Citation

```bibtex
@inproceedings{TODO}
```
