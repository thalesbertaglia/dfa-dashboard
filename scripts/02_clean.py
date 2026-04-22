import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger("ocr_clean")


@dataclass(frozen=True)
class StepLog:
    name: str
    changed: bool
    metrics: dict[str, Any]


FilterFn = Callable[[str], tuple[str, StepLog]]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def iter_json_files(root_dir: Path, *, pattern: str = "*.json") -> list[Path]:
    root_dir = root_dir.expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    return sorted([p for p in root_dir.rglob(pattern) if p.is_file()])


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_pages(ocr: dict[str, Any]) -> list[dict[str, Any]]:
    pages = ocr.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Unexpected OCR JSON schema: missing 'pages' list.")
    return pages


def get_page_markdown(page: dict[str, Any]) -> str:
    md = page.get("markdown")
    return md if isinstance(md, str) else ""


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text, flags=re.UNICODE))


def _remove_lines_matching(
    patterns: list[re.Pattern[str]], text: str
) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if any(p.search(line) for p in patterns):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def f_remove_image_placeholders() -> FilterFn:
    patterns = [
        re.compile(r"!\[.*?\]\(.*?\)"),
        re.compile(r"<img\b", re.IGNORECASE),
        re.compile(r"data:image/", re.IGNORECASE),
        re.compile(r"\bbase64\b", re.IGNORECASE),
    ]

    def _fn(text: str) -> tuple[str, StepLog]:
        out, removed = _remove_lines_matching(patterns, text)
        return out, StepLog(
            name="remove_image_placeholders",
            changed=(removed > 0),
            metrics={"lines_removed": removed},
        )

    return _fn


def f_remove_isolated_page_numbers() -> FilterFn:
    pat = re.compile(r"^\s*\d{1,4}\s*$")

    def _fn(text: str) -> tuple[str, StepLog]:
        kept: list[str] = []
        removed = 0
        for line in text.splitlines():
            if pat.match(line):
                removed += 1
                continue
            kept.append(line)
        out = "\n".join(kept)
        return out, StepLog(
            name="remove_isolated_page_numbers",
            changed=(removed > 0),
            metrics={"lines_removed": removed},
        )

    return _fn


def f_remove_markdown_footnote_definitions() -> FilterFn:
    pat = re.compile(r"^\s*\[\^\w+\]\s*:\s+")

    def _fn(text: str) -> tuple[str, StepLog]:
        kept: list[str] = []
        removed = 0
        for line in text.splitlines():
            if pat.match(line):
                removed += 1
                continue
            kept.append(line)
        out = "\n".join(kept)
        return out, StepLog(
            name="remove_markdown_footnote_definitions",
            changed=(removed > 0),
            metrics={"lines_removed": removed},
        )

    return _fn


def f_remove_unicode_footnote_blocks() -> FilterFn:
    pat = re.compile(r"^\s*[¹²³⁴⁵⁶⁷⁸⁹⁰]+(\s+|$)")

    def _fn(text: str) -> tuple[str, StepLog]:
        kept: list[str] = []
        removed = 0
        for line in text.splitlines():
            if pat.match(line):
                removed += 1
                continue
            kept.append(line)
        out = "\n".join(kept)
        return out, StepLog(
            name="remove_unicode_footnote_blocks",
            changed=(removed > 0),
            metrics={"lines_removed": removed},
        )

    return _fn


def f_strip_inline_unicode_footnote_markers() -> FilterFn:
    pat = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ0-9\)\]\.,;:])([¹²³⁴⁵⁶⁷⁸⁹⁰]+)\b")

    def _fn(text: str) -> tuple[str, StepLog]:
        out, n = pat.subn(r"\1", text)
        return out, StepLog(
            name="strip_inline_unicode_footnote_markers",
            changed=(n > 0),
            metrics={"replacements": n},
        )

    return _fn


def f_normalise_whitespace() -> FilterFn:
    def _fn(text: str) -> tuple[str, StepLog]:
        before_len = len(text)
        out = re.sub(r"[ \t]+", " ", text)
        out = re.sub(r"\n{3,}", "\n\n", out)
        out = out.strip() + "\n"
        return out, StepLog(
            name="normalise_whitespace",
            changed=(len(out) != before_len),
            metrics={"before_len": before_len, "after_len": len(out)},
        )

    return _fn


def build_pipeline() -> list[FilterFn]:
    return [
        f_remove_image_placeholders(),
        f_remove_markdown_footnote_definitions(),
        f_remove_unicode_footnote_blocks(),
        f_strip_inline_unicode_footnote_markers(),
        f_remove_isolated_page_numbers(),
        f_normalise_whitespace(),
    ]


def relpath_under(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def bucket_for_document(pages_total: int, pages_kept: int, threshold: int) -> str:
    dropped = pages_total - pages_kept
    return "reliable" if dropped <= threshold else "flag"


def process_one_json(
    json_path: Path,
    *,
    input_root: Path,
    out_md_root: Path,
    out_log_root: Path,
    min_words_per_page: int,
    pages_drop_threshold: int,
) -> dict[str, Any]:
    ocr = load_json(json_path)
    pages = extract_pages(ocr)
    pages_total = len(pages)

    pipeline = build_pipeline()

    kept_pages_md: list[str] = []
    page_logs: list[dict[str, Any]] = []

    for page_index, page in enumerate(pages):
        raw = get_page_markdown(page)
        text = raw
        steps: list[StepLog] = []

        for fn in pipeline:
            text, log = fn(text)
            steps.append(log)

        wc = word_count(text)
        keep = wc >= min_words_per_page

        original_page_num = page_index + 1
        if keep:
            kept_pages_md.append(
                f"<!-- PAGE {original_page_num} of {pages_total} -->\n\n{text.strip()}\n"
            )

        page_logs.append(
            {
                "source_json": str(json_path),
                "page_index": page_index,
                "original_page_num": original_page_num,
                "pages_total": pages_total,
                "keep": keep,
                "min_words_per_page": min_words_per_page,
                "word_count": wc,
                "raw_len": len(raw),
                "clean_len": len(text),
                "steps": [log.__dict__ for log in steps],
            }
        )

    pages_kept = sum(1 for r in page_logs if r["keep"])
    pages_dropped = pages_total - pages_kept
    bucket = bucket_for_document(pages_total, pages_kept, pages_drop_threshold)

    rel_stem = relpath_under(json_path, input_root).with_suffix("")
    out_md_path = (out_md_root / bucket / rel_stem).with_suffix(".clean.md")
    out_log_path = (out_log_root / bucket / rel_stem).with_suffix(".jsonl")

    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_log_path.parent.mkdir(parents=True, exist_ok=True)

    out_md_path.write_text("\n\n".join(kept_pages_md).strip() + "\n", encoding="utf-8")
    with out_log_path.open("wt", encoding="utf-8") as f:
        for rec in page_logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "json": str(json_path),
        "bucket": bucket,
        "clean_md": str(out_md_path),
        "page_log": str(out_log_path),
        "pages_total": pages_total,
        "pages_kept": pages_kept,
        "pages_dropped": pages_dropped,
        "pages_drop_threshold": pages_drop_threshold,
        "min_words_per_page": min_words_per_page,
    }


def process_directory(
    root_dir: str | Path,
    *,
    input_pattern: str = "F*.json",
    min_words_per_page: int = 30,
    pages_drop_threshold: int = 2,
) -> Path:
    setup_logging()
    input_root = Path(root_dir).expanduser().resolve()

    out_md_root = input_root / "_clean_md"
    out_log_root = input_root / "_clean_logs"
    out_md_root.mkdir(parents=True, exist_ok=True)
    out_log_root.mkdir(parents=True, exist_ok=True)

    json_files = iter_json_files(input_root, pattern=input_pattern)
    LOGGER.info(
        "Found %s OCR JSON file(s) under %s (pattern=%s)",
        len(json_files),
        input_root,
        input_pattern,
    )
    LOGGER.info(
        "Settings: min_words_per_page=%s pages_drop_threshold=%s",
        min_words_per_page,
        pages_drop_threshold,
    )

    summaries: list[dict[str, Any]] = []
    for i, json_path in enumerate(json_files, start=1):
        LOGGER.info("[%s/%s] Clean: %s", i, len(json_files), json_path)
        try:
            summaries.append(
                process_one_json(
                    json_path,
                    input_root=input_root,
                    out_md_root=out_md_root,
                    out_log_root=out_log_root,
                    min_words_per_page=min_words_per_page,
                    pages_drop_threshold=pages_drop_threshold,
                )
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.exception("Failed: %s", json_path)
            summaries.append({"json": str(json_path), "ok": False, "error": str(e)})

    bucket_counts = {
        "reliable": sum(1 for s in summaries if s.get("bucket") == "reliable"),
        "flag": sum(1 for s in summaries if s.get("bucket") == "flag"),
        "errors": sum(1 for s in summaries if s.get("ok") is False),
    }

    report = {
        "input_root": str(input_root),
        "input_pattern": input_pattern,
        "min_words_per_page": min_words_per_page,
        "pages_drop_threshold": pages_drop_threshold,
        "counts": bucket_counts,
        "items": summaries,
    }

    report_path = out_log_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Wrote report: %s", report_path)
    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    parser.add_argument("--pattern", default="F*.json")
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--drop-threshold", type=int, default=2)
    args = parser.parse_args()

    process_directory(
        args.root_dir,
        input_pattern=args.pattern,
        min_words_per_page=args.min_words,
        pages_drop_threshold=args.drop_threshold,
    )
