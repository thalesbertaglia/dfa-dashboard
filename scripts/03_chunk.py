import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from markdown_it import MarkdownIt

DEFAULT_MIN_WORDS_SOFT = 12
DEFAULT_MAX_WORDS_SOFT = 400
DEFAULT_SELECT_THRESHOLD = 0.50
DEFAULT_TOP_K = 60

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_NON_SPACE_RE = re.compile(r"\S")
_WS_RE = re.compile(r"\s+")
_HEADING_TAG_RE = re.compile(r"^h([1-6])$")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalise_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def alpha_ratio(text: str) -> float:
    chars = _NON_SPACE_RE.findall(text or "")
    if not chars:
        return 0.0
    alpha = sum(1 for ch in chars if ch.isalpha())
    return alpha / len(chars)


def digit_ratio(text: str) -> float:
    chars = _NON_SPACE_RE.findall(text or "")
    if not chars:
        return 0.0
    digits = sum(1 for ch in chars if ch.isdigit())
    return digits / len(chars)


def non_space_ratio(text: str) -> float:
    s = text or ""
    if not s:
        return 0.0
    return len(_NON_SPACE_RE.findall(s)) / len(s)


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def collect_inline_in_container(
    tokens: list[Any], close_type: str, start_idx: int
) -> tuple[str, int]:
    parts: list[str] = []
    depth = 1
    j = start_idx
    n = len(tokens)

    while j < n:
        t = tokens[j]

        if t.type.endswith("_open"):
            depth += 1
        elif t.type.endswith("_close"):
            depth -= 1
            if t.type == close_type and depth == 0:
                return " ".join(parts).strip(), j

        if t.type == "inline":
            content = (getattr(t, "content", "") or "").strip()
            if content:
                parts.append(content)

        j += 1

    return " ".join(parts).strip(), n - 1


def update_heading_stack(
    stack_by_level: dict[int, str], heading_level: int, title: str
) -> None:
    for lvl in sorted(list(stack_by_level.keys()), reverse=True):
        if lvl >= heading_level:
            del stack_by_level[lvl]
    stack_by_level[heading_level] = title


def current_section_path(stack_by_level: dict[int, str]) -> list[str]:
    if not stack_by_level:
        return []
    return [stack_by_level[lvl] for lvl in sorted(stack_by_level.keys())]


def ensure_section_id(
    sections: list[dict[str, Any]],
    section_ids: dict[tuple[str, ...], str],
    sec_path: list[str],
) -> str:
    key = tuple(sec_path)
    existing = section_ids.get(key)
    if existing is not None:
        return existing

    section_id = f"s{len(sections)}"
    level = 0 if not sec_path else len(sec_path)
    title = "__root__" if not sec_path else sec_path[-1]

    sections.append(
        {
            "section_id": section_id,
            "level": level,
            "title": title,
            "path": list(sec_path),
            "ordinal": len(sections),
        }
    )
    section_ids[key] = section_id
    return section_id


@dataclass(frozen=True)
class MeaningConfig:
    min_words_soft: int
    max_words_soft: int
    low_alpha_threshold: float
    high_digit_threshold: float


def meaning_flags(wc: int, ar: float, dr: float, cfg: MeaningConfig) -> list[str]:
    flags: list[str] = []

    if wc == 0:
        return ["empty"]

    if wc < cfg.min_words_soft:
        flags.append("very_short")

    if ar < cfg.low_alpha_threshold:
        flags.append("low_alpha")

    if dr > cfg.high_digit_threshold:
        flags.append("high_digit")

    if ("low_alpha" in flags) or ("high_digit" in flags and ar < 0.70):
        flags.append("likely_noise")

    if wc > cfg.max_words_soft:
        flags.append("very_long")

    return flags


def meaning_score(wc: int, ar: float, dr: float, cfg: MeaningConfig) -> float:
    if wc == 0:
        return 0.0
    if ar < 0.35:
        return 0.0

    length_component = clamp(
        (wc - cfg.min_words_soft) / (cfg.min_words_soft * 3), 0.0, 1.0
    )
    alpha_component = clamp((ar - 0.40) / 0.60, 0.0, 1.0)
    digit_penalty = clamp((dr - 0.20) / 0.50, 0.0, 1.0)

    score = 0.55 * length_component + 0.45 * alpha_component - 0.35 * digit_penalty
    return clamp(score, 0.0, 1.0)


@dataclass(frozen=True)
class SelectionConfig:
    threshold: float
    top_k: int | None


def apply_selection(paragraphs: list[dict[str, Any]], cfg: SelectionConfig) -> None:
    selected_by_threshold: set[str] = set()

    for p in paragraphs:
        score = float(p["meaning"]["score"])
        selected = score >= cfg.threshold
        p["selection"] = {"selected": bool(selected), "policy": "threshold"}
        if selected:
            selected_by_threshold.add(p["para_id"])

    if cfg.top_k is None:
        return

    ranked = sorted(
        paragraphs,
        key=lambda x: (float(x["meaning"]["score"]), int(x["chunk_word_count"])),
        reverse=True,
    )

    chosen_ids: set[str] = set()
    for p in ranked:
        if float(p["meaning"]["score"]) <= 0.0:
            continue
        chosen_ids.add(p["para_id"])
        if len(chosen_ids) >= cfg.top_k:
            break

    for p in paragraphs:
        pid = p["para_id"]
        was = bool(p["selection"]["selected"])
        now = was or (pid in chosen_ids)
        p["selection"]["selected"] = now

        if not now:
            continue

        if pid in selected_by_threshold and pid in chosen_ids:
            p["selection"]["policy"] = "threshold+top_k"
        elif pid in chosen_ids:
            p["selection"]["policy"] = "top_k"
        else:
            p["selection"]["policy"] = "threshold"


def apply_chunk_quality_for_run_llm(paragraphs: list[dict[str, Any]]) -> None:
    for p in paragraphs:
        selected = bool(p.get("selection", {}).get("selected"))
        p["chunk_quality"] = "ok" if selected else "noisy"


def parse_markdown_to_paragraphs(
    md_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    md = MarkdownIt("commonmark")
    tokens = md.parse(md_text or "")

    stack_by_level: dict[int, str] = {}

    sections: list[dict[str, Any]] = []
    section_ids: dict[tuple[str, ...], str] = {}
    ensure_section_id(sections, section_ids, [])

    paragraphs: list[dict[str, Any]] = []

    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]

        if t.type == "heading_open":
            m = _HEADING_TAG_RE.match(t.tag or "")
            heading_level = int(m.group(1)) if m else 0

            title = ""
            if i + 1 < n and tokens[i + 1].type == "inline":
                title = normalise_text(tokens[i + 1].content or "")

            if heading_level and title:
                update_heading_stack(stack_by_level, heading_level, title)
                sec_path = current_section_path(stack_by_level)
                ensure_section_id(sections, section_ids, sec_path)

            i += 1
            continue

        if t.type == "paragraph_open":
            raw, end_idx = collect_inline_in_container(tokens, "paragraph_close", i + 1)
            text = normalise_text(raw)
            if text:
                sec_path = current_section_path(stack_by_level)
                section_id = ensure_section_id(sections, section_ids, sec_path)
                para_id = f"p{len(paragraphs)}"
                paragraphs.append(
                    {
                        "para_id": para_id,
                        "section_id": section_id,
                        "section_path": list(sec_path),
                        "ordinal": len(paragraphs),
                        "text": text,
                    }
                )
            i = end_idx + 1
            continue

        if t.type == "list_item_open":
            raw, end_idx = collect_inline_in_container(tokens, "list_item_close", i + 1)
            text = normalise_text(raw)
            if text:
                sec_path = current_section_path(stack_by_level)
                section_id = ensure_section_id(sections, section_ids, sec_path)
                para_id = f"p{len(paragraphs)}"
                paragraphs.append(
                    {
                        "para_id": para_id,
                        "section_id": section_id,
                        "section_path": list(sec_path),
                        "ordinal": len(paragraphs),
                        "text": text,
                    }
                )
            i = end_idx + 1
            continue

        i += 1

    return sections, paragraphs


def enrich_paragraphs(paragraphs: list[dict[str, Any]], cfg: MeaningConfig) -> None:
    for p in paragraphs:
        text = p["text"]
        wc = word_count(text)
        ar = alpha_ratio(text)
        dr = digit_ratio(text)
        nsr = non_space_ratio(text)

        flags = meaning_flags(wc, ar, dr, cfg)
        score = meaning_score(wc, ar, dr, cfg)

        p["char_len"] = len(text)
        p["chunk_word_count"] = wc
        p["ratios"] = {"alpha": ar, "digit": dr, "non_space": nsr}
        p["meaning"] = {"score": score, "flags": flags}


def build_payload(
    doc_id: str,
    doc_feedback: str | None,
    source_path: str,
    sections: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    meaning_cfg: MeaningConfig,
    selection_cfg: SelectionConfig,
) -> dict[str, Any]:
    total = len(paragraphs)
    selected = sum(1 for p in paragraphs if p.get("selection", {}).get("selected"))

    by_flag: dict[str, int] = {}
    for p in paragraphs:
        for f in p.get("meaning", {}).get("flags", []):
            by_flag[f] = by_flag.get(f, 0) + 1

    return {
        "created_at": now_utc_iso(),
        "doc_id": doc_id,
        "source_path": source_path,
        "config": {
            "min_words_soft": meaning_cfg.min_words_soft,
            "max_words_soft": meaning_cfg.max_words_soft,
            "low_alpha_threshold": meaning_cfg.low_alpha_threshold,
            "high_digit_threshold": meaning_cfg.high_digit_threshold,
            "select_threshold": selection_cfg.threshold,
            "select_top_k": selection_cfg.top_k,
        },
        "feedback": [
            {"para_id": "feedback", "text": doc_feedback, "chunk_quality": "ok"}
        ],
        "sections": sections,
        "paragraphs": paragraphs,
        "stats": {
            "sections": len(sections),
            "paragraphs": total,
            "selected_paragraphs": selected,
            "selected_ratio": (selected / total) if total else 0.0,
            "counts_by_flag": by_flag,
        },
    }


def convert_file(
    md_path: Path,
    out_path: Path,
    doc_id: str,
    doc_feedback: str | None,
    meaning_cfg: MeaningConfig,
    selection_cfg: SelectionConfig,
) -> None:
    md_text = md_path.read_text(encoding="utf-8", errors="replace")
    sections, paragraphs = parse_markdown_to_paragraphs(md_text)
    enrich_paragraphs(paragraphs, meaning_cfg)
    apply_selection(paragraphs, selection_cfg)
    apply_chunk_quality_for_run_llm(paragraphs)

    payload = build_payload(
        doc_id=doc_id,
        doc_feedback=doc_feedback,
        source_path=str(md_path),
        sections=sections,
        paragraphs=paragraphs,
        meaning_cfg=meaning_cfg,
        selection_cfg=selection_cfg,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def convert_file_no_attachment(
    out_path: Path,
    doc_id: str,
    doc_feedback: str | None,
) -> None:
    payload = {
        "created_at": now_utc_iso(),
        "doc_id": doc_id,
        "feedback": [
            {"para_id": "feedback", "text": doc_feedback, "chunk_quality": "ok"}
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert markdown into paragraph JSON compatible with run_llm.py (chunk_quality + chunk_word_count)."
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        default="../data/dataset_with_feedback.csv",
        help="Path to the dataset CSV (for feedback text integration).",
    )
    p.add_argument(
        "--ocr-path",
        type=Path,
        help="Path to directory containing .md files.",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True, help="Output directory for .json files."
    )
    p.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Optional doc_id override (single-file only).",
    )

    p.add_argument("--min-words-soft", type=int, default=DEFAULT_MIN_WORDS_SOFT)
    p.add_argument("--max-words-soft", type=int, default=DEFAULT_MAX_WORDS_SOFT)
    p.add_argument("--low-alpha-threshold", type=float, default=0.55)
    p.add_argument("--high-digit-threshold", type=float, default=0.35)

    p.add_argument("--select-threshold", type=float, default=DEFAULT_SELECT_THRESHOLD)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--no-top-k",
        action="store_true",
        help="Disable top-k selection (threshold-only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset: pd.DataFrame = pd.read_csv(args.dataset)
    inp: Path = args.ocr_path
    out_dir: Path = args.out_dir

    meaning_cfg = MeaningConfig(
        min_words_soft=int(args.min_words_soft),
        max_words_soft=int(args.max_words_soft),
        low_alpha_threshold=float(args.low_alpha_threshold),
        high_digit_threshold=float(args.high_digit_threshold),
    )

    top_k = None if args.no_top_k else int(args.top_k)
    selection_cfg = SelectionConfig(
        threshold=float(args.select_threshold),
        top_k=top_k,
    )

    for _, doc in dataset.iterrows():
        doc_id = doc["feedback_reference"]
        doc_feedback = doc["feedback"]
        md_path = inp / f"{doc_id}.md"
        out_path = out_dir / f"{doc_id}.clean.chunk.json"

        if not md_path.is_file():
            convert_file_no_attachment(
                out_path=out_path, doc_id=doc_id, doc_feedback=doc_feedback
            )

    for md_path in sorted(inp.rglob("*.md")):
        doc_id = md_path.stem
        out_path = out_dir / f"{doc_id}.chunk.json"
        convert_file(
            md_path=md_path,
            out_path=out_path,
            doc_id=doc_id,
            doc_feedback=doc_feedback,
            meaning_cfg=meaning_cfg,
            selection_cfg=selection_cfg,
        )


if __name__ == "__main__":
    main()
