import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import litellm
from litellm import acompletion

litellm._turn_on_debug()
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("extract_topics")

KNOWN_ID_SUFFIXES = (
    ".chunk.json",
    ".paragraph.json",
    ".json",
)


class ExtractedItem(BaseModel):
    topic: str = Field(..., min_length=1)
    topic_source: Literal["dfa", "new"] = "new"
    evidence_quotes: list[str] = Field(default_factory=list, min_length=1)


class ExtractionOutput(BaseModel):
    items: list[ExtractedItem] = Field(default_factory=list)


@dataclass(slots=True)
class Config:
    model: str
    temperature: float
    max_completion_tokens: int
    timeout_s: int
    prompt_template: str
    dfa_context: str
    resume: bool
    topics_registry_path: Path | None
    use_topics_registry: bool
    save_raw_response_on_success: bool
    workers: int


@dataclass(slots=True)
class Attempt:
    output: ExtractionOutput | None
    raw_response: str | None
    latency_ms: int
    errors: list[str]


def configure_logging() -> None:
    log_dir = Path(".extract_topics_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"extract_topics_{time.strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logger.propagate = True

    for name in ("litellm", "LiteLLM"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logger.info("Logging to %s", log_path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def read_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8")


def parse_ignore_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def read_ignore_file(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()

    out: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def strip_known_suffixes(name: str) -> str:
    for suffix in KNOWN_ID_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def doc_id_from_path(path: Path) -> str:
    return strip_known_suffixes(path.name)


def output_has_errors(output_path: Path) -> bool:
    """Return True if the output file contains at least one failed (non-skipped, non-ok) record."""
    if not output_path.exists():
        return False
    try:
        data = load_json(output_path)
    except Exception:
        return False
    for rec in data.get("extractions", {}).get("paragraphs", []) or []:
        meta = rec.get("meta", {})
        if (
            not meta.get("skipped")
            and not meta.get("validation_ok")
            and meta.get("errors")
        ):
            return True
    return False


def iter_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )


def iter_units(doc: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []

    for item in doc.get("paragraphs", []) or []:
        units.append({"unit_kind": "paragraph", **item})

    for item in doc.get("feedback", []) or []:
        units.append({"unit_kind": "feedback", **item})

    return units


def should_skip_unit(unit: dict[str, Any]) -> tuple[bool, str | None]:
    if unit.get("chunk_quality") == "noisy":
        return True, "chunk_quality=noisy"
    return False, None


def render_prompt(
    template: str,
    chunk_text: str,
    dfa_context: str,
    known_topics: list[str],
) -> str:
    return (
        template.replace("{{chunk_text}}", chunk_text)
        .replace("{{dfa_context}}", dfa_context)
        .replace(
            "{{known_topics}}",
            "\n".join(f"- {topic}" for topic in known_topics)
            if known_topics
            else "(none yet)",
        )
    )


def build_output_doc(
    doc_id: str | None, source_path: str | None, cfg: Config
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "source_path": source_path,
        "extraction_run": {
            "granularity": "paragraph",
            "schema": "dfa_promptlab_v2",
            "started_at": now_iso(),
            "finished_at": None,
            "resume": cfg.resume,
            "llm_config": {
                "model": cfg.model,
                "temperature": cfg.temperature,
                "max_completion_tokens": cfg.max_completion_tokens,
                "timeout_s": cfg.timeout_s,
            },
        },
        "extractions": {"paragraphs": []},
    }


def record_key(unit_kind: str, para_id: str) -> str:
    return f"{unit_kind}:{para_id}"


def load_existing_records(output_path: Path) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}

    prev = load_json(output_path)
    records: dict[str, dict[str, Any]] = {}

    for rec in prev.get("extractions", {}).get("paragraphs", []) or []:
        para_id = rec.get("para_id")
        if not para_id:
            continue
        meta = rec.get("meta", {})
        unit_kind = meta.get("unit_kind", "paragraph")
        records[record_key(unit_kind, para_id)] = rec

    return records


def load_topics_registry(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []

    try:
        payload = load_json(path)
    except Exception:
        logger.warning("Could not read topics registry at %s", path)
        return []

    if isinstance(payload, list):
        return sorted({str(x).strip() for x in payload if str(x).strip()})

    if isinstance(payload, dict) and isinstance(payload.get("topics"), list):
        return sorted({str(x).strip() for x in payload["topics"] if str(x).strip()})

    logger.warning("Unsupported topics registry format at %s", path)
    return []


def save_topics_registry(path: Path | None, topics: Iterable[str]) -> None:
    if path is None:
        return
    atomic_write_json(
        path,
        {"topics": sorted({topic.strip() for topic in topics if topic.strip()})},
    )


def merge_topics(existing: list[str], new_topics: Iterable[str]) -> list[str]:
    merged = set(existing)
    for topic in new_topics:
        topic = topic.strip()
        if topic:
            merged.add(topic)
    return sorted(merged)


def build_record(
    *,
    para_id: str,
    unit_kind: str,
    items: list[dict[str, Any]],
    skipped: bool,
    skip_reason: str | None,
    validation_ok: bool,
    errors: list[str],
    latency_ms: int | None,
    raw_response: str | None,
    save_raw_response_on_success: bool,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "skipped": skipped,
        "skip_reason": skip_reason,
        "validation_ok": validation_ok,
        "errors": errors,
        "repaired": False,
        "latency_ms": latency_ms,
        "unit_kind": unit_kind,
    }

    if raw_response is not None and (save_raw_response_on_success or not validation_ok):
        meta["raw_response"] = raw_response

    return {
        "para_id": para_id,
        "items": items,
        "meta": meta,
    }


async def extract_structured_output(prompt: str, cfg: Config) -> Attempt:
    started = time.time()

    try:
        response = await acompletion(
            model=cfg.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise information extraction system.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=ExtractionOutput,
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_completion_tokens,
            timeout=cfg.timeout_s,
        )
    except Exception as exc:
        return Attempt(
            output=None,
            raw_response=None,
            latency_ms=int((time.time() - started) * 1000),
            errors=[f"completion failed: {type(exc).__name__}: {exc}"],
        )

    latency_ms = int((time.time() - started) * 1000)
    message = response.choices[0].message  # type: ignore[union-attr]
    parsed = getattr(message, "parsed", None)
    raw_str = getattr(message, "content", None)
    if not isinstance(raw_str, str):
        raw_str = None

    if isinstance(parsed, ExtractionOutput):
        return Attempt(
            output=parsed, raw_response=raw_str, latency_ms=latency_ms, errors=[]
        )

    # Fallback for when structured output returns JSON text instead of a parsed object
    if raw_str and raw_str.strip():
        try:
            return Attempt(
                output=ExtractionOutput.model_validate_json(raw_str),
                raw_response=raw_str,
                latency_ms=latency_ms,
                errors=[],
            )
        except ValidationError as exc:
            return Attempt(
                output=None,
                raw_response=raw_str,
                latency_ms=latency_ms,
                errors=[
                    f"schema validation failed: {exc}",
                    f"raw_response_preview={raw_str[:500]!r}",
                ],
            )

    return Attempt(
        output=None,
        raw_response=None,
        latency_ms=latency_ms,
        errors=["model returned neither parsed output nor JSON text content"],
    )


async def process_one_doc(
    *,
    input_path: Path,
    output_path: Path,
    cfg: Config,
    registry_lock: asyncio.Lock | None,
) -> dict[str, Any]:
    doc = load_json(input_path)
    doc_id = str(doc.get("doc_id") or doc_id_from_path(input_path))
    source_path = doc.get("source_path")
    units = iter_units(doc)

    existing_records = load_existing_records(output_path) if cfg.resume else {}
    known_topics = (
        load_topics_registry(cfg.topics_registry_path)
        if cfg.use_topics_registry
        else []
    )

    out = build_output_doc(doc_id, source_path, cfg)

    n_total = 0
    n_reused = 0
    n_skipped = 0
    n_ok = 0
    n_failed = 0

    logger.info("Processing %s | units=%d", input_path.name, len(units))

    for unit in units:
        n_total += 1
        unit_kind = str(unit.get("unit_kind", "paragraph"))
        para_id = str(unit.get("para_id", ""))

        existing = existing_records.get(record_key(unit_kind, para_id))
        if cfg.resume and existing is not None:
            meta = existing.get("meta", {})
            if meta.get("skipped") or meta.get("validation_ok"):
                out["extractions"]["paragraphs"].append(existing)
                n_reused += 1
                continue

        if not para_id:
            record = build_record(
                para_id="__missing_para_id__",
                unit_kind=unit_kind,
                items=[],
                skipped=False,
                skip_reason=None,
                validation_ok=False,
                errors=["unit is missing para_id"],
                latency_ms=None,
                raw_response=None,
                save_raw_response_on_success=False,
            )
            out["extractions"]["paragraphs"].append(record)
            n_failed += 1
            continue

        skip, skip_reason = should_skip_unit(unit)
        if skip:
            record = build_record(
                para_id=para_id,
                unit_kind=unit_kind,
                items=[],
                skipped=True,
                skip_reason=skip_reason,
                validation_ok=False,
                errors=[],
                latency_ms=None,
                raw_response=None,
                save_raw_response_on_success=False,
            )
            out["extractions"]["paragraphs"].append(record)
            n_skipped += 1
            continue

        text = unit.get("text")
        if not isinstance(text, str) or not text.strip():
            record = build_record(
                para_id=para_id,
                unit_kind=unit_kind,
                items=[],
                skipped=False,
                skip_reason=None,
                validation_ok=False,
                errors=["unit text is missing or blank"],
                latency_ms=None,
                raw_response=None,
                save_raw_response_on_success=False,
            )
            out["extractions"]["paragraphs"].append(record)
            n_failed += 1
            continue

        prompt = render_prompt(cfg.prompt_template, text, cfg.dfa_context, known_topics)
        attempt = await extract_structured_output(prompt, cfg)

        if attempt.output is None:
            record = build_record(
                para_id=para_id,
                unit_kind=unit_kind,
                items=[],
                skipped=False,
                skip_reason=None,
                validation_ok=False,
                errors=attempt.errors,
                latency_ms=attempt.latency_ms,
                raw_response=attempt.raw_response,
                save_raw_response_on_success=cfg.save_raw_response_on_success,
            )
            out["extractions"]["paragraphs"].append(record)
            n_failed += 1
            continue

        items = [item.model_dump() for item in attempt.output.items]
        record = build_record(
            para_id=para_id,
            unit_kind=unit_kind,
            items=items,
            skipped=False,
            skip_reason=None,
            validation_ok=True,
            errors=[],
            latency_ms=attempt.latency_ms,
            raw_response=attempt.raw_response,
            save_raw_response_on_success=cfg.save_raw_response_on_success,
        )
        out["extractions"]["paragraphs"].append(record)
        n_ok += 1

        if cfg.use_topics_registry and cfg.topics_registry_path is not None:
            new_topics = [
                item.topic
                for item in attempt.output.items
                if item.topic_source != "dfa"
            ]
            if new_topics:
                if registry_lock is None:
                    known_topics = merge_topics(known_topics, new_topics)
                    save_topics_registry(cfg.topics_registry_path, known_topics)
                else:
                    async with registry_lock:
                        current = load_topics_registry(cfg.topics_registry_path)
                        merged = merge_topics(current, new_topics)
                        save_topics_registry(cfg.topics_registry_path, merged)
                        known_topics = merged

    out["extraction_run"]["finished_at"] = now_iso()
    atomic_write_json(output_path, out)

    logger.info(
        "Done %s | total=%d reused=%d skipped=%d ok=%d failed=%d | out=%s",
        doc_id,
        n_total,
        n_reused,
        n_skipped,
        n_ok,
        n_failed,
        output_path,
    )
    return out


async def process_one_doc_safe(
    *,
    input_path: Path,
    output_path: Path,
    cfg: Config,
    semaphore: asyncio.Semaphore,
    registry_lock: asyncio.Lock | None,
) -> None:
    async with semaphore:
        try:
            await process_one_doc(
                input_path=input_path,
                output_path=output_path,
                cfg=cfg,
                registry_lock=registry_lock,
            )
        except Exception:
            logger.exception("Document processing failed | input=%s", input_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structured DFA topic extraction over chunk JSON files."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path)
    src.add_argument("--input-dir", type=Path)

    dst = parser.add_mutually_exclusive_group(required=True)
    dst.add_argument("--output", type=Path)
    dst.add_argument("--output-dir", type=Path)

    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--dfa-context", type=Path, default=None)

    parser.add_argument(
        "--model", type=str, default=os.getenv("LLM_MODEL", "gpt-5-nano")
    )
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "1"))
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "40000")),
    )
    parser.add_argument(
        "--timeout-s", type=int, default=int(os.getenv("LLM_TIMEOUT_S", "60"))
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Scan --output-dir for existing extraction files and only reprocess documents "
            "that contain at least one failed record. Implies --resume so successful records "
            "are preserved. Requires --input-dir and --output-dir."
        ),
    )
    parser.add_argument("--ignore-ids", type=str, default=None)
    parser.add_argument("--ignore-file", type=Path, default=None)

    parser.add_argument("--topics-registry", type=Path, default=None)
    parser.add_argument("--no-topics-registry", action="store_true")

    parser.add_argument("--save-raw-response-on-success", action="store_true")
    parser.add_argument("--workers", type=int, default=8)

    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    configure_logging()

    litellm.enable_json_schema_validation = True

    cfg = Config(
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        timeout_s=args.timeout_s,
        prompt_template=read_text(args.prompt),
        dfa_context=read_text(args.dfa_context),
        resume=args.resume,
        topics_registry_path=args.topics_registry,
        use_topics_registry=not args.no_topics_registry
        and args.topics_registry is not None,
        save_raw_response_on_success=args.save_raw_response_on_success,
        workers=max(1, args.workers),
    )

    if args.retry_errors:
        if args.input_dir is None or args.output_dir is None:
            raise SystemExit("--retry-errors requires --input-dir and --output-dir")
        cfg.resume = True

    ignore_ids = parse_ignore_ids(args.ignore_ids) | read_ignore_file(args.ignore_file)

    try:
        if hasattr(litellm, "supports_response_schema"):
            if not litellm.supports_response_schema(model=cfg.model):
                logger.warning(
                    "Model may not support native structured outputs via response schema: %s",
                    cfg.model,
                )
    except Exception:
        logger.debug(
            "Could not verify structured-output support for model=%s", cfg.model
        )

    registry_lock = asyncio.Lock() if cfg.use_topics_registry else None

    if args.input is not None:
        if args.output is None:
            raise SystemExit("--output is required with --input")

        doc_id = doc_id_from_path(args.input)
        if doc_id in ignore_ids:
            logger.info("Ignoring doc_id=%s", doc_id)
            return

        await process_one_doc(
            input_path=args.input,
            output_path=args.output,
            cfg=cfg,
            registry_lock=registry_lock,
        )
        return

    if args.input_dir is None or args.output_dir is None:
        raise SystemExit("Directory mode requires --input-dir and --output-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(cfg.workers)
    tasks: list[asyncio.Task[None]] = []

    all_input_files = iter_input_files(args.input_dir)

    if args.retry_errors:
        with_errors = [
            p
            for p in all_input_files
            if doc_id_from_path(p) not in ignore_ids
            and output_has_errors(
                args.output_dir / f"{doc_id_from_path(p)}.paragraph.json"
            )
        ]
        logger.info(
            "retry-errors: %d/%d documents have failed records and will be reprocessed",
            len(with_errors),
            len(all_input_files),
        )
        input_files_to_run = with_errors
    else:
        input_files_to_run = all_input_files

    for input_path in input_files_to_run:
        doc_id = doc_id_from_path(input_path)
        if doc_id in ignore_ids:
            logger.info("Ignoring doc_id=%s", doc_id)
            continue

        output_path = args.output_dir / f"{doc_id}.paragraph.json"
        tasks.append(
            asyncio.create_task(
                process_one_doc_safe(
                    input_path=input_path,
                    output_path=output_path,
                    cfg=cfg,
                    semaphore=semaphore,
                    registry_lock=registry_lock,
                )
            )
        )

    if tasks:
        await asyncio.gather(*tasks)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
