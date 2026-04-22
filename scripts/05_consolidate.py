import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def doc_id_from_file(p: Path) -> str:
    name = p.name
    for suffix in [".paragraph.json", ".chunk.json", ".json"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return p.stem


def index_chunks_by_para_id(chunks_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    for p in chunks_doc.get("paragraphs", []) or []:
        pid = p.get("para_id")
        if pid:
            out[pid] = p

    for p in chunks_doc.get("feedback", []) or []:
        pid = p.get("para_id")
        if pid:
            out[pid] = p

    return out


def resolve_chunk_path(
    chunks_dir: Path, ex_doc: dict[str, Any], ex_path: Path
) -> Path | None:
    # Prefer doc_id inside JSON, because filenames may omit suffixes like ".clean"
    doc_id = ex_doc.get("doc_id")
    candidates: list[Path] = []

    if isinstance(doc_id, str) and doc_id.strip():
        candidates.append(chunks_dir / f"{doc_id}.chunk.json")
    stem_id = doc_id_from_file(ex_path)
    candidates.append(chunks_dir / f"{stem_id}.chunk.json")
    candidates.append(chunks_dir / f"{stem_id}.clean.chunk.json")

    for c in candidates:
        if c.exists():
            return c
    return None


def enrich_extractions_doc(
    *,
    extraction_path: Path,
    chunks_dir: Path,
    out_path: Path,
    include_chunk_meta: bool,
) -> dict[str, Any]:
    ex = load_json(extraction_path)

    chunks_path = resolve_chunk_path(chunks_dir, ex, extraction_path)
    if chunks_path is None:
        raise FileNotFoundError(
            f"Could not find matching chunk file for extraction={extraction_path.name} "
            f"(doc_id={ex.get('doc_id')!r}) in chunks_dir={chunks_dir}"
        )

    ch = load_json(chunks_path)

    ex_doc_id = ex.get("doc_id") or doc_id_from_file(extraction_path)
    ch_doc_id = ch.get("doc_id") or doc_id_from_file(chunks_path)
    doc_id_mismatch = ex_doc_id != ch_doc_id

    ch_index = index_chunks_by_para_id(ch)

    n_total = 0
    n_found = 0
    n_missing = 0

    enriched_paras: list[dict[str, Any]] = []
    for rec in ex.get("extractions", {}).get("paragraphs", []) or []:
        n_total += 1
        pid = rec.get("para_id")
        chunk_p = ch_index.get(pid) if pid else None

        new_rec = dict(rec)

        if chunk_p is None:
            n_missing += 1
            new_rec["text"] = None
            if include_chunk_meta:
                new_rec["chunk_meta"] = None
        else:
            n_found += 1
            new_rec["text"] = chunk_p.get("text")
            if include_chunk_meta:
                new_rec["chunk_meta"] = {
                    "section_id": chunk_p.get("section_id"),
                    "section_path": chunk_p.get("section_path"),
                    "span": chunk_p.get("span"),
                    "chunk_quality": chunk_p.get("chunk_quality"),
                    "chunk_word_count": chunk_p.get("chunk_word_count"),
                    "char_len": chunk_p.get("char_len"),
                }

        enriched_paras.append(new_rec)

    out = dict(ex)
    out["consolidated_at"] = now_utc_iso()
    out["consolidation"] = {
        "chunks_path": str(chunks_path),
        "include_chunk_meta": include_chunk_meta,
        "doc_id_mismatch": doc_id_mismatch,
        "ex_doc_id": ex_doc_id,
        "chunks_doc_id": ch_doc_id,
        "stats": {
            "paragraphs_total": n_total,
            "paragraphs_text_found": n_found,
            "paragraphs_text_missing": n_missing,
        },
    }
    out.setdefault("extractions", {})
    out["extractions"]["paragraphs"] = enriched_paras

    atomic_write_json(out_path, out)
    return out


def iter_extraction_files(extractions_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in extractions_dir.iterdir()
            if p.is_file() and p.name.endswith(".paragraph.json")
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Enrich LLM extraction JSONs with paragraph text from corresponding chunk JSONs."
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--extractions", type=Path, help="Single *.paragraph.json file")
    src.add_argument(
        "--extractions-dir", type=Path, help="Directory with *.paragraph.json files"
    )

    ap.add_argument(
        "--chunks-dir",
        type=Path,
        required=True,
        help="Directory containing *.chunk.json files",
    )

    dst = ap.add_mutually_exclusive_group(required=True)
    dst.add_argument("--out", type=Path, help="Single output file")
    dst.add_argument("--out-dir", type=Path, help="Output directory")

    ap.add_argument(
        "--include-chunk-meta",
        action="store_true",
        help="Also include small chunk metadata (section_id/path/span/quality/word_count).",
    )

    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast if any extraction file cannot be matched to a chunk file.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    chunks_dir: Path = args.chunks_dir

    if args.extractions:
        out_path: Path = args.out
        enrich_extractions_doc(
            extraction_path=args.extractions,
            chunks_dir=chunks_dir,
            out_path=out_path,
            include_chunk_meta=bool(args.include_chunk_meta),
        )
        return

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for ex_path in iter_extraction_files(args.extractions_dir):
        out_path = out_dir / ex_path.name  # keep same filename
        try:
            enrich_extractions_doc(
                extraction_path=ex_path,
                chunks_dir=chunks_dir,
                out_path=out_path,
                include_chunk_meta=bool(args.include_chunk_meta),
            )
        except Exception as e:
            failures.append(f"{ex_path.name}: {e}")
            if args.strict:
                raise

    if failures:
        report = out_dir / "_consolidation_failures.txt"
        report.write_text("\n".join(failures) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
