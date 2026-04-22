import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mistralai import Mistral

LOGGER = logging.getLogger("mistral_ocr")


@dataclass(frozen=True)
class OcrResult:
    pdf_path: Path
    json_path: Path
    md_path: Path
    ok: bool
    error: str | None
    pages: int | None


def build_client(api_key: str | None = None) -> Mistral:
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY (env var) or api_key argument.")
    return Mistral(api_key=api_key)


def iter_pdfs(root_dir: Path) -> list[Path]:
    root_dir = root_dir.expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    return sorted([p for p in root_dir.rglob("*.pdf") if p.is_file()])


def upload_pdf(client: Mistral, pdf_path: Path) -> str:
    pdf_path = pdf_path.expanduser().resolve()
    uploaded = client.files.upload(
        file={"file_name": pdf_path.name, "content": pdf_path.open("rb")},
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded.id)
    return signed_url.url


def ocr_pdf(
    client: Mistral,
    pdf_path: Path,
    *,
    model: str = "mistral-ocr-latest",
    include_image_base64: bool = True,
):
    return client.ocr.process(
        model=model,
        document={"type": "document_url", "document_url": upload_pdf(client, pdf_path)},
        include_image_base64=include_image_base64,
    )


def write_json(ocr_response, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ocr_response.model_dump_json(indent=2), encoding="utf-8")


def write_markdown(ocr_response, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wt", encoding="utf-8") as f:
        for page_index, page in enumerate(ocr_response.pages):
            f.write(
                f"<!-- PAGE {page_index + 1} of {len(ocr_response.pages)} -->\n\n{page.markdown}"
            )


def _safe_relpath(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def process_pdf(
    client: Mistral,
    pdf_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    model: str = "mistral-ocr-latest",
    include_image_base64: bool = True,
) -> OcrResult:
    rel = _safe_relpath(pdf_path, input_root)
    stem = rel.with_suffix("")  # keep nested folders, drop .pdf
    json_path = (output_root / stem).with_suffix(".json")
    md_path = (output_root / stem).with_suffix(".md")

    try:
        resp = ocr_pdf(
            client,
            pdf_path,
            model=model,
            include_image_base64=include_image_base64,
        )
        write_json(resp, json_path)
        write_markdown(resp, md_path)

        pages = len(getattr(resp, "pages", []) or [])
        return OcrResult(
            pdf_path=pdf_path,
            json_path=json_path,
            md_path=md_path,
            ok=True,
            error=None,
            pages=pages,
        )
    except Exception as e:  # noqa: BLE001
        return OcrResult(
            pdf_path=pdf_path,
            json_path=json_path,
            md_path=md_path,
            ok=False,
            error=str(e),
            pages=None,
        )


def write_report(results: list[OcrResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total": len(results),
        "ok": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok),
        "items": [
            {
                "pdf_path": str(r.pdf_path),
                "json_path": str(r.json_path),
                "md_path": str(r.md_path),
                "ok": r.ok,
                "pages": r.pages,
                "error": r.error,
            }
            for r in results
        ],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def process_directory(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    api_key: str | None = None,
    model: str = "mistral-ocr-latest",
    include_image_base64: bool = True,
) -> Path:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (input_root / "_mistral_ocr")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    setup_logging(log_path=output_root / "process.log")

    client = build_client(api_key=api_key)
    pdfs = iter_pdfs(input_root)

    LOGGER.info("Input: %s", input_root)
    LOGGER.info("Output: %s", output_root)
    LOGGER.info("Found %s PDF(s)", len(pdfs))

    results: list[OcrResult] = []
    for i, pdf_path in enumerate(pdfs, start=1):
        LOGGER.info("[%s/%s] OCR: %s", i, len(pdfs), pdf_path)
        res = process_pdf(
            client,
            pdf_path,
            input_root=input_root,
            output_root=output_root,
            model=model,
            include_image_base64=include_image_base64,
        )
        results.append(res)

        if res.ok:
            LOGGER.info(
                "OK (%s page(s)) -> %s | %s", res.pages, res.json_path, res.md_path
            )
        else:
            LOGGER.error("FAILED -> %s", res.error)

    report_path = output_root / "report.json"
    write_report(results, report_path)

    ok = sum(1 for r in results if r.ok)
    failed = len(results) - ok
    LOGGER.info("Done. OK=%s FAILED=%s Report=%s", ok, failed, report_path)

    return report_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python mistral_pdf_ocr.py <input_dir> [output_dir]")

    in_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    process_directory(in_dir, output_dir=out_dir)
