from __future__ import annotations

import json
import logging
import os
import re
import shutil
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from schemas import DocumentExtractionResponse, DocumentPage, ExtractionResult

LOGGER = logging.getLogger("engramic.extraction")
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "demo_cache"
load_dotenv(BASE_DIR / ".env")

NIK_PATTERN = re.compile(r"(?<!\d)\d{16}(?!\d)")
DOB_PATTERN = re.compile(
    r"\b(?:tanggal\s+lahir|tgl\.?\s*lahir|dob)\s*[:\-]?\s*"
    r"(?:\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE,
)

MAX_PDF_PAGES = int(os.environ.get("ENGRAMIC_MAX_PDF_PAGES", "20"))
MIN_EMBEDDED_TEXT_CHARS = int(os.environ.get("ENGRAMIC_MIN_PDF_TEXT_CHARS", "40"))


def scrub_pii(raw_text: str, patient_name: str | None = None) -> str:
    """Best-effort demo scrubber; never treat regex-only scrubbing as full PDP compliance."""
    scrubbed = NIK_PATTERN.sub("[NIK_REDACTED]", raw_text)
    scrubbed = DOB_PATTERN.sub("Tanggal lahir: [DOB_REDACTED]", scrubbed)
    if patient_name and patient_name.strip():
        scrubbed = re.sub(
            rf"\b{re.escape(patient_name.strip())}\b",
            "[PATIENT_001]",
            scrubbed,
            flags=re.IGNORECASE,
        )
    return scrubbed


def _load_cached(note_id: str) -> ExtractionResult | None:
    path = CACHE_DIR / f"{note_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"] = "cache"
    return ExtractionResult.model_validate(payload)


def _rule_based_fallback(scrubbed: str) -> ExtractionResult:
    """Last-resort fallback for non-demo notes: safe, narrow, and deliberately incomplete."""
    entities: list[dict] = []
    lowered = scrubbed.lower()
    seen_drugs: set[str] = set()
    for variant, canonical in (
        ("amlodipin", "Amlodipine"),
        ("amlodipine", "Amlodipine"),
        ("lisinopril", "Lisinopril"),
        ("metformin", "Metformin"),
        ("captopril", "Captopril"),
        ("atorvastatin", "Atorvastatin"),
    ):
        if variant not in lowered:
            continue
        if canonical in seen_drugs:
            continue
        stopped = bool(re.search(rf"(?:{variant}.{{0,25}}(?:distop|dihentikan|stop|discontinue))|(?:(?:distop|dihentikan|stop|discontinue).{{0,25}}{variant})", lowered))
        continued = bool(re.search(rf"(?:lanjutkan|continue|tetap).{{0,25}}{variant}", lowered))
        dose_match = re.search(rf"{variant}.{{0,15}}?(\d+(?:\.\d+)?\s*mg)", lowered)
        entities.append({
            "entity_type": "medication", "name": canonical,
            "dose": dose_match.group(1) if dose_match else None,
            "status": "discontinued" if stopped else "active", "negated": False,
            "discontinued": stopped, "explicit_state_change": stopped or continued,
            "raw_text_span": canonical,
        })
        seen_drugs.add(canonical)
    condition_patterns = (
        (r"\b(?:hipertensi|hypertension|htn)\b", "Hypertension"),
        (r"\b(?:type\s*2 diabetes mellitus|diabetes melitus tipe\s*2|type\s*2 diabetes)\b", "Type 2 diabetes mellitus"),
        (r"\b(?:acute pharyngitis|faringitis akut)\b", "Acute pharyngitis"),
    )
    for pattern, canonical in condition_patterns:
        match = re.search(pattern, scrubbed, flags=re.IGNORECASE)
        if not match:
            continue
        entities.append({
            "entity_type": "condition",
            "name": canonical,
            "status": "active",
            "negated": False,
            "discontinued": False,
            "explicit_state_change": False,
            "raw_text_span": match.group(0),
        })

    allergy_match = re.search(
        r"\bAllerg(?:y|ies)\s*:\s*([A-Za-z][A-Za-z ]{1,35}?)\s*[-:]\s*(?:reported\s+)?([^\r\n;]+)",
        scrubbed,
        flags=re.IGNORECASE,
    )
    if allergy_match and "none reported" not in allergy_match.group(0).lower():
        entities.append({
            "entity_type": "allergy",
            "name": allergy_match.group(1).strip().title(),
            "reaction": allergy_match.group(2).strip().rstrip("."),
            "status": "active",
            "negated": False,
            "discontinued": False,
            "explicit_state_change": False,
            "raw_text_span": allergy_match.group(0).strip(),
        })

    diagnostic_match = re.search(
        r"\b(Throat examination)\s+(.+?)(?=\s+No inflammation\b|[\r\n])",
        scrubbed,
        flags=re.IGNORECASE,
    )
    if diagnostic_match:
        entities.append({
            "entity_type": "lab",
            "name": diagnostic_match.group(1).strip().title(),
            "value": diagnostic_match.group(2).strip(),
            "unit": None,
            "negated": False,
            "discontinued": False,
            "explicit_state_change": False,
            "raw_text_span": diagnostic_match.group(0).strip(),
        })
    return ExtractionResult(
        entities=entities,
        warnings=[
            "Live AI extraction and a matching demo cache were unavailable. "
            "Deterministic rule-based extraction was used; review every proposal and check the source for missing facts."
        ],
        source="rule_based",
        scrubbed_text=scrubbed,
    )


def _live_extract(scrubbed_text: str) -> ExtractionResult:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=5.0, max_retries=0)
    system = """You extract clinical entities from bilingual Indonesian/English notes.
Return only facts explicitly supported by the note. Set negated=true for denied findings.
For medication phrases such as distop/dihentikan/stop/discontinued, set status=discontinued,
discontinued=true, and explicit_state_change=true. For mulai/start/continue/lanjutkan,
set explicit_state_change=true. Preserve the supporting phrase in raw_text_span.
Do not infer dates, doses, diagnoses, or relationships that are absent."""
    completion = client.beta.chat.completions.parse(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": scrubbed_text}],
        response_format=ExtractionResult,
        store=False,
        temperature=0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parsed extraction")
    parsed.source = "live_llm"
    parsed.scrubbed_text = scrubbed_text
    return parsed


def extract_clinical_entities(
    raw_text: str, note_id: str, patient_name: str | None = None
) -> ExtractionResult:
    LOGGER.info("Scrubbing PII for note %s", note_id)
    scrubbed = scrub_pii(raw_text, patient_name)
    if os.environ.get("ENGRAMIC_FORCE_CACHE", "").lower() in {"1", "true", "yes"}:
        cached = _load_cached(note_id)
        if cached:
            cached.scrubbed_text = scrubbed
            return cached
    try:
        LOGGER.info("Calling structured extraction for note %s", note_id)
        return _live_extract(scrubbed)
    except Exception as exc:
        LOGGER.warning("Live extraction failed for %s: %s", note_id, exc)
        cached = _load_cached(note_id)
        if cached:
            cached.warnings.append("Live extraction unavailable; verified demo cache used.")
            cached.scrubbed_text = scrubbed
            return cached
        return _rule_based_fallback(scrubbed)


def _configure_tesseract() -> None:
    import pytesseract

    configured = os.environ.get("TESSERACT_CMD")
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    elif not shutil.which("tesseract"):
        common = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if common.exists():
            pytesseract.pytesseract.tesseract_cmd = str(common)


def get_ocr_status() -> dict[str, str | bool]:
    import pytesseract

    _configure_tesseract()
    try:
        version = str(pytesseract.get_tesseract_version())
        return {
            "ready": True,
            "engine": "tesseract",
            "version": version,
            "language": os.environ.get("TESSERACT_LANG", "eng+ind"),
        }
    except Exception as exc:
        return {
            "ready": False,
            "engine": "tesseract",
            "error": str(exc),
            "setup": "Install Tesseract or set TESSERACT_CMD in .env",
        }


def _ocr_page(page) -> tuple[str, float | None, list[str]]:
    import pytesseract
    from PIL import Image

    _configure_tesseract()
    # Render at 2x (144 DPI) for better OCR while keeping the demo responsive.
    pixmap = page.get_pixmap(matrix=__import__("fitz").Matrix(2, 2), alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
    warnings: list[str] = []
    language = os.environ.get("TESSERACT_LANG", "eng+ind")
    try:
        text = pytesseract.image_to_string(image, lang=language, config="--psm 6", timeout=15)
        data = pytesseract.image_to_data(
            image,
            lang=language,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
            timeout=15,
        )
    except pytesseract.TesseractError as exc:
        if language != "eng":
            warnings.append(f"OCR language '{language}' unavailable; English fallback used.")
            text = pytesseract.image_to_string(image, lang="eng", config="--psm 6", timeout=15)
            data = pytesseract.image_to_data(
                image,
                lang="eng",
                config="--psm 6",
                output_type=pytesseract.Output.DICT,
                timeout=15,
            )
        else:
            raise RuntimeError(f"Tesseract OCR failed: {exc}") from exc

    confidences = []
    for raw in data.get("conf", []):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confidences.append(value)
    confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
    if confidence is not None and confidence < 70:
        warnings.append("Low OCR confidence; careful clinician review required.")
    return text.strip(), confidence, warnings


def extract_pdf_document(
    pdf_bytes: bytes,
    filename: str,
    note_id: str,
    patient_name: str | None = None,
) -> DocumentExtractionResponse:
    """Extract a PDF into a reviewable clinical preview; this never mutates the graph."""
    import fitz

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Uploaded file is not a valid PDF")
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unable to open PDF: {exc}") from exc
    if document.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported")
    if document.page_count == 0:
        raise ValueError("PDF contains no pages")
    if document.page_count > MAX_PDF_PAGES:
        raise ValueError(f"PDF has {document.page_count} pages; maximum is {MAX_PDF_PAGES}")

    pages: list[DocumentPage] = []
    document_warnings: list[str] = []
    for index, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        method = "embedded_text"
        confidence = None
        page_warnings: list[str] = []
        if len(text) < MIN_EMBEDDED_TEXT_CHARS:
            method = "tesseract_ocr"
            try:
                text, confidence, page_warnings = _ocr_page(page)
            except Exception as exc:
                text = ""
                page_warnings.append(f"OCR unavailable: {exc}")
        scrubbed = scrub_pii(text, patient_name)
        if not scrubbed:
            page_warnings.append("No readable text found on this page.")
        pages.append(DocumentPage(
            page=index,
            method=method,
            text=scrubbed,
            character_count=len(scrubbed),
            ocr_confidence=confidence,
            warnings=page_warnings,
        ))
        document_warnings.extend(f"Page {index}: {warning}" for warning in page_warnings)
    document.close()

    readable_pages = [page for page in pages if page.text]
    if not readable_pages:
        raise ValueError("No readable text could be extracted; install/configure Tesseract for scanned PDFs")
    combined = "\n\n".join(f"[PAGE {page.page}]\n{page.text}" for page in readable_pages)
    extraction = extract_clinical_entities(combined, note_id, patient_name=None)
    for entity in extraction.entities:
        if entity.source_page is not None or not entity.raw_text_span:
            continue
        evidence = entity.raw_text_span.casefold()
        matching_page = next(
            (page.page for page in readable_pages if evidence in page.text.casefold()),
            None,
        )
        entity.source_page = matching_page
    extraction.warnings.extend(document_warnings)
    return DocumentExtractionResponse(
        document_id=f"doc_{uuid4().hex[:12]}",
        filename=Path(filename).name or "uploaded.pdf",
        content_type="application/pdf",
        size_bytes=len(pdf_bytes),
        page_count=len(pages),
        used_ocr=any(page.method == "tesseract_ocr" for page in pages),
        pages=pages,
        extraction=extraction,
        warnings=document_warnings,
    )
