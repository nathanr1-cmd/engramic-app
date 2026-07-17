from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from auth import authorize_patient_access, grant_patient_access, initialize_auth, require_api_role, require_role
from extraction import extract_clinical_entities, extract_pdf_document, get_ocr_status
from graph_engine import STORE, duplicate_candidates, get_active_subgraph, get_full_graph, merge_entities, resolve_conflict
from schemas import Allergy, Condition, ConflictResolution, DocumentExtractionResponse, ExtractRequest, Facility, LabResult, Medication, MergeRequest, Patient, PatientCreate
import clinical_db
from web import router as web_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="Engramic API", version="0.1.0")
ALLOWED_ORIGINS = [origin.strip().rstrip("/") for origin in os.environ.get("ENGRAMIC_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",") if origin.strip()]
if render_url := os.environ.get("RENDER_EXTERNAL_URL", "").strip().rstrip("/"):
    ALLOWED_ORIGINS.append(render_url)
ALLOWED_ORIGINS = list(dict.fromkeys(ALLOWED_ORIGINS))
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Accept"])


@app.middleware("http")
async def same_origin_writes(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and origin and origin.rstrip("/") not in ALLOWED_ORIGINS:
        return JSONResponse(status_code=403, content={"detail": "Cross-origin write rejected"})
    return await call_next(request)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
app.include_router(web_router)
initialize_auth()
clinical_db.initialize()
MAX_UPLOAD_BYTES = int(os.environ.get("ENGRAMIC_MAX_UPLOAD_MB", "10")) * 1024 * 1024


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ocr")
def ocr_health():
    return get_ocr_status()


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page(request: Request) -> str:
    require_role(request, "doctor")
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Engramic PDF Intake</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, sans-serif; color: #14322b; background: #f4f8f6; }
    body { margin: 0; padding: 32px 18px; }
    main { max-width: 820px; margin: auto; }
    h1 { margin-bottom: 6px; } p { color: #536761; }
    .fields { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 24px 0; }
    label { font-size: 13px; font-weight: 700; }
    input { box-sizing: border-box; width: 100%; margin-top: 6px; padding: 11px; border: 1px solid #b8c9c3; border-radius: 9px; }
    #drop { padding: 42px 20px; text-align: center; border: 2px dashed #3b8b74; border-radius: 16px; background: white; cursor: pointer; }
    #drop.over { background: #e5f5ef; border-color: #176c55; }
    button { margin-top: 16px; padding: 12px 18px; border: 0; border-radius: 9px; background: #176c55; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .5; cursor: wait; }
    pre { white-space: pre-wrap; word-break: break-word; padding: 18px; border-radius: 12px; background: #10251f; color: #dff7ed; min-height: 44px; }
    .notice { padding: 12px 14px; border-radius: 9px; background: #fff5cf; color: #624e00; font-size: 14px; }
    @media (max-width: 650px) { .fields { grid-template-columns: 1fr; } }
  </style>
</head>
<body><main>
  <h1>Engramic PDF Intake</h1>
  <p>Drop a clinical PDF to extract a review preview. Nothing is merged into the patient graph automatically.</p>
  <div class="notice">Printed PDFs work immediately. Scanned PDFs require Tesseract OCR to report ready at <code>/health/ocr</code>.</div>
  <div class="fields">
    <label>Patient ID<input id="patient" value="patient_budi" required></label>
    <label>Note ID<input id="note" value="pdf_note_1" required></label>
    <label>Patient name<input id="name" value="Pak Budi"></label>
  </div>
  <div id="drop" role="button" tabindex="0" aria-label="Drop PDF here or choose a file">
    <strong id="fileLabel">Drop PDF here</strong><br><span>or click to choose a file (maximum 10 MB)</span>
    <input id="file" type="file" accept="application/pdf,.pdf" hidden>
  </div>
  <button id="extract" disabled>Extract for review</button>
  <h2>Preview</h2><pre id="result">No document processed yet.</pre>
  <script>
    const drop = document.querySelector('#drop'), file = document.querySelector('#file');
    const button = document.querySelector('#extract'), result = document.querySelector('#result');
    let selected;
    function choose(f) {
      if (!f || (f.type && f.type !== 'application/pdf') || !f.name.toLowerCase().endsWith('.pdf')) {
        result.textContent = 'Please choose a PDF file.'; return;
      }
      selected = f; document.querySelector('#fileLabel').textContent = f.name; button.disabled = false;
    }
    drop.onclick = () => file.click(); drop.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') file.click(); };
    file.onchange = () => choose(file.files[0]);
    for (const event of ['dragenter','dragover']) drop.addEventListener(event, e => { e.preventDefault(); drop.classList.add('over'); });
    for (const event of ['dragleave','drop']) drop.addEventListener(event, e => { e.preventDefault(); drop.classList.remove('over'); });
    drop.addEventListener('drop', e => choose(e.dataTransfer.files[0]));
    button.onclick = async () => {
      const patient = document.querySelector('#patient').value.trim(), note = document.querySelector('#note').value.trim();
      if (!selected || !patient || !note) { result.textContent = 'Patient ID, note ID, and PDF are required.'; return; }
      const data = new FormData(); data.append('document', selected); data.append('patient_id', patient); data.append('note_id', note);
      data.append('patient_name', document.querySelector('#name').value.trim());
      button.disabled = true; button.textContent = 'Extracting...'; result.textContent = 'Reading PDF and preparing clinical preview...';
      try {
        const response = await fetch('/documents/extract', { method: 'POST', body: data });
        const body = await response.json(); result.textContent = JSON.stringify(body, null, 2);
        if (!response.ok) result.textContent = `Error ${response.status}\n` + result.textContent;
      } catch (error) { result.textContent = 'Upload failed: ' + error; }
      finally { button.disabled = false; button.textContent = 'Extract for review'; }
    };
  </script>
</main></body></html>"""


@app.post("/patients", response_model=Patient)
def create_patient(body: PatientCreate, request: Request) -> Patient:
    user = require_api_role(request, "doctor")
    if body.id in STORE.nodes:
        raise HTTPException(409, "Patient ID already exists")
    matches = duplicate_candidates(body.model_dump())
    if matches:
        raise HTTPException(409, {"message": "Possible duplicate patient", "matches": [
            {"patient_id": item["patient"].id, "name": item["patient"].name, "reasons": item["reasons"]}
            for item in matches]})
    patient = STORE.add_patient(Patient(**body.model_dump()))
    grant_patient_access(user.id, patient.id)
    clinical_db.audit("patient_created", patient.id, user.email, after=patient.model_dump(mode="json"))
    return patient


@app.post("/notes/extract")
def extract_note(body: ExtractRequest, request: Request):
    user = require_api_role(request, "doctor")
    authorize_patient_access(request, body.patient_id)
    result = extract_clinical_entities(body.raw_text, body.note_id, body.patient_name)
    clinical_db.audit("extraction_completed", body.patient_id, user.email, note_id=body.note_id,
                      after={"entity_count": len(result.entities), "source": result.source})
    return result


@app.post("/documents/extract")
async def extract_document(
    request: Request,
    document: UploadFile = File(..., description="PDF clinical document"),
    patient_id: str = Form(...),
    note_id: str = Form(...),
    patient_name: str | None = Form(None),
) -> DocumentExtractionResponse:
    """Upload/drop a PDF and return OCR + entity previews without modifying the graph."""
    user = require_api_role(request, "doctor")
    authorize_patient_access(request, patient_id)
    if patient_id not in STORE.nodes or not isinstance(STORE.nodes[patient_id], Patient):
        raise HTTPException(404, "Patient not found")
    if document.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(415, "Only PDF files are supported")
    payload = await document.read(MAX_UPLOAD_BYTES + 1)
    await document.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"PDF exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        result = extract_pdf_document(payload, document.filename or "uploaded.pdf", note_id, patient_name)
        clinical_db.store_document(patient_id, note_id, document.filename or "uploaded.pdf",
                                   document.content_type or "application/pdf", payload,
                                   result.model_dump(mode="json"))
        clinical_db.audit("extraction_completed", patient_id, user.email, note_id=note_id,
                          document_id=result.document_id,
                          after={"entity_count": len(result.extraction.entities), "pages": result.page_count})
        return result
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logging.exception("PDF extraction failed safely")
        raise HTTPException(422, f"PDF extraction failed: {exc}") from exc


@app.post("/notes/merge")
def merge_note(body: MergeRequest, request: Request):
    user = require_api_role(request, "doctor")
    authorize_patient_access(request, body.patient_id)
    try:
        from uuid import uuid4
        review_id = body.review_id or uuid4().hex
        with clinical_db.transaction() as db:
            for decision in body.review_decisions:
                db.execute("INSERT INTO review_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    uuid4().hex, review_id, body.patient_id, body.note_id, body.document_id,
                    decision.fact_index, decision.decision,
                    json.dumps(decision.original, ensure_ascii=False, default=str) if decision.original else None,
                    json.dumps(decision.reviewed, ensure_ascii=False, default=str) if decision.reviewed else None,
                    user.email, clinical_db.utcnow()))
                event = {"include": "extracted_fact_included", "exclude": "extracted_fact_excluded", "reject": "extracted_fact_rejected"}[decision.decision]
                clinical_db.audit(event, body.patient_id, user.email, note_id=body.note_id,
                                  document_id=body.document_id, before=decision.original,
                                  after=decision.reviewed, correlation_id=review_id, db=db)
                if decision.decision == "include" and decision.original != decision.reviewed:
                    clinical_db.audit("extracted_fact_edited", body.patient_id, user.email,
                                      note_id=body.note_id, document_id=body.document_id,
                                      before=decision.original, after=decision.reviewed,
                                      correlation_id=review_id, db=db)
            clinical_db.audit("review_approved", body.patient_id, user.email, note_id=body.note_id,
                              document_id=body.document_id, after={"included": len(body.entities)},
                              correlation_id=review_id, db=db)
        return merge_entities(body.patient_id, body.entities, body.note_id, body.facility_id,
                              body.encounter_date, user.email, body.condition_links,
                              body.extraction_method, body.document_id, review_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logging.exception("Merge failed safely")
        raise HTTPException(422, f"Merge rejected: {exc}") from exc


@app.get("/documents/{document_id}/file")
def get_document_file(document_id: str, request: Request):
    user = require_api_role(request, "doctor")
    row = clinical_db.document(document_id)
    if not row:
        raise HTTPException(404, "Document not found")
    authorize_patient_access(request, row["patient_id"])
    path = clinical_db.DOCUMENT_DIR / row["storage_name"]
    if not path.is_file() or path.parent.resolve() != clinical_db.DOCUMENT_DIR.resolve():
        raise HTTPException(404, "Document file not found")
    return FileResponse(path, media_type="application/pdf", filename=row["filename"])


@app.get("/documents/{document_id}/pages")
def get_document_pages(document_id: str, request: Request):
    require_api_role(request, "doctor")
    row = clinical_db.document(document_id)
    if not row:
        raise HTTPException(404, "Document not found")
    authorize_patient_access(request, row["patient_id"])
    return {"document_id": document_id, "pages": clinical_db.document_pages(document_id)}


@app.post("/conflicts/{conflict_id}/resolve")
def resolve_conflict_api(conflict_id: str, body: ConflictResolution, request: Request):
    user = require_api_role(request, "doctor")
    item = clinical_db.conflict(conflict_id)
    if not item:
        raise HTTPException(404, "Conflict not found")
    authorize_patient_access(request, item["patient_id"])
    try:
        return resolve_conflict(conflict_id, body.action, user.email, body.comment)
    except KeyError:
        raise HTTPException(404, "Conflict not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/patients/{patient_id}/graph")
def graph(patient_id: str, request: Request):
    authorize_patient_access(request, patient_id)
    if patient_id not in STORE.nodes:
        raise HTTPException(404, "Patient not found")
    return get_full_graph(patient_id)


def _patient_view(patient_id: str) -> str:
    graph = get_full_graph(patient_id)
    active = [f"{n.name} {n.dose or ''}".strip() for n in graph.nodes if isinstance(n, Medication) and n.status.value == "active"]
    stopped = [f"{n.name} {n.dose or ''}".strip() for n in graph.nodes if isinstance(n, Medication) and n.status.value == "discontinued"]
    allergies = [n.substance for n in graph.nodes if isinstance(n, Allergy) and n.status == "active"]
    return " ".join([
        f"Obat Anda saat ini: {', '.join(active) if active else 'belum tercatat'}.",
        f"Obat yang sudah berhenti: {', '.join(stopped) if stopped else 'tidak ada yang tercatat'}.",
        f"Alergi yang tercatat: {', '.join(allergies) if allergies else 'tidak ada'}.",
        "Pastikan informasi ini bersama dokter atau petugas kesehatan Anda.",
    ])


@app.get("/patients/{patient_id}/patient-view")
def patient_view(patient_id: str, request: Request) -> dict[str, str]:
    authorize_patient_access(request, patient_id)
    if patient_id not in STORE.nodes:
        raise HTTPException(404, "Patient not found")
    return {"summary": _patient_view(patient_id)}


def _deterministic_referral(patient_id: str) -> str:
    full = get_full_graph(patient_id)
    patient = next((n for n in full.nodes if isinstance(n, Patient)), None)
    conditions = [n.name for n in full.nodes if isinstance(n, Condition) and n.status.value in {"active", "suspected"}]
    active = [f"{n.name} {n.dose or ''} {n.frequency or ''}".strip() for n in full.nodes if isinstance(n, Medication) and n.status.value == "active"]
    stopped = [f"{n.name} {n.dose or ''}".strip() for n in full.nodes if isinstance(n, Medication) and n.status.value == "discontinued"]
    conflicts = [r for n in full.nodes for r in getattr(n, "conflict_reasons", [])]
    provenance = sorted({p for n in full.nodes for p in n.provenance})
    return f"SURAT RUJUKAN\n\nPasien: {patient.name if patient else patient_id}\nMasalah aktif: {', '.join(conditions) or 'Belum tercatat'}\nObat saat ini: {', '.join(active) or 'Belum tercatat'}\nObat dihentikan: {', '.join(stopped) or 'Tidak ada yang tercatat'}\nKonflik terbuka: {'; '.join(conflicts) or 'Tidak ada'}\nSumber: {', '.join(provenance) or 'Tidak tersedia'}\n\nHarap verifikasi ringkasan ini terhadap catatan sumber dan penilaian klinis sebelum digunakan untuk keputusan medis."


async def _summary_stream(patient_id: str) -> AsyncIterator[str]:
    fallback = _deterministic_referral(patient_id)
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=5.0, max_retries=0)
        snapshot = get_full_graph(patient_id).model_dump(mode="json")
        stream = await client.chat.completions.create(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), messages=[{"role": "system", "content": "Write a concise Indonesian clinical referral letter using only the verified graph snapshot. Separate active and discontinued medications, cite note IDs inline, list conflicts, and end with a verification disclaimer."}, {"role": "user", "content": json.dumps(snapshot, ensure_ascii=False)}], stream=True, store=False, temperature=0)
        async for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text: yield text
    except Exception as exc:
        logging.warning("Summary LLM unavailable; deterministic fallback used: %s", exc)
        for line in fallback.splitlines(keepends=True):
            yield line
            await asyncio.sleep(0)


@app.get("/patients/{patient_id}/summary")
def summary(patient_id: str, request: Request):
    authorize_patient_access(request, patient_id)
    if patient_id not in STORE.nodes:
        raise HTTPException(404, "Patient not found")
    return StreamingResponse(_summary_stream(patient_id), media_type="text/plain; charset=utf-8")
