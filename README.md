# Engramic full-stack clinical record

Engramic is a role-aware FastAPI application built around a review-first, stateful clinical knowledge graph. The LLM or deterministic fallback proposes structured facts; only an authenticated clinician can review and merge them into the verified graph.

## Architecture

- **Application:** FastAPI with server-rendered Jinja templates and a responsive graphite/lime glass UI.
- **Clinical state:** Existing Pydantic schemas and deterministic graph engine backed by SQLite. `graph_data.json` is imported once with a preserved backup.
- **Authentication:** Persistent SQLite users, scrypt password hashes, signed HttpOnly session cookies, and server-side role checks.
- **Patient isolation:** Persistent `doctor_patient_access` assignments plus patient-account ownership checks.
- **Document intake:** PyMuPDF embedded-text extraction with Tesseract OCR fallback; document metadata and extracted page text are retained in SQLite.
- **AI:** Structured OpenAI extraction when configured, verified demo cache for Notes 1–3, and a narrow rule-based fallback.

The frontend and API share one origin. Clinical source text is never merged automatically.

Version 2 adds auditable include/exclude/reject decisions, persisted conflict resolution, richer patient profiles, and deterministic duplicate warnings. The stronger assignment-based authorization remains in force: doctors see only assigned records and patients see only their own record.

## Setup and run

```powershell
cd C:\Users\Kiers\OneDrive\Documents\research\engramic-app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
$env:ENGRAMIC_FORCE_CACHE="true"
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

### Local demo accounts

These accounts are seeded only when `ENGRAMIC_SEED_DEMO_ACCOUNTS=true`:

| Role | Email | Password | Access |
| --- | --- | --- | --- |
| Doctor | `doctor@engramic.id` | `Doctor123!` | Assigned to `patient_budi` |
| Patient | `patient@engramic.id` | `Patient123!` | Own record only |

Set `ENGRAMIC_SEED_DEMO_ACCOUNTS=false`, use a long random `ENGRAMIC_SESSION_SECRET`, and enable secure cookies before any shared deployment.

## Route map

Public:

- `/`
- `/login`
- `/register` (patient accounts only)

Doctor:

- `/doctor/dashboard`
- `/doctor/patients`
- `/doctor/patients/{patientId}`
- `/doctor/notes/new`
- `/doctor/referral-summary`

Patient:

- `/patient/dashboard`
- `/patient/medications`
- `/patient/conditions`
- `/patient/history`

API and diagnostics:

- `GET /health`
- `GET /health/ocr`
- `POST /patients`
- `POST /notes/extract`
- `POST /documents/extract`
- `POST /notes/merge`
- `GET /patients/{patient_id}/graph`
- `GET /patients/{patient_id}/patient-view`
- `GET /patients/{patient_id}/summary`
- `/docs`

## Review-first clinical flow

1. A doctor selects an authorized patient.
2. The doctor types a note or uploads a PDF.
3. Engramic returns scrubbed source, warnings, OCR/page metadata, and unverified entity proposals.
4. The doctor edits proposals, excludes incorrect items, and links medications to conditions.
5. Explicit confirmation calls the deterministic merge endpoint.
6. The UI shows created, updated, skipped, conflict, and fuzzy-review-required results.
7. The verified graph, patient view, history, and referral summary update from merged facts only.

The doctor dashboard also includes a repeat-safe **Load PRD demo** option. It loads the original hypertension/Amlodipine/Lisinopril story plus coherent dyslipidemia→Atorvastatin and type-2-diabetes→Metformin branches with verified lab context.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Optional live extraction and summary generation |
| `OPENAI_MODEL` | OpenAI model name |
| `ENGRAMIC_FORCE_CACHE` | Prefer verified demo cache for known note IDs |
| `ENGRAMIC_SESSION_SECRET` | Signs session cookies; required with secure cookies |
| `ENGRAMIC_SECURE_COOKIES` | Sets cookie `Secure` flag behind HTTPS |
| `ENGRAMIC_SEED_DEMO_ACCOUNTS` | Enables local-only demo accounts |
| `ENGRAMIC_ALLOWED_ORIGINS` | Comma-separated origins allowed for credentialed requests and writes |
| `ENGRAMIC_USERS_DB` | SQLite auth/assignment database path |
| `ENGRAMIC_GRAPH_FILE` | Clinical graph persistence path |
| `ENGRAMIC_MAX_UPLOAD_MB` | PDF upload limit |
| `ENGRAMIC_MAX_PDF_PAGES` | PDF page limit |
| `ENGRAMIC_MIN_PDF_TEXT_CHARS` | OCR fallback threshold |
| `TESSERACT_CMD` | Optional Tesseract executable path |
| `TESSERACT_LANG` | OCR language set, default `eng+ind` |

## Tests

Run the focused authentication, isolation, extraction, PDF, merge, conflict, and streaming-summary tests:

```powershell
cd C:\Users\Kiers\OneDrive\Documents\research\engramic-app
.\.venv\Scripts\python.exe -m unittest -v tests.test_app
```

## Safety boundaries

- Regex PII scrubbing is a demo safeguard, not proof of UU PDP compliance.
- Fuzzy scores from 85–94 require clinician confirmation; uncertain items are not silently merged.
- Contradictory or temporally ambiguous facts are flagged, never resolved by AI.
- The referral summary is generated from verified graph data and retains its verification disclaimer.
- Consent workflows, deletion/correction workflows, FHIR integration, and production terminology services remain outside this prototype.
- This application is not for emergency use.
