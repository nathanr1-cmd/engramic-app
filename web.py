from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from auth import SESSION_COOKIE, authenticate, authorize_patient_access, authorized_patient_ids, create_session, create_user, get_current_user, grant_patient_access, require_role
from graph_engine import STORE, duplicate_candidates, get_full_graph, resolve_conflict
from schemas import Allergy, Condition, Facility, LabResult, Medication, Patient
import clinical_db

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


def _redirect_for(role: str) -> str:
    return "/doctor/dashboard" if role == "doctor" else "/patient/dashboard"


def _context(request: Request, **values):
    return {"request": request, "user": get_current_user(request), **values}


def _render(request: Request, name: str, status_code: int = 200, **values):
    return templates.TemplateResponse(request=request, name=name, context=_context(request, **values), status_code=status_code)


def _all_patients(user=None, *, archived: bool = False) -> list[Patient]:
    allowed = authorized_patient_ids(user) if user else set()
    return sorted(
        (node for node in STORE.nodes.values()
         if isinstance(node, Patient) and node.id in allowed and (node.archived_at is not None) == archived),
        key=lambda p: p.name,
    )


def _patient_data(patient_id: str) -> dict:
    graph = get_full_graph(patient_id)
    patient = next((n for n in graph.nodes if isinstance(n, Patient)), None)
    medications = [n for n in graph.nodes if isinstance(n, Medication)]
    conditions = [n for n in graph.nodes if isinstance(n, Condition)]
    labs = sorted((n for n in graph.nodes if isinstance(n, LabResult)), key=lambda n: n.date or date.min, reverse=True)
    allergies = [n for n in graph.nodes if isinstance(n, Allergy)]
    facilities = [n for n in graph.nodes if isinstance(n, Facility)]
    records = sorted(
        [record for node in graph.nodes for record in node.provenance_records],
        key=lambda record: record.encounter_date or date.min,
        reverse=True,
    )
    timeline_events = sorted(
        [
            {
                "record": record,
                "node": node,
                "label": getattr(node, "name", getattr(node, "test_name", getattr(node, "substance", node.id))),
                "node_type": node.node_type,
            }
            for node in graph.nodes
            for record in node.provenance_records
        ],
        key=lambda event: event["record"].encounter_date or date.min,
        reverse=True,
    )
    conflicts = [node for node in graph.nodes if getattr(node, "conflict_flag", False)]
    active_medications = [f"{item.name} {item.dose or ''}".strip() for item in medications if item.status.value == "active"]
    stopped_medications = [f"{item.name} {item.dose or ''}".strip() for item in medications if item.status.value == "discontinued"]
    active_allergies = [item.substance for item in allergies if item.status == "active"]
    patient_summary = " ".join([
        f"Obat Anda saat ini: {', '.join(active_medications) if active_medications else 'belum tercatat'}.",
        f"Obat yang sudah berhenti: {', '.join(stopped_medications) if stopped_medications else 'tidak ada yang tercatat'}.",
        f"Alergi yang tercatat: {', '.join(active_allergies) if active_allergies else 'tidak ada'}.",
    ])
    return {"patient": patient, "medications": medications, "conditions": conditions, "labs": labs,
            "allergies": allergies, "facilities": facilities, "records": records,
            "timeline_events": timeline_events, "conflicts": conflicts, "patient_summary": patient_summary,
            "audits": clinical_db.audits_for(patient_id),
            "patient_conflicts": clinical_db.conflicts(patient_id=patient_id)}


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return _render(request, "home.html")


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    user = get_current_user(request)
    if user:
        return RedirectResponse(_redirect_for(user.role), status_code=303)
    return _render(request, "auth.html", mode="login", error=None)


@router.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = authenticate(email, password)
    if not user:
        return _render(request, "auth.html", status_code=400, mode="login", error="Email or password is incorrect.")
    response = RedirectResponse(_redirect_for(user.role), status_code=303)
    response.set_cookie(SESSION_COOKIE, create_session(user), httponly=True, samesite="lax", secure=os.environ.get("ENGRAMIC_SECURE_COOKIES") == "true", max_age=604800)
    return response


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return _render(request, "auth.html", mode="register", error=None)


@router.post("/register")
def register(request: Request, name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    error = None
    role = "patient"
    if len(password) < 8:
        error = "Use at least 8 characters for your password."
    if error:
        return _render(request, "auth.html", status_code=400, mode="register", error=error)
    patient_id = None
    if role == "patient":
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "patient"
        patient_id = f"patient_{slug}_{uuid4().hex[:5]}"
    try:
        user = create_user(name, email, password, role, patient_id)
    except ValueError as exc:
        return _render(request, "auth.html", status_code=400, mode="register", error=str(exc))
    if role == "patient" and patient_id:
        STORE.add_patient(Patient(id=patient_id, name=name))
    response = RedirectResponse(_redirect_for(user.role), status_code=303)
    response.set_cookie(SESSION_COOKIE, create_session(user), httponly=True, samesite="lax", secure=os.environ.get("ENGRAMIC_SECURE_COOKIES") == "true", max_age=604800)
    return response


@router.post("/logout")
def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@router.get("/doctor/dashboard", response_class=HTMLResponse)
def doctor_dashboard(request: Request):
    user = require_role(request, "doctor")
    patients = _all_patients(user)
    patient_graphs = [get_full_graph(patient.id) for patient in patients]
    allowed_patient_ids = {patient.id for patient in patients}
    conflicts = sum(1 for item in clinical_db.conflicts(state="open") if item["patient_id"] in allowed_patient_ids)
    active_medications = sum(1 for graph in patient_graphs for node in graph.nodes if isinstance(node, Medication) and node.status.value == "active")
    recent_activity = sorted(
        [
            {"patient": patient, "record": record, "label": getattr(node, "name", getattr(node, "test_name", getattr(node, "substance", node.id)))}
            for patient, graph in zip(patients, patient_graphs)
            for node in graph.nodes
            for record in node.provenance_records
        ],
        key=lambda event: event["record"].reviewed_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )[:5]
    return _render(request, "doctor_dashboard.html", user=user, patients=patients, conflicts=conflicts, active_medications=active_medications, recent_activity=recent_activity)


@router.get("/doctor/patients", response_class=HTMLResponse)
def doctor_patients(request: Request, q: str = "", view: str = "active"):
    user = require_role(request, "doctor")
    archived = view == "archived"
    patients = [p for p in _all_patients(user, archived=archived) if q.lower() in p.name.lower() or q.lower() in p.id.lower()]
    return _render(request, "patients.html", user=user, patients=patients, query=q,
                   view="archived" if archived else "active",
                   active_count=len(_all_patients(user)), archived_count=len(_all_patients(user, archived=True)))


def _patient_form_values(form) -> dict:
    dob = form.get("date_of_birth") or None
    return {
        "name": (form.get("name") or "").strip(),
        "identifier": (form.get("identifier") or "").strip() or None,
        "date_of_birth": date.fromisoformat(dob) if dob else None,
        "age": int(form["age"]) if form.get("age") else None,
        "gender": form.get("gender") or None,
        "phone": (form.get("phone") or "").strip() or None,
        "address": (form.get("address") or "").strip() or None,
        "emergency_contact_name": (form.get("emergency_contact_name") or "").strip() or None,
        "emergency_contact_relationship": (form.get("emergency_contact_relationship") or "").strip() or None,
        "emergency_contact_phone": (form.get("emergency_contact_phone") or "").strip() or None,
        "assigned_facility_id": form.get("assigned_facility_id") or None,
    }


@router.get("/doctor/patients/new", response_class=HTMLResponse)
def doctor_patient_new(request: Request):
    user = require_role(request, "doctor")
    facilities = [node for node in STORE.nodes.values() if isinstance(node, Facility)]
    return _render(request, "patient_form.html", user=user, patient=None, facilities=facilities,
                   errors=[], duplicates=[], editing=False)


@router.post("/doctor/patients/new", response_class=HTMLResponse)
async def doctor_patient_create(request: Request):
    user = require_role(request, "doctor")
    form = await request.form()
    facilities = [node for node in STORE.nodes.values() if isinstance(node, Facility)]
    try:
        values = _patient_form_values(form)
    except (ValueError, TypeError):
        return _render(request, "patient_form.html", status_code=422, user=user, patient=None,
                       facilities=facilities, errors=["Date of birth or age is invalid."],
                       duplicates=[], editing=False)
    errors = []
    record_id = (form.get("id") or "").strip()
    if not values["name"]:
        errors.append("Full name is required.")
    if not record_id or not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", record_id):
        errors.append("Record ID must use 3–80 letters, numbers, underscores, or hyphens.")
    if record_id in STORE.nodes:
        errors.append("That record ID is already in use.")
    duplicates = duplicate_candidates(values)
    exact = any(item["exact_identifier"] for item in duplicates)
    create_anyway = form.get("create_anyway") == "yes"
    if exact:
        errors.append("That patient identifier is already in use and cannot be overridden.")
    draft = Patient(id=record_id or "draft", **values)
    if errors or (duplicates and not create_anyway):
        return _render(request, "patient_form.html", status_code=409 if duplicates else 422,
                       user=user, patient=draft, facilities=facilities, errors=errors,
                       duplicates=duplicates, editing=False)
    patient = STORE.add_patient(Patient(id=record_id, **values))
    grant_patient_access(user.id, patient.id)
    clinical_db.audit("patient_created", patient.id, user.email, after=patient.model_dump(mode="json"))
    if duplicates:
        clinical_db.audit("duplicate_warning_overridden", patient.id, user.email,
                          after={"matches": [item["patient"].id for item in duplicates]}, reason="create_anyway")
    return RedirectResponse(f"/doctor/patients/{patient.id}", status_code=303)


@router.get("/doctor/patients/{patient_id}/edit", response_class=HTMLResponse)
def doctor_patient_edit(request: Request, patient_id: str):
    user = require_role(request, "doctor")
    authorize_patient_access(request, patient_id)
    patient = STORE.nodes.get(patient_id)
    if not isinstance(patient, Patient):
        return RedirectResponse("/doctor/patients", status_code=303)
    facilities = [node for node in STORE.nodes.values() if isinstance(node, Facility)]
    return _render(request, "patient_form.html", user=user, patient=patient, facilities=facilities,
                   errors=[], duplicates=[], editing=True)


@router.post("/doctor/patients/{patient_id}/edit", response_class=HTMLResponse)
async def doctor_patient_update(request: Request, patient_id: str):
    user = require_role(request, "doctor")
    authorize_patient_access(request, patient_id)
    patient = STORE.nodes.get(patient_id)
    if not isinstance(patient, Patient):
        return RedirectResponse("/doctor/patients", status_code=303)
    form = await request.form()
    facilities = [node for node in STORE.nodes.values() if isinstance(node, Facility)]
    try:
        values = _patient_form_values(form)
    except (ValueError, TypeError):
        return _render(request, "patient_form.html", status_code=422, user=user, patient=patient,
                       facilities=facilities, errors=["Date of birth or age is invalid."],
                       duplicates=[], editing=True)
    errors = [] if values["name"] else ["Full name is required."]
    duplicates = duplicate_candidates(values, patient_id)
    if any(item["exact_identifier"] for item in duplicates):
        errors.append("That patient identifier is already in use.")
    if errors:
        return _render(request, "patient_form.html", status_code=409, user=user, patient=patient,
                       facilities=facilities, errors=errors, duplicates=duplicates, editing=True)
    before = patient.model_dump(mode="json")
    updated = STORE.update_patient(patient_id, values)
    clinical_db.audit("patient_edited", patient_id, user.email, before=before,
                      after=updated.model_dump(mode="json"))
    return RedirectResponse(f"/doctor/patients/{patient_id}", status_code=303)


@router.post("/doctor/patients/{patient_id}/archive")
def doctor_patient_archive(request: Request, patient_id: str):
    user = require_role(request, "doctor")
    authorize_patient_access(request, patient_id)
    patient = STORE.nodes.get(patient_id)
    if not isinstance(patient, Patient):
        return RedirectResponse("/doctor/patients", status_code=303)
    if patient.archived_at is None:
        before = patient.model_dump(mode="json")
        updated = STORE.update_patient(patient_id, {
            "archived_at": datetime.now(timezone.utc),
            "archived_by": user.email,
        })
        clinical_db.audit("patient_archived", patient_id, user.email, before=before,
                          after=updated.model_dump(mode="json"))
    return RedirectResponse("/doctor/patients?view=archived", status_code=303)


@router.post("/doctor/patients/{patient_id}/restore")
def doctor_patient_restore(request: Request, patient_id: str):
    user = require_role(request, "doctor")
    authorize_patient_access(request, patient_id)
    patient = STORE.nodes.get(patient_id)
    if not isinstance(patient, Patient):
        return RedirectResponse("/doctor/patients?view=archived", status_code=303)
    if patient.archived_at is not None:
        before = patient.model_dump(mode="json")
        updated = STORE.update_patient(patient_id, {"archived_at": None, "archived_by": None})
        clinical_db.audit("patient_restored", patient_id, user.email, before=before,
                          after=updated.model_dump(mode="json"))
    return RedirectResponse(f"/doctor/patients/{patient_id}", status_code=303)


@router.get("/doctor/conflicts", response_class=HTMLResponse)
def doctor_conflicts(request: Request, state: str = "open", patient_id: str = "", entity_type: str = ""):
    user = require_role(request, "doctor")
    allowed = authorized_patient_ids(user)
    selected_patient = patient_id if patient_id in allowed else ""
    valid_state = state if state in {"open", "resolved", "uncertain", ""} else "open"
    items = [item for item in clinical_db.conflicts(valid_state, selected_patient, entity_type)
             if item["patient_id"] in allowed]
    return _render(request, "conflicts.html", user=user, conflicts=items, patients=_all_patients(user),
                   state=valid_state, selected_patient=selected_patient, entity_type=entity_type)


@router.get("/doctor/conflicts/{conflict_id}", response_class=HTMLResponse)
def doctor_conflict_detail(request: Request, conflict_id: str):
    user = require_role(request, "doctor")
    item = clinical_db.conflict(conflict_id)
    if not item:
        return RedirectResponse("/doctor/conflicts", status_code=303)
    authorize_patient_access(request, item["patient_id"])
    return _render(request, "conflict_detail.html", user=user, conflict=item,
                   patient=STORE.nodes.get(item["patient_id"]))


@router.post("/doctor/conflicts/{conflict_id}")
async def doctor_conflict_resolve(request: Request, conflict_id: str):
    user = require_role(request, "doctor")
    item = clinical_db.conflict(conflict_id)
    if not item:
        return RedirectResponse("/doctor/conflicts", status_code=303)
    authorize_patient_access(request, item["patient_id"])
    form = await request.form()
    try:
        resolve_conflict(conflict_id, form.get("action"), user.email, form.get("comment") or None)
    except (KeyError, ValueError):
        return RedirectResponse(f"/doctor/conflicts/{conflict_id}", status_code=303)
    return RedirectResponse("/doctor/conflicts", status_code=303)


@router.get("/doctor/patients/{patient_id}", response_class=HTMLResponse)
def doctor_patient_detail(request: Request, patient_id: str):
    user = require_role(request, "doctor")
    authorize_patient_access(request, patient_id)
    return _render(request, "patient_detail.html", user=user, **_patient_data(patient_id))


@router.get("/doctor/notes/new", response_class=HTMLResponse)
def doctor_note_new(request: Request, patient_id: str = ""):
    user = require_role(request, "doctor")
    return _render(request, "note_new.html", user=user, patients=_all_patients(user), selected_patient=patient_id)


@router.get("/doctor/referral-summary", response_class=HTMLResponse)
def doctor_referral(request: Request, patient_id: str = ""):
    user = require_role(request, "doctor")
    return _render(request, "referral.html", user=user, patients=_all_patients(user), selected_patient=patient_id)


def _patient_page(request: Request, template: str, title: str):
    user = require_role(request, "patient")
    return _render(request, template, user=user, title=title, **_patient_data(user.patient_id))


@router.get("/patient/dashboard", response_class=HTMLResponse)
def patient_dashboard(request: Request): return _patient_page(request, "patient_dashboard.html", "Your health overview")


@router.get("/patient/medications", response_class=HTMLResponse)
def patient_medications(request: Request): return _patient_page(request, "patient_section.html", "Medications")


@router.get("/patient/conditions", response_class=HTMLResponse)
def patient_conditions(request: Request): return _patient_page(request, "patient_section.html", "Conditions")


@router.get("/patient/history", response_class=HTMLResponse)
def patient_history(request: Request): return _patient_page(request, "patient_section.html", "Health history")
