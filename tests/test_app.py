from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

RUNTIME = Path(tempfile.mkdtemp(prefix="engramic-tests-"))
os.environ["ENGRAMIC_USERS_DB"] = str(RUNTIME / "users.db")
os.environ["ENGRAMIC_GRAPH_FILE"] = str(RUNTIME / "graph.json")
os.environ["ENGRAMIC_CLINICAL_DB"] = str(RUNTIME / "clinical.db")
os.environ["ENGRAMIC_DOCUMENT_DIR"] = str(RUNTIME / "documents")
os.environ["ENGRAMIC_FORCE_CACHE"] = "true"
os.environ["ENGRAMIC_SESSION_SECRET"] = "test-only-session-secret"

from fastapi.testclient import TestClient

from app import app
from auth import create_user
import clinical_db


DOCTOR_LOGIN = {"email": "doctor@engramic.id", "password": "Doctor123!"}


class EngramicFlowTests(unittest.TestCase):
    def doctor_client(self) -> TestClient:
        client = TestClient(app)
        response = client.post("/login", data=DOCTOR_LOGIN, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/doctor/dashboard")
        return client

    def create_patient(self, client: TestClient, patient_id: str) -> None:
        response = client.post(
            "/patients",
            json={"id": patient_id, "name": patient_id.replace("patient_", "Test ").title(), "age": 50, "gender": "male"},
        )
        self.assertEqual(response.status_code, 200, response.text)

    @staticmethod
    def active_medication(name: str = "Amlodipine") -> dict:
        return {
            "entity_type": "medication", "name": name, "dose": "5 mg",
            "frequency": "once daily", "route": "oral", "reason": "Hypertension",
            "prescriber": None, "status": "active", "start_date": "2026-07-14",
            "end_date": None, "negated": False, "discontinued": False,
            "explicit_state_change": True, "raw_text_span": f"Start {name} 5 mg",
            "source_page": None,
        }

    def test_role_redirects_and_public_registration_is_patient_only(self):
        unauthenticated = TestClient(app)
        self.assertEqual(
            unauthenticated.post("/notes/extract", json={"patient_id": "patient_budi", "note_id": "x", "raw_text": "test"}).status_code,
            401,
        )
        doctor = TestClient(app)
        response = doctor.post("/login", data=DOCTOR_LOGIN, follow_redirects=False)
        self.assertEqual(response.headers["location"], "/doctor/dashboard")
        patient = TestClient(app)
        response = patient.post(
            "/register",
            data={"name": "Isolation Patient", "email": "isolation@example.test", "password": "SafePass123!", "role": "doctor"},
            follow_redirects=False,
        )
        self.assertEqual(response.headers["location"], "/patient/dashboard")
        self.assertEqual(patient.get("/patients/patient_budi/graph").status_code, 403)
        self.assertEqual(
            patient.post("/notes/extract", json={"patient_id": "patient_budi", "note_id": "x", "raw_text": "test"}).status_code,
            403,
        )
        wrong_role = patient.get("/doctor/dashboard", follow_redirects=False)
        self.assertEqual(wrong_role.status_code, 303)
        self.assertEqual(wrong_role.headers["location"], "/patient/dashboard")

    def test_unassigned_doctor_cannot_access_patient(self):
        create_user("Dr. Unassigned", "unassigned@example.test", "SafePass123!", "doctor", None)
        client = TestClient(app)
        client.post("/login", data={"email": "unassigned@example.test", "password": "SafePass123!"})
        self.assertEqual(client.get("/patients/patient_budi/graph").status_code, 403)
        self.assertEqual(client.get("/doctor/patients/patient_budi").status_code, 403)

    def test_typed_extraction_requires_review_before_merge(self):
        client = self.doctor_client()
        patient_id = "patient_typed_flow"
        self.create_patient(client, patient_id)
        note = {
            "patient_id": patient_id, "patient_name": "Test Typed Flow", "note_id": "note_1",
            "raw_text": "Didiagnosis hipertensi. Mulai Amlodipin 5 mg sekali sehari.",
            "facility_id": "facility_puskesmas", "encounter_date": "2026-07-14",
        }
        extraction = client.post("/notes/extract", json=note)
        self.assertEqual(extraction.status_code, 200, extraction.text)
        before = client.get(f"/patients/{patient_id}/graph").json()
        self.assertEqual([node["node_type"] for node in before["nodes"]], ["patient"])
        merge = client.post(
            "/notes/merge",
            json={
                "patient_id": patient_id, "note_id": "note_1", "encounter_date": "2026-07-14",
                "entities": extraction.json()["entities"], "reviewed_by": "spoofed@example.test",
                "extraction_method": "cache",
            },
        )
        self.assertEqual(merge.status_code, 200, merge.text)
        graph = client.get(f"/patients/{patient_id}/graph").json()
        medication = next(node for node in graph["nodes"] if node["node_type"] == "medication")
        self.assertEqual(medication["provenance_records"][0]["reviewed_by"], DOCTOR_LOGIN["email"])

    def test_pdf_extraction_returns_review_metadata_without_mutation(self):
        import fitz

        client = self.doctor_client()
        patient_id = "patient_pdf_flow"
        self.create_patient(client, patient_id)
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Pak Budi didiagnosis hipertensi. Mulai Amlodipin 5 mg sekali sehari oral.")
        payload = pdf.tobytes()
        pdf.close()
        response = client.post(
            "/documents/extract",
            data={"patient_id": patient_id, "note_id": "note_1", "patient_name": "Test Pdf Flow"},
            files={"document": ("clinical-note.pdf", payload, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["document_id"].startswith("doc_"))
        self.assertEqual(body["pages"][0]["method"], "embedded_text")
        self.assertTrue(body["extraction"]["entities"])
        self.assertTrue(any(entity["source_page"] == 1 for entity in body["extraction"]["entities"]))
        self.assertEqual(client.get(f"/patients/{patient_id}/graph").json()["nodes"][0]["node_type"], "patient")

    def test_condition_name_link_creates_cohesive_medication_edge(self):
        client = self.doctor_client()
        patient_id = "patient_link_flow"
        self.create_patient(client, patient_id)
        condition = {
            "entity_type": "condition", "name": "Dyslipidemia", "status": "active",
            "date": "2026-07-20", "negated": False, "discontinued": False,
            "explicit_state_change": True, "raw_text_span": "Dyslipidemia", "source_page": None,
        }
        medication = self.active_medication("Atorvastatin")
        medication.update({"dose": "20 mg", "reason": "Dyslipidemia", "raw_text_span": "Start Atorvastatin 20 mg"})
        response = client.post(
            "/notes/merge",
            json={
                "patient_id": patient_id, "note_id": "linked_note", "encounter_date": "2026-07-20",
                "entities": [condition, medication], "condition_links": {"Atorvastatin": "Dyslipidemia"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        graph = client.get(f"/patients/{patient_id}/graph").json()
        condition_node = next(node for node in graph["nodes"] if node["node_type"] == "condition")
        medication_node = next(node for node in graph["nodes"] if node["node_type"] == "medication")
        self.assertTrue(any(edge["source"] == condition_node["id"] and edge["target"] == medication_node["id"] and edge["relation"] == "TREATED_BY" for edge in graph["edges"]))

    def test_all_required_pages_render_for_their_roles(self):
        doctor = self.doctor_client()
        for path in ["/doctor/dashboard", "/doctor/patients", "/doctor/patients/patient_budi", "/doctor/notes/new", "/doctor/referral-summary"]:
            self.assertEqual(doctor.get(path).status_code, 200, path)
        patient = TestClient(app)
        patient.post("/login", data={"email": "patient@engramic.id", "password": "Patient123!"})
        for path in ["/patient/dashboard", "/patient/medications", "/patient/conditions", "/patient/history"]:
            self.assertEqual(patient.get(path).status_code, 200, path)

    def test_conflict_is_returned_and_summary_keeps_disclaimer(self):
        client = self.doctor_client()
        patient_id = "patient_conflict_flow"
        self.create_patient(client, patient_id)
        condition = {
            "entity_type": "condition", "name": "Hypertension", "status": "active",
            "date": "2026-07-14", "negated": False, "discontinued": False,
            "explicit_state_change": True, "raw_text_span": "Hypertension", "source_page": None,
        }
        first = client.post(
            "/notes/merge",
            json={"patient_id": patient_id, "note_id": "conflict_1", "encounter_date": "2026-07-14", "entities": [condition, self.active_medication()]},
        )
        self.assertEqual(first.status_code, 200, first.text)
        stopped = self.active_medication()
        stopped.update({"status": "discontinued", "discontinued": True, "start_date": None, "end_date": None, "raw_text_span": "Amlodipine stopped"})
        second = client.post(
            "/notes/merge",
            json={"patient_id": patient_id, "note_id": "conflict_2", "entities": [stopped]},
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(len(second.json()["conflicts"]), 1)
        summary = client.get(f"/patients/{patient_id}/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertIn("Harap verifikasi", summary.text)

    def test_v2_review_decisions_are_persisted_without_merging_excluded_fact(self):
        client = self.doctor_client()
        patient_id = "patient_review_audit"
        self.create_patient(client, patient_id)
        original = {"entity_type": "condition", "name": "Excluded condition", "status": "active", "raw_text_span": "source text"}
        response = client.post("/notes/merge", json={
            "patient_id": patient_id, "note_id": "review_audit", "entities": [],
            "review_decisions": [{"fact_index": 0, "decision": "exclude", "original": original, "reviewed": None}],
        })
        self.assertEqual(response.status_code, 200, response.text)
        graph = client.get(f"/patients/{patient_id}/graph").json()
        self.assertFalse(any(node.get("name") == "Excluded Condition" for node in graph["nodes"]))
        self.assertTrue(any(row["event_type"] == "extracted_fact_excluded" for row in clinical_db.audits_for(patient_id)))

    def test_v2_patient_profile_duplicate_detection_and_pages_render(self):
        client = self.doctor_client()
        for path in ["/doctor/patients/new", "/doctor/conflicts"]:
            self.assertEqual(client.get(path).status_code, 200, path)
        first = client.post("/patients", json={"id": "patient_profile_one", "name": "Ari Pratama", "identifier": "NIK-TEST-001"})
        self.assertEqual(first.status_code, 200, first.text)
        duplicate = client.post("/patients", json={"id": "patient_profile_two", "name": "Ari P.", "identifier": "niktest001"})
        self.assertEqual(duplicate.status_code, 409, duplicate.text)

    def test_v2_conflict_resolution_is_persisted(self):
        client = self.doctor_client()
        patient_id = "patient_conflict_resolution"
        self.create_patient(client, patient_id)
        active = self.active_medication()
        self.assertEqual(client.post("/notes/merge", json={"patient_id": patient_id, "note_id": "state_1", "encounter_date": "2026-07-14", "entities": [active]}).status_code, 200)
        stopped = self.active_medication()
        stopped.update({"status": "discontinued", "discontinued": True, "start_date": None, "end_date": None, "raw_text_span": "status unclear"})
        conflict = client.post("/notes/merge", json={"patient_id": patient_id, "note_id": "state_2", "entities": [stopped]}).json()["conflicts"][0]
        resolved = client.post(f"/conflicts/{conflict['id']}/resolve", json={"action": "mark_uncertain", "comment": "Insufficient dated evidence"})
        self.assertEqual(resolved.status_code, 200, resolved.text)
        self.assertEqual(clinical_db.conflict(conflict["id"])["state"], "uncertain")

    def test_patient_archive_is_reversible_and_audited(self):
        client = self.doctor_client()
        patient_id = "patient_archive_flow"
        self.create_patient(client, patient_id)
        self.assertIn(patient_id, client.get("/doctor/patients").text)

        archived = client.post(f"/doctor/patients/{patient_id}/archive", follow_redirects=False)
        self.assertEqual(archived.status_code, 303)
        self.assertEqual(archived.headers["location"], "/doctor/patients?view=archived")
        self.assertNotIn(patient_id, client.get("/doctor/patients").text)
        self.assertIn(patient_id, client.get("/doctor/patients?view=archived").text)
        self.assertTrue(any(row["event_type"] == "patient_archived" for row in clinical_db.audits_for(patient_id)))

        restored = client.post(f"/doctor/patients/{patient_id}/restore", follow_redirects=False)
        self.assertEqual(restored.status_code, 303)
        self.assertIn(patient_id, client.get("/doctor/patients").text)
        self.assertTrue(any(row["event_type"] == "patient_restored" for row in clinical_db.audits_for(patient_id)))


if __name__ == "__main__":
    unittest.main()
