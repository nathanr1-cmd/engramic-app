from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("ENGRAMIC_CLINICAL_DB", BASE_DIR / "clinical.db"))
GRAPH_FILE = Path(os.environ.get("ENGRAMIC_GRAPH_FILE", BASE_DIR / "graph_data.json"))
DOCUMENT_DIR = Path(os.environ.get("ENGRAMIC_DOCUMENT_DIR", BASE_DIR / "clinical_documents"))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH, factory=ClosingConnection)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    db = connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize() -> None:
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
          id TEXT PRIMARY KEY, patient_id TEXT, node_type TEXT NOT NULL,
          payload_json TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_patient ON graph_nodes(patient_id,node_type);
        CREATE TABLE IF NOT EXISTS graph_membership (
          patient_id TEXT NOT NULL, node_id TEXT NOT NULL, PRIMARY KEY(patient_id,node_id)
        );
        CREATE TABLE IF NOT EXISTS graph_edges (
          id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL,
          source TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL,
          payload_json TEXT NOT NULL, UNIQUE(patient_id,source,relation,target)
        );
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, note_id TEXT NOT NULL,
          filename TEXT NOT NULL, content_type TEXT NOT NULL, storage_name TEXT NOT NULL,
          size_bytes INTEGER NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS document_pages (
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          page INTEGER NOT NULL, method TEXT NOT NULL, text TEXT NOT NULL,
          confidence REAL, warnings_json TEXT NOT NULL, PRIMARY KEY(document_id,page)
        );
        CREATE TABLE IF NOT EXISTS conflicts (
          id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, node_id TEXT NOT NULL,
          entity_type TEXT NOT NULL, entity_name TEXT NOT NULL, field TEXT NOT NULL,
          existing_value TEXT NOT NULL, incoming_value TEXT NOT NULL,
          existing_note_id TEXT, incoming_note_id TEXT, existing_date TEXT, incoming_date TEXT,
          existing_evidence TEXT, incoming_evidence TEXT, reason TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'open' CHECK(state IN ('open','resolved','uncertain')),
          resolution_action TEXT, resolved_by TEXT, resolved_at TEXT, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conflicts_state ON conflicts(state,patient_id,entity_type);
        CREATE TABLE IF NOT EXISTS review_decisions (
          id TEXT PRIMARY KEY, review_id TEXT NOT NULL, patient_id TEXT NOT NULL,
          note_id TEXT NOT NULL, document_id TEXT, fact_index INTEGER NOT NULL,
          decision TEXT NOT NULL CHECK(decision IN ('include','exclude','reject')),
          original_json TEXT, reviewed_json TEXT, actor TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_events (
          id TEXT PRIMARY KEY, event_type TEXT NOT NULL, patient_id TEXT NOT NULL,
          entity_id TEXT, conflict_id TEXT, note_id TEXT, document_id TEXT,
          actor TEXT NOT NULL, created_at TEXT NOT NULL, before_json TEXT,
          after_json TEXT, reason TEXT, correlation_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_events(patient_id,created_at DESC);
        CREATE TABLE IF NOT EXISTS migrations (name TEXT PRIMARY KEY, completed_at TEXT NOT NULL);
        """)


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, default=str)


def audit(event_type: str, patient_id: str, actor: str, *, entity_id: str | None = None,
          conflict_id: str | None = None, note_id: str | None = None,
          document_id: str | None = None, before: Any = None, after: Any = None,
          reason: str | None = None, correlation_id: str | None = None,
          db: sqlite3.Connection | None = None) -> str:
    event_id = str(uuid4())
    values = (event_id, event_type, patient_id, entity_id, conflict_id, note_id,
              document_id, actor, utcnow(), _json(before), _json(after), reason, correlation_id)
    sql = "INSERT INTO audit_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
    if db is not None:
        db.execute(sql, values)
    else:
        with connect() as conn:
            conn.execute(sql, values)
    return event_id


def load_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]]] | None:
    initialize()
    with connect() as db:
        rows = db.execute("SELECT id,patient_id,payload_json FROM graph_nodes").fetchall()
        if not rows:
            return None
        nodes = [json.loads(row["payload_json"]) for row in rows]
        edges = [json.loads(row["payload_json"]) for row in db.execute("SELECT payload_json FROM graph_edges")]
        patient_nodes: dict[str, set[str]] = {}
        for row in db.execute("SELECT patient_id,node_id FROM graph_membership"):
            patient_nodes.setdefault(row["patient_id"], set()).add(row["node_id"])
        if not patient_nodes:
            for row in rows:
                if row["patient_id"]:
                    patient_nodes.setdefault(row["patient_id"], set()).add(row["id"])
        return nodes, edges, patient_nodes


def save_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
               patient_nodes: dict[str, set[str]], db: sqlite3.Connection | None = None) -> None:
    def perform(conn: sqlite3.Connection) -> None:
        membership = {node_id: patient_id for patient_id, ids in patient_nodes.items() for node_id in ids}
        conn.execute("DELETE FROM graph_edges")
        conn.execute("DELETE FROM graph_membership")
        conn.execute("DELETE FROM graph_nodes")
        now = utcnow()
        for node in nodes:
            conn.execute("INSERT INTO graph_nodes VALUES (?,?,?,?,?)",
                         (node["id"], membership.get(node["id"]), node["node_type"], _json(node), now))
        for patient_id, ids in patient_nodes.items():
            for node_id in ids:
                conn.execute("INSERT INTO graph_membership VALUES (?,?)", (patient_id, node_id))
        for patient_id, ids in patient_nodes.items():
            for edge in edges:
                if edge["source"] in ids or edge["target"] in ids:
                    conn.execute("INSERT OR IGNORE INTO graph_edges(patient_id,source,relation,target,payload_json) VALUES (?,?,?,?,?)",
                                 (patient_id, edge["source"], edge["relation"], edge["target"], _json(edge)))
    if db is not None:
        perform(db)
    else:
        with transaction() as conn:
            perform(conn)


def import_json_once() -> dict[str, Any] | None:
    initialize()
    with connect() as db:
        if db.execute("SELECT 1 FROM migrations WHERE name='graph_json_v1'").fetchone():
            return None
        if not GRAPH_FILE.exists():
            db.execute("INSERT INTO migrations VALUES ('graph_json_v1',?)", (utcnow(),))
            return None
        data = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
        backup = GRAPH_FILE.with_suffix(".json.pre_sqlite_backup")
        if not backup.exists():
            shutil.copy2(GRAPH_FILE, backup)
        db.execute("INSERT INTO migrations VALUES ('graph_json_v1',?)", (utcnow(),))
        return data


def store_document(patient_id: str, note_id: str, filename: str, content_type: str,
                   payload: bytes, response: dict[str, Any]) -> str:
    document_id = response["document_id"]
    storage_name = f"{document_id}.pdf"
    (DOCUMENT_DIR / storage_name).write_bytes(payload)
    with transaction() as db:
        db.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                   (document_id, patient_id, note_id, filename, content_type, storage_name,
                    len(payload), _json({k: v for k, v in response.items() if k != "pages"}), utcnow()))
        db.execute("DELETE FROM document_pages WHERE document_id=?", (document_id,))
        for page in response.get("pages", []):
            db.execute("INSERT INTO document_pages VALUES (?,?,?,?,?,?)",
                       (document_id, page["page"], page["method"], page["text"],
                        page.get("ocr_confidence"), _json(page.get("warnings", []))))
    return document_id


def document(document_id: str, patient_id: str | None = None) -> sqlite3.Row | None:
    with connect() as db:
        query = "SELECT * FROM documents WHERE id=?" + (" AND patient_id=?" if patient_id else "")
        values = (document_id, patient_id) if patient_id else (document_id,)
        return db.execute(query, values).fetchone()


def document_pages(document_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM document_pages WHERE document_id=? ORDER BY page", (document_id,))]


def audits_for(patient_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM audit_events WHERE patient_id=? ORDER BY created_at DESC LIMIT ?", (patient_id, limit))]


def conflicts(state: str = "", patient_id: str = "", entity_type: str = "") -> list[dict[str, Any]]:
    clauses, values = [], []
    for column, value in (("state", state), ("patient_id", patient_id), ("entity_type", entity_type)):
        if value:
            clauses.append(f"{column}=?")
            values.append(value)
    sql = "SELECT * FROM conflicts" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY created_at DESC"
    with connect() as db:
        return [dict(row) for row in db.execute(sql, values)]


def conflict(conflict_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM conflicts WHERE id=?", (conflict_id,)).fetchone()
        return dict(row) if row else None
