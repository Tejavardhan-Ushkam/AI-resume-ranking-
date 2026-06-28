import json
import uuid
import sqlite3
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone, timezone

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import settings
from models.schemas import CandidateCreate, CandidateInDB


# ─── SQLite Setup ────────────────────────────────────────────────────────────

DB_PATH = Path("./candidate_store.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            data TEXT NOT NULL,  -- full JSON blob
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS job_descriptions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            data TEXT NOT NULL,  -- full JSON blob with decomposition
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_candidates_email ON candidates(email);
        CREATE INDEX IF NOT EXISTS idx_jd_created ON job_descriptions(created_at);
    """)
    conn.commit()
    conn.close()
    logger.info("SQLite DB initialized")


# ─── ChromaDB Setup ──────────────────────────────────────────────────────────

_chroma_client = None
_chroma_collection = None


def get_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
        )
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready: {_chroma_collection.count()} vectors")
    return _chroma_collection


# ─── Candidate CRUD ──────────────────────────────────────────────────────────

def save_candidate(candidate: CandidateInDB) -> CandidateInDB:
    conn = get_db_connection()
    conn.execute(
        "INSERT OR REPLACE INTO candidates (id, name, email, data, created_at) VALUES (?, ?, ?, ?, ?)",
        (candidate.id, candidate.name, candidate.email,
         candidate.model_dump_json(), candidate.created_at.isoformat())
    )
    conn.commit()
    conn.close()
    return candidate


def get_all_candidates() -> List[CandidateInDB]:
    conn = get_db_connection()
    rows = conn.execute("SELECT data FROM candidates").fetchall()
    conn.close()
    return [CandidateInDB(**json.loads(row["data"])) for row in rows]


def get_candidate_by_id(candidate_id: str) -> Optional[CandidateInDB]:
    conn = get_db_connection()
    row = conn.execute("SELECT data FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    conn.close()
    return CandidateInDB(**json.loads(row["data"])) if row else None


def get_candidates_by_ids(ids: List[str]) -> List[CandidateInDB]:
    if not ids:
        return []
    conn = get_db_connection()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT data FROM candidates WHERE id IN ({placeholders})", ids
    ).fetchall()
    conn.close()
    return [CandidateInDB(**json.loads(row["data"])) for row in rows]


def count_candidates() -> int:
    conn = get_db_connection()
    count = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    conn.close()
    return count


# ─── JD CRUD ─────────────────────────────────────────────────────────────────

def save_job_description(jd_data: dict) -> str:
    jd_id = str(uuid.uuid4())
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO job_descriptions (id, title, data, created_at) VALUES (?, ?, ?, ?)",
        (jd_id, jd_data.get("title", "Untitled"), json.dumps(jd_data), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()
    return jd_id