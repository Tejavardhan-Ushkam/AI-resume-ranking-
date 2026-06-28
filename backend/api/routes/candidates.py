import uuid
from typing import List
from datetime import datetime, timezone, timezone

from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from models.schemas import CandidateCreate, CandidateInDB
from models.database import save_candidate, get_all_candidates, get_candidate_by_id, count_candidates
from core.career_dna import compute_career_dna
from core.signal_engine import candidate_to_text, embed_text, index_candidate_in_chroma

router = APIRouter(prefix="/candidates", tags=["Candidates"])


def _create_candidate_full(candidate_data: CandidateCreate) -> CandidateInDB:
    """Full candidate processing pipeline."""
    candidate_id = str(uuid.uuid4())
    
    # Compute Career DNA fingerprint
    career_dna = compute_career_dna(candidate_data)
    
    # Build candidate in DB model
    candidate_in_db = CandidateInDB(
        **candidate_data.model_dump(),
        id=candidate_id,
        career_dna=career_dna,
        created_at=datetime.now(timezone.utc),
    )
    
    # Embed and index in ChromaDB
    text_repr = candidate_to_text(candidate_in_db)
    embedding = embed_text(text_repr)
    candidate_in_db.embedding_id = candidate_id
    
    index_candidate_in_chroma(candidate_in_db, embedding)
    save_candidate(candidate_in_db)
    
    return candidate_in_db


@router.post("/", response_model=CandidateInDB, status_code=201)
async def create_candidate(candidate: CandidateCreate):
    """Add a new candidate to the system."""
    try:
        result = _create_candidate_full(candidate)
        logger.info(f"Candidate created: {result.name} ({result.id})")
        return result
    except Exception as e:
        logger.error(f"Failed to create candidate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk", response_model=dict, status_code=201)
async def create_candidates_bulk(candidates: List[CandidateCreate], background_tasks: BackgroundTasks):
    """Bulk upload candidates (async background indexing for large sets)."""
    if len(candidates) > 500:
        raise HTTPException(status_code=400, detail="Max 500 candidates per bulk upload")
    
    created = []
    errors = []
    
    for c in candidates:
        try:
            result = _create_candidate_full(c)
            created.append(result.id)
        except Exception as e:
            errors.append({"email": c.email, "error": str(e)})
    
    return {
        "created_count": len(created),
        "error_count": len(errors),
        "errors": errors[:10],  # return first 10 errors
    }


@router.get("/", response_model=List[CandidateInDB])
async def list_candidates(skip: int = 0, limit: int = 50):
    """List all candidates."""
    all_cands = get_all_candidates()
    return all_cands[skip : skip + limit]


@router.get("/count")
async def get_candidate_count():
    return {"count": count_candidates()}


@router.get("/{candidate_id}", response_model=CandidateInDB)
async def get_candidate(candidate_id: str):
    """Get a specific candidate by ID."""
    candidate = get_candidate_by_id(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate