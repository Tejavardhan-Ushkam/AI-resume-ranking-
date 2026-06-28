import time
import uuid
from fastapi import APIRouter, HTTPException
from loguru import logger

from models.schemas import SearchRequest, SearchResponse
from models.database import get_candidates_by_ids, save_job_description, count_candidates
from core.jd_analyzer import decompose_jd
from core.signal_engine import embed_text, semantic_search
from core.ranker import rank_candidates
from core.explainer import enrich_results_with_summaries

router = APIRouter(prefix="/search", tags=["Search & Ranking"])


@router.post("/", response_model=SearchResponse)
async def search_candidates(request: SearchRequest):
    """
    Core intelligent candidate discovery endpoint.
    
    Pipeline:
    1. Decompose JD into multi-axis structured representation
    2. Embed enriched JD description
    3. ANN search in ChromaDB for semantic candidates
    4. Ensemble re-ranking (semantic + skills + Career DNA + behavioral)
    5. Enrich top results with AI explanations
    """
    start_time = time.time()
    
    total_candidates = count_candidates()
    if total_candidates == 0:
        raise HTTPException(
            status_code=400,
            detail="No candidates in system. Please upload candidates first via POST /candidates/bulk"
        )

    logger.info(f"Search request: '{request.job_title or 'Untitled'}' over {total_candidates} candidates")

    # ── Step 1: Deep JD Analysis ─────────────────────────────────────────────
    jd = await decompose_jd(request.job_description, request.job_title or "Role")
    logger.debug(f"JD decomposed: {jd.role_seniority} {jd.role_type}, {len(jd.required_skills)} req skills")

    # ── Step 2: Embed JD ─────────────────────────────────────────────────────
    jd_embedding = embed_text(jd.enriched_description)

    # ── Step 3: Semantic Search (ANN) ────────────────────────────────────────
    # Fetch 3x top_k for re-ranking headroom
    ann_results = semantic_search(
        jd_embedding,
        top_k=min(request.top_k * 3, total_candidates),
        filters=request.filters,
    )

    if not ann_results:
        raise HTTPException(status_code=404, detail="No candidates found matching filters")

    candidate_ids = [cid for cid, _ in ann_results]
    semantic_scores = {cid: score for cid, score in ann_results}

    # ── Step 4: Load Candidates & Ensemble Rank ───────────────────────────────
    candidates = get_candidates_by_ids(candidate_ids)
    ranked = rank_candidates(candidates, semantic_scores, jd)

    # Trim to requested top_k
    top_results = ranked[: request.top_k]

    # ── Step 5: AI Summaries for Top 10 ──────────────────────────────────────
    top_results = await enrich_results_with_summaries(
        top_results, jd, request.job_title or "Role", top_k=10
    )

    # ── Persist JD ───────────────────────────────────────────────────────────
    job_id = save_job_description({
        "title": request.job_title or "Role",
        "description": request.job_description,
        "decomposition": jd.model_dump(),
    })

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"Search complete: {len(top_results)} results in {elapsed_ms:.0f}ms")

    return SearchResponse(
        job_id=job_id,
        jd_decomposition=jd,
        results=top_results,
        total_candidates_evaluated=len(candidates),
        processing_time_ms=round(elapsed_ms, 1),
    )