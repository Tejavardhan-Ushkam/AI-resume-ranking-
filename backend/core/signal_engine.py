"""
Signal Engine: Computes and stores candidate embeddings.
Uses sentence-transformers locally (no API key needed) for fast, offline-capable embeddings.
"""
import numpy as np
from typing import List, Tuple
from loguru import logger

from models.schemas import CandidateInDB, JDDecomposition
from config import settings

# Lazy-loaded embedding model
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _embedding_model


def candidate_to_text(candidate: CandidateInDB) -> str:
    """
    Build a rich text representation of a candidate for embedding.
    Carefully weighted: recent roles weighted higher than old ones.
    """
    parts = [
        f"Name: {candidate.name}",
        f"Title: {candidate.current_title}",
        f"Experience: {candidate.years_of_experience} years",
        f"Summary: {candidate.summary}",
        f"Skills: {', '.join(candidate.skills)}",
    ]

    # Work experience - ordered by recency, most recent 3 get full text
    if candidate.work_experience:
        from datetime import datetime
        
        def exp_sort_key(e):
            if not e.end_date:
                return "9999-99"
            return e.end_date
        
        sorted_exp = sorted(candidate.work_experience, key=exp_sort_key, reverse=True)
        
        for i, exp in enumerate(sorted_exp[:5]):
            weight_prefix = "Recent: " if i < 2 else "Past: "
            parts.append(
                f"{weight_prefix}{exp.title} at {exp.company}. {exp.description[:200]}"
            )

    # Education
    if candidate.education:
        edu = candidate.education[0]
        parts.append(f"Education: {edu.degree} in {edu.field} from {edu.institution}")

    return " | ".join(parts)


def embed_text(text: str) -> List[float]:
    """Embed a single text string."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts efficiently."""
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    return embeddings.tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two normalized vectors (dot product since normalized)."""
    a_np = np.array(a)
    b_np = np.array(b)
    # Already normalized, so cosine sim = dot product
    return float(np.dot(a_np, b_np))


def index_candidate_in_chroma(candidate: CandidateInDB, embedding: List[float]):
    """Store candidate embedding in ChromaDB for fast ANN retrieval."""
    from models.database import get_chroma
    collection = get_chroma()
    
    # Metadata for pre-filtering
    metadata = {
        "name": candidate.name,
        "years_experience": candidate.years_of_experience,
        "skills_count": len(candidate.skills),
        "current_title": candidate.current_title,
        "location": candidate.location,
        "skills_str": "|".join(s.lower() for s in candidate.skills[:20]),
    }
    
    collection.upsert(
        ids=[candidate.id],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def semantic_search(
    jd_embedding: List[float],
    top_k: int = 20,
    filters: dict = None,
) -> List[Tuple[str, float]]:
    """
    ANN search in ChromaDB. Returns (candidate_id, distance) pairs.
    ChromaDB cosine distance: 0 = identical, 2 = opposite.
    Convert to similarity: sim = 1 - distance/2
    """
    from models.database import get_chroma
    collection = get_chroma()
    
    if collection.count() == 0:
        return []

    where_clause = None
    if filters:
        where_clause = _build_chroma_filter(filters)

    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=min(top_k, collection.count()),
        where=where_clause,
        include=["distances", "metadatas"],
    )

    pairs = []
    for cid, dist in zip(results["ids"][0], results["distances"][0]):
        similarity = 1.0 - (dist / 2.0)  # convert cosine distance to similarity
        pairs.append((cid, round(similarity, 4)))

    return pairs


def _build_chroma_filter(filters: dict) -> dict:
    """Convert API filters to ChromaDB where clause."""
    conditions = []
    
    if "min_years_experience" in filters:
        conditions.append({"years_experience": {"$gte": filters["min_years_experience"]}})
    
    if "max_years_experience" in filters:
        conditions.append({"years_experience": {"$lte": filters["max_years_experience"]}})

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def compute_skills_overlap(
    candidate_skills: List[str],
    required_skills: List[str],
    preferred_skills: List[str],
) -> Tuple[float, List[str], List[str]]:
    """
    Compute skills overlap score + return matched and missing skills.
    Required skills weighted 2x preferred.
    """
    cand_lower = {s.lower() for s in candidate_skills}
    req_lower = {s.lower() for s in required_skills}
    pref_lower = {s.lower() for s in preferred_skills}

    # Fuzzy partial match: "python" matches "python3", "pytorch" partial matches "torch"
    def skill_matches(candidate_set: set, target: str) -> bool:
        target_l = target.lower()
        return any(
            target_l in cs or cs in target_l
            for cs in candidate_set
        )

    matched_req = [s for s in required_skills if skill_matches(cand_lower, s)]
    matched_pref = [s for s in preferred_skills if skill_matches(cand_lower, s)]
    missing = [s for s in required_skills if not skill_matches(cand_lower, s)]

    req_score = len(matched_req) / max(len(required_skills), 1)
    pref_score = len(matched_pref) / max(len(preferred_skills), 1) if preferred_skills else 0.5

    final_score = (0.7 * req_score) + (0.3 * pref_score)
    all_matched = matched_req + matched_pref

    return round(final_score, 3), all_matched, missing