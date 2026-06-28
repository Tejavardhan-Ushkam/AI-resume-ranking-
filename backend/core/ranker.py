"""
Ensemble Ranker: Combines semantic similarity, skills overlap, Career DNA fit,
and behavioral signals into a final calibrated score.

The key insight: different JDs should weight different signals differently.
A startup engineering role weights skills heavily.
A director role weights Career DNA (growth velocity, leadership) heavily.
"""
import numpy as np
from typing import List, Tuple
from loguru import logger

from models.schemas import (
    CandidateInDB, JDDecomposition, ScoreBreakdown,
    CandidateRankResult
)
from config import settings
from core.signal_engine import compute_skills_overlap
from core.career_dna import career_dna_to_vector


def compute_career_dna_fit(
    candidate: CandidateInDB,
    jd: JDDecomposition,
) -> float:
    """
    Match CareerDNA axes against JD requirements.
    JD decomposition tells us which axes matter most.
    """
    if not candidate.career_dna:
        return 0.4  # neutral fallback

    dna = candidate.career_dna
    vector = career_dna_to_vector(dna)

    # Build a target Career DNA vector based on JD
    # e.g., senior IC role needs high skill_depth + recency, less collaboration
    seniority = jd.role_seniority
    role_type = jd.role_type

    target_dna = {
        "skill_depth_score": 0.8 if seniority in ("senior", "lead", "executive") else 0.5,
        "growth_velocity": 0.7 if seniority in ("lead", "executive") else 0.5,
        "domain_breadth": 0.7 if role_type == "hybrid" else 0.5,
        "tenure_stability": 0.7,  # always valued
        "recency_score": 0.8,     # always valued
        "collaboration_signal": 0.8 if role_type in ("manager", "hybrid") else 0.4,
    }

    target_vector = np.array(list(target_dna.values()))
    actual_vector = np.array(vector)

    # Weighted Euclidean distance converted to similarity
    # Axes weighted by JD importance
    axis_weights = np.array([
        jd.skill_importance_weight,
        0.15,  # growth velocity
        0.10,  # domain breadth
        0.10,  # tenure stability
        0.15,  # recency
        jd.culture_fit_weight * 0.5,  # collaboration
    ])
    # Normalize weights
    axis_weights = axis_weights / axis_weights.sum()

    squared_diff = axis_weights * (actual_vector - target_vector) ** 2
    distance = np.sqrt(squared_diff.sum())

    # Max possible distance (all weights, max diff=1) gives us normalization
    max_distance = np.sqrt(axis_weights.sum())
    similarity = 1.0 - (distance / max_distance)

    return round(float(max(0.0, min(1.0, similarity))), 3)


def compute_behavioral_score(candidate: CandidateInDB, jd: JDDecomposition) -> float:
    """
    Score based on activity signals and metadata.
    Open-to-work, high completeness, active profile = better.
    """
    b = candidate.behavioral_signals
    
    scores = []
    scores.append(b.profile_completeness)
    
    # Activity recency (less days = better)
    if b.last_active_days_ago == 0:
        activity_score = 0.5  # unknown
    elif b.last_active_days_ago <= 7:
        activity_score = 1.0
    elif b.last_active_days_ago <= 30:
        activity_score = 0.8
    elif b.last_active_days_ago <= 90:
        activity_score = 0.5
    else:
        activity_score = 0.2
    scores.append(activity_score)

    # Open to work bonus
    scores.append(1.0 if b.open_to_work else 0.4)

    # Enrichment signals
    certs_score = min(b.certifications_count / 5, 1.0)
    projects_score = min(b.projects_count / 5, 1.0)
    endorsements_score = min(b.endorsements_count / 20, 1.0)
    scores.append((certs_score + projects_score + endorsements_score) / 3)

    # Response rate
    scores.append(b.response_rate)

    return round(float(np.mean(scores)), 3)


def ensemble_score(
    semantic_sim: float,
    skills_score: float,
    dna_fit: float,
    behavioral_score: float,
    jd: JDDecomposition,
) -> ScoreBreakdown:
    """
    Adaptive ensemble: weights come from JD decomposition + role type.
    """
    # Dynamic weights from JD
    w_semantic = settings.WEIGHT_SEMANTIC  # 0.35 base
    w_skills = settings.WEIGHT_SKILLS      # 0.25 base
    w_dna = settings.WEIGHT_CAREER_DNA     # 0.25 base
    w_behavioral = settings.WEIGHT_BEHAVIORAL  # 0.15 base

    # Adjust based on JD role type
    if jd.role_type == "manager":
        w_dna += 0.05
        w_skills -= 0.05
    elif jd.role_seniority in ("lead", "executive"):
        w_dna += 0.05
        w_behavioral -= 0.05

    # Normalize
    total = w_semantic + w_skills + w_dna + w_behavioral
    w_semantic /= total
    w_skills /= total
    w_dna /= total
    w_behavioral /= total

    final = (
        w_semantic * semantic_sim
        + w_skills * skills_score
        + w_dna * dna_fit
        + w_behavioral * behavioral_score
    )

    return ScoreBreakdown(
        semantic_match=round(semantic_sim, 3),
        skills_overlap=round(skills_score, 3),
        career_dna_fit=round(dna_fit, 3),
        behavioral_score=round(behavioral_score, 3),
        final_score=round(float(final), 3),
    )


def identify_strengths(
    candidate: CandidateInDB,
    jd: JDDecomposition,
    matched_skills: List[str],
    score_breakdown: ScoreBreakdown,
) -> List[str]:
    """Generate 2-3 key strength bullets specific to this JD."""
    strengths = []
    
    if len(matched_skills) >= 3:
        strengths.append(f"Strong skill match: {', '.join(matched_skills[:3])}")
    
    if candidate.years_of_experience >= jd.min_years_experience * 1.5:
        strengths.append(
            f"{candidate.years_of_experience:.0f} years experience (exceeds {jd.min_years_experience:.0f}yr requirement)"
        )
    
    if candidate.career_dna:
        dna = candidate.career_dna
        if dna.growth_velocity > 0.7 and jd.role_seniority in ("senior", "lead"):
            strengths.append("High career growth velocity — consistent upward trajectory")
        if dna.tenure_stability > 0.75:
            strengths.append("Excellent tenure stability — reliable long-term contributor")
        if dna.collaboration_signal > 0.7 and jd.role_type in ("manager", "hybrid"):
            strengths.append("Strong leadership and collaboration signals")
    
    return strengths[:3] or ["Good overall profile match"]


def identify_risks(
    candidate: CandidateInDB,
    jd: JDDecomposition,
    missing_skills: List[str],
    score_breakdown: ScoreBreakdown,
) -> List[str]:
    """Generate honest risk/gap bullets."""
    risks = []
    
    if missing_skills:
        risks.append(f"Missing key skills: {', '.join(missing_skills[:3])}")
    
    if candidate.years_of_experience < jd.min_years_experience:
        gap = jd.min_years_experience - candidate.years_of_experience
        risks.append(f"Under-experienced by ~{gap:.0f} years vs requirement")
    
    if candidate.career_dna:
        dna = candidate.career_dna
        if dna.tenure_stability < 0.4:
            risks.append("Short average job tenure — potential retention risk")
        if dna.recency_score < 0.3:
            risks.append("Recent experience may not align with current role scope")
    
    return risks[:3] or []


def rank_candidates(
    candidates: List[CandidateInDB],
    semantic_scores: dict,  # {candidate_id: float}
    jd: JDDecomposition,
) -> List[CandidateRankResult]:
    """
    Full ranking pipeline.
    """
    results = []

    for candidate in candidates:
        semantic_sim = semantic_scores.get(candidate.id, 0.0)

        skills_score, matched_skills, missing_skills = compute_skills_overlap(
            candidate.skills,
            jd.required_skills,
            jd.preferred_skills,
        )

        dna_fit = compute_career_dna_fit(candidate, jd)
        behavioral = compute_behavioral_score(candidate, jd)

        score_breakdown = ensemble_score(
            semantic_sim, skills_score, dna_fit, behavioral, jd
        )

        strengths = identify_strengths(candidate, jd, matched_skills, score_breakdown)
        risks = identify_risks(candidate, jd, missing_skills, score_breakdown)

        match_pct = round(score_breakdown.final_score * 100, 1)

        results.append(
            CandidateRankResult(
                candidate=candidate,
                rank=0,  # set after sorting
                score_breakdown=score_breakdown,
                match_percentage=match_pct,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                strengths=strengths,
                risks=risks,
                ai_summary="",  # filled by explainer
            )
        )

    # Sort by final score descending
    results.sort(key=lambda r: r.score_breakdown.final_score, reverse=True)

    # Assign ranks
    for i, r in enumerate(results):
        r.rank = i + 1

    return results