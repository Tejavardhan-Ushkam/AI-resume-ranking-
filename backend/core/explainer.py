"""
Explainer: Generates human-readable AI summaries explaining WHY each candidate
was ranked where they were. Uses Claude for rich, JD-specific explanations.
"""
import asyncio
from typing import List
from loguru import logger

from models.schemas import CandidateRankResult, JDDecomposition
from config import settings


SUMMARY_TEMPLATE = """Candidate: {name} | Score: {score:.0f}% | Rank #{rank}
Skills matched: {skills} | Missing: {missing}
Strengths: {strengths}"""


async def generate_ai_summary(
    result: CandidateRankResult,
    jd: JDDecomposition,
    job_title: str,
) -> str:
    """Generate a concise AI explanation using Claude."""
    if not settings.ANTHROPIC_API_KEY:
        return _generate_rule_based_summary(result, jd, job_title)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        candidate = result.candidate
        dna = candidate.career_dna

        prompt = f"""You are an expert technical recruiter writing a concise candidate assessment.

Role: {job_title}
Required Skills: {', '.join(jd.required_skills[:8])}
Seniority: {jd.role_seniority}

Candidate: {candidate.name}
Title: {candidate.current_title} | {candidate.years_of_experience} years experience
Skills: {', '.join(candidate.skills[:10])}
Summary: {candidate.summary[:200]}
Match Score: {result.match_percentage}%
Matched Skills: {', '.join(result.matched_skills[:5])}
Missing Skills: {', '.join(result.missing_skills[:3])}
Career DNA - Skill Depth: {dna.skill_depth_score if dna else 'N/A':.2f}, Growth Velocity: {dna.growth_velocity if dna else 'N/A':.2f}

Write exactly 2 sentences:
1. Why this candidate stands out for this role (be specific, cite concrete evidence)
2. One honest gap or watch-out

Be direct, specific, no fluff."""

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()

    except Exception as e:
        logger.warning(f"AI summary failed for {result.candidate.name}: {e}")
        return _generate_rule_based_summary(result, jd, job_title)


def _generate_rule_based_summary(
    result: CandidateRankResult,
    jd: JDDecomposition,
    job_title: str,
) -> str:
    """Deterministic fallback summary when AI is unavailable."""
    candidate = result.candidate
    
    # Strength sentence
    if result.matched_skills:
        strength_part = f"Strong match on {', '.join(result.matched_skills[:3])}"
    else:
        strength_part = f"General profile alignment with {job_title} requirements"
    
    if candidate.years_of_experience >= jd.min_years_experience:
        exp_part = f"with {candidate.years_of_experience:.0f} years of relevant experience"
    else:
        exp_part = f"though slightly below the {jd.min_years_experience:.0f}-year experience bar"
    
    # Risk sentence
    if result.missing_skills:
        risk_part = f"Watch-out: gaps in {', '.join(result.missing_skills[:2])} may require onboarding investment."
    elif candidate.career_dna and candidate.career_dna.tenure_stability < 0.4:
        risk_part = "Watch-out: shorter average tenure history — assess long-term retention fit."
    else:
        risk_part = "No significant gaps identified against stated role requirements."

    return f"{strength_part} {exp_part}. {risk_part}"


async def enrich_results_with_summaries(
    results: List[CandidateRankResult],
    jd: JDDecomposition,
    job_title: str,
    top_k: int = 10,
) -> List[CandidateRankResult]:
    """
    Adds AI summaries to top-K results. Uses asyncio for parallel generation.
    Only generates for top results to save cost/time.
    """
    tasks = []
    for result in results[:top_k]:
        tasks.append(generate_ai_summary(result, jd, job_title))

    summaries = await asyncio.gather(*tasks, return_exceptions=True)

    for result, summary in zip(results[:top_k], summaries):
        if isinstance(summary, Exception):
            result.ai_summary = _generate_rule_based_summary(result, jd, job_title)
        else:
            result.ai_summary = summary

    # Fill rule-based for the rest
    for result in results[top_k:]:
        result.ai_summary = _generate_rule_based_summary(result, jd, job_title)

    return results