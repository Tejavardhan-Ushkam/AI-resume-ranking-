"""
Career DNA: A novel multi-axis career fingerprint for richer candidate representation.

Instead of treating a resume as a flat text blob, we model 6 independent axes:
1. Skill Depth Score    — Seniority/expertise within claimed skill domains
2. Growth Velocity      — Rate of career progression (promotions, scope expansion)
3. Domain Breadth       — Versatility across industries/functions  
4. Tenure Stability     — Job tenure reliability signal
5. Recency Score        — How fresh/relevant recent experience is
6. Collaboration Signal — Leadership, mentoring, cross-functional signals

This enables axis-aware matching against the JD decomposition.
"""
from datetime import datetime, timezone, timezone
from typing import List
from dateutil import parser as dateparser
import re

from models.schemas import CandidateCreate, CareerDNA, WorkExperience


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_date(date_str: str) -> datetime:
    """Parse 'YYYY-MM' or natural language date strings. Always returns naive UTC datetime."""
    if not date_str:
        return datetime.utcnow()
    try:
        if re.match(r'^\d{4}-\d{2}$', date_str):
            return datetime.strptime(date_str, "%Y-%m")
        parsed = dateparser.parse(date_str)
        if parsed and parsed.tzinfo is not None:
            return parsed.replace(tzinfo=None)
        return parsed or datetime.utcnow()
    except Exception:
        return datetime.utcnow()


def _now() -> datetime:
    """Current time as naive UTC datetime (for consistent arithmetic)."""
    return datetime.utcnow()


def tenure_months(exp: WorkExperience) -> float:
    """Duration of a work experience entry in months."""
    start = parse_date(exp.start_date)
    end = parse_date(exp.end_date) if exp.end_date else _now()
    return max(0.0, (end - start).days / 30.44)


# ─── Individual Axis Calculators ─────────────────────────────────────────────

def compute_skill_depth_score(candidate: CandidateCreate) -> float:
    """
    Proxy: Years of experience × skill count × title seniority bonus.
    Senior people with many skills get higher depth scores.
    """
    seniority_bonus = 0.0
    title_lower = candidate.current_title.lower()
    if any(k in title_lower for k in ["principal", "staff", "architect", "head", "director", "vp"]):
        seniority_bonus = 0.3
    elif any(k in title_lower for k in ["senior", "sr.", "lead"]):
        seniority_bonus = 0.2
    elif any(k in title_lower for k in ["junior", "jr.", "associate", "intern"]):
        seniority_bonus = -0.1

    # Skills breadth contributes to depth (more skills = likely deeper)
    skill_factor = min(len(candidate.skills) / 20, 1.0)
    
    # Years of experience normalized (10+ years = max)
    exp_factor = min(candidate.years_of_experience / 10, 1.0)

    raw = (0.5 * exp_factor) + (0.3 * skill_factor) + (0.2 * (0.5 + seniority_bonus))
    return round(min(max(raw, 0.0), 1.0), 3)


def compute_growth_velocity(candidate: CandidateCreate) -> float:
    """
    Measures career progression speed:
    - Promotions (ascending title seniority over time)
    - Average tenure per role (too short = job hopper, too long = stagnation)
    - Company diversity (multiple logos = broader exposure)
    """
    if not candidate.work_experience:
        return 0.3  # neutral

    experiences = sorted(
        candidate.work_experience,
        key=lambda x: parse_date(x.start_date)
    )

    # Average tenure (ideal: 18-36 months)
    tenures = [tenure_months(e) for e in experiences]
    avg_tenure = sum(tenures) / len(tenures) if tenures else 0
    
    if avg_tenure < 6:
        tenure_score = 0.2  # job hopper
    elif avg_tenure < 18:
        tenure_score = 0.6
    elif avg_tenure <= 36:
        tenure_score = 1.0  # ideal
    elif avg_tenure <= 60:
        tenure_score = 0.7
    else:
        tenure_score = 0.4  # stagnation risk

    # Unique companies
    unique_companies = len(set(e.company for e in experiences))
    company_score = min(unique_companies / 5, 1.0)

    # Title progression heuristic
    seniority_order = ["intern", "junior", "associate", "mid", "", "senior", "lead", "principal", "director", "head", "vp", "chief"]
    
    def title_rank(title: str) -> int:
        t = title.lower()
        for i, kw in enumerate(seniority_order):
            if kw and kw in t:
                return i
        return 4  # default "mid"

    if len(experiences) >= 2:
        first_rank = title_rank(experiences[0].title)
        last_rank = title_rank(experiences[-1].title)
        progression = (last_rank - first_rank) / max(len(experiences), 1)
        progression_score = min(max((progression + 2) / 4, 0), 1.0)
    else:
        progression_score = 0.5

    raw = (0.4 * tenure_score) + (0.3 * company_score) + (0.3 * progression_score)
    return round(raw, 3)


def compute_domain_breadth(candidate: CandidateCreate) -> float:
    """
    Measures versatility: unique industries, diverse skill categories.
    """
    skill_categories = {
        "frontend": ["react", "vue", "angular", "css", "html", "javascript", "typescript"],
        "backend": ["python", "java", "go", "node", "django", "fastapi", "spring", "rails"],
        "data": ["sql", "pandas", "spark", "hadoop", "etl", "data pipeline"],
        "ml_ai": ["machine learning", "deep learning", "tensorflow", "pytorch", "nlp", "llm"],
        "devops": ["docker", "kubernetes", "aws", "gcp", "azure", "terraform", "ci/cd"],
        "mobile": ["ios", "android", "react native", "flutter", "swift", "kotlin"],
        "management": ["leadership", "agile", "scrum", "product management", "strategy"],
    }

    skills_lower = [s.lower() for s in candidate.skills]
    summary_lower = candidate.summary.lower()
    all_text = " ".join(skills_lower) + " " + summary_lower

    covered_categories = 0
    for cat_keywords in skill_categories.values():
        if any(kw in all_text for kw in cat_keywords):
            covered_categories += 1

    breadth = covered_categories / len(skill_categories)

    # Also consider number of industries from work experience
    companies_text = " ".join(
        e.company.lower() + " " + e.description.lower()
        for e in candidate.work_experience
    )

    return round(min(breadth * 1.2, 1.0), 3)  # slight boost for breadth


def compute_tenure_stability(candidate: CandidateCreate) -> float:
    """
    Reliability signal: penalize job hopping, reward healthy tenure.
    """
    if not candidate.work_experience:
        return 0.5

    tenures = [tenure_months(e) for e in candidate.work_experience]
    if not tenures:
        return 0.5

    avg_tenure = sum(tenures) / len(tenures)
    short_stints = sum(1 for t in tenures if t < 12)
    hopper_penalty = short_stints / len(tenures)

    # Good stability: avg tenure 18-36 months, few short stints
    if avg_tenure >= 24 and hopper_penalty < 0.2:
        base = 0.9
    elif avg_tenure >= 12 and hopper_penalty < 0.4:
        base = 0.7
    elif avg_tenure >= 6:
        base = 0.5
    else:
        base = 0.3

    return round(base - (0.2 * hopper_penalty), 3)


def compute_recency_score(candidate: CandidateCreate) -> float:
    """
    How relevant is their recent experience?
    Recent roles count more than old ones.
    """
    if not candidate.work_experience:
        return 0.3

    now = _now()
    total_weight = 0.0
    weighted_recency = 0.0

    for exp in candidate.work_experience:
        end = parse_date(exp.end_date) if exp.end_date else now
        months_ago = (now - end).days / 30.44
        
        # Exponential decay: recent = high weight, old = low weight
        weight = max(0.0, 1.0 - (months_ago / 120))  # 10-year decay
        tenure = tenure_months(exp)
        relevance = min(tenure / 24, 1.0)  # longer recent role = more relevant
        
        weighted_recency += weight * relevance
        total_weight += 1.0

    score = weighted_recency / max(total_weight, 1)
    return round(min(score, 1.0), 3)


def compute_collaboration_signal(candidate: CandidateCreate) -> float:
    """
    Leadership, mentoring, cross-functional signals from text.
    """
    leadership_keywords = [
        "led", "managed", "mentored", "supervised", "directed", "headed",
        "coached", "guided", "team of", "reports", "cross-functional",
        "stakeholder", "collaborated", "partnership", "initiative"
    ]
    
    all_text = candidate.summary.lower()
    for exp in candidate.work_experience:
        all_text += " " + exp.description.lower()
    all_text += " ".join(candidate.skills).lower()

    hits = sum(1 for kw in leadership_keywords if kw in all_text)
    raw = min(hits / 8, 1.0)

    # Boost for explicit leadership titles
    if any(k in candidate.current_title.lower() for k in ["lead", "manager", "head", "director", "principal"]):
        raw = min(raw + 0.2, 1.0)

    return round(raw, 3)


# ─── Main Entry ──────────────────────────────────────────────────────────────

def compute_career_dna(candidate: CandidateCreate) -> CareerDNA:
    """
    Compute the full 6-axis Career DNA fingerprint for a candidate.
    """
    return CareerDNA(
        skill_depth_score=compute_skill_depth_score(candidate),
        growth_velocity=compute_growth_velocity(candidate),
        domain_breadth=compute_domain_breadth(candidate),
        tenure_stability=compute_tenure_stability(candidate),
        recency_score=compute_recency_score(candidate),
        collaboration_signal=compute_collaboration_signal(candidate),
    )


def career_dna_to_vector(dna: CareerDNA) -> List[float]:
    """Convert CareerDNA to a 6D numeric vector for distance calculations."""
    return [
        dna.skill_depth_score,
        dna.growth_velocity,
        dna.domain_breadth,
        dna.tenure_stability,
        dna.recency_score,
        dna.collaboration_signal,
    ]