"""
JD Analyzer: Deep multi-axis job description decomposition.
Uses Claude (or fallback heuristics) to extract structured signals from raw JD text.
"""
import json
import re
from typing import Optional
from loguru import logger

from models.schemas import JDDecomposition
from config import settings


SENIORITY_KEYWORDS = {
    "junior": ["junior", "entry", "associate", "graduate", "intern", "fresher", "0-2 years", "1-2 years"],
    "mid": ["mid", "intermediate", "3-5 years", "2-4 years", "2-5 years"],
    "senior": ["senior", "sr.", "5+ years", "6+ years", "7+ years", "experienced"],
    "lead": ["lead", "principal", "staff", "10+ years", "8+ years"],
    "executive": ["head of", "director", "vp", "chief", "c-level", "cto", "ceo"],
}

CULTURE_SIGNALS_MAP = {
    "fast-paced": ["fast-paced", "fast paced", "startup", "agile", "dynamic", "high-growth"],
    "collaborative": ["collaborative", "team player", "cross-functional", "partnership"],
    "autonomous": ["self-starter", "independent", "ownership", "autonomous", "minimal supervision"],
    "data-driven": ["data-driven", "metrics", "kpis", "analytical", "evidence-based"],
    "innovative": ["innovative", "cutting-edge", "state-of-the-art", "novel", "creative"],
    "customer-focused": ["customer", "client", "user-centric", "product-led"],
    "remote-friendly": ["remote", "distributed", "async", "work from home"],
}

COMMON_SKILLS = [
    # Programming
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#", "ruby", "scala", "kotlin", "swift",
    # Web
    "react", "vue", "angular", "node.js", "django", "fastapi", "flask", "spring", "rails", "next.js",
    # Data
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    # ML/AI
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "nlp", "llm",
    "data science", "pandas", "numpy", "spark", "hadoop",
    # Cloud/DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci/cd", "linux",
    # Soft
    "leadership", "communication", "project management", "agile", "scrum",
]


def extract_skills_heuristic(text: str) -> list[str]:
    """Fast keyword-based skill extraction as baseline."""
    text_lower = text.lower()
    found = []
    for skill in COMMON_SKILLS:
        # word boundary check
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def detect_seniority(text: str) -> str:
    text_lower = text.lower()
    scores = {level: 0 for level in SENIORITY_KEYWORDS}
    for level, keywords in SENIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[level] += 1
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "mid"


def detect_culture_signals(text: str) -> list[str]:
    text_lower = text.lower()
    signals = []
    for signal, keywords in CULTURE_SIGNALS_MAP.items():
        if any(kw in text_lower for kw in keywords):
            signals.append(signal)
    return signals or ["professional"]


def extract_years_experience(text: str) -> float:
    """Extract minimum years of experience from text."""
    patterns = [
        r'minimum\s+(\d+)\s+years?',
        r'at\s+least\s+(\d+)\s+years?',
        r'(\d+)\+\s*years?\s+(?:of\s+)?experience',
        r'(\d+)\s*\+\s*years?\s+(?:of\s+)?',
        r'(\d+)\s+years?\s+(?:of\s+)?(?:experience|exp)',
        r'(\d+)-\d+\s+years?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))
    return 2.0  # sensible default


def detect_role_type(text: str) -> str:
    text_lower = text.lower()
    # Strong manager signals
    strong_manager = ["people manager", "direct reports", "hiring manager", "manage a team", "head of"]
    weak_manager = ["manage", "team lead", "lead a team"]
    ic_signals = ["hands-on", "individual contributor", "coding", "write code", "technical execution", "engineer", "develop", "implement"]

    strong_hits = sum(1 for kw in strong_manager if kw in text_lower)
    weak_hits = sum(1 for kw in weak_manager if kw in text_lower)
    ic_hits = sum(1 for kw in ic_signals if kw in text_lower)

    is_manager = strong_hits >= 1 or weak_hits >= 2
    is_ic = ic_hits >= 2

    if is_manager and is_ic:
        return "hybrid"
    if is_manager:
        return "manager"
    return "ic"


async def analyze_jd_with_ai(jd_text: str, title: str) -> Optional[dict]:
    """
    Use Claude to perform deep semantic JD analysis.
    Returns structured JSON or None if API unavailable.
    """
    if not settings.ANTHROPIC_API_KEY:
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        prompt = f"""Analyze this job description and return ONLY valid JSON (no markdown, no explanation).

Job Title: {title}
Job Description:
{jd_text}

Return this exact JSON structure:
{{
  "required_skills": ["list of must-have technical and soft skills"],
  "preferred_skills": ["list of nice-to-have skills"],
  "implicit_requirements": ["skills/traits implied but not stated explicitly"],
  "culture_signals": ["workplace culture descriptors"],
  "role_seniority": "junior|mid|senior|lead|executive",
  "role_type": "ic|manager|hybrid",
  "domain_focus": ["primary technical/business domains"],
  "min_years_experience": 3,
  "education_requirement": "Bachelor's|Master's|PhD|null",
  "enriched_summary": "2-3 sentences capturing the essence of what makes a PERFECT candidate for this role, going beyond keywords to capture the true fit criteria"
}}"""

        message = client.messages.create(
            model="claude-3-haiku-20240307",  # fast + cheap for structured extraction
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        return json.loads(raw)

    except Exception as e:
        logger.warning(f"AI JD analysis failed, using heuristics: {e}")
        return None


async def decompose_jd(jd_text: str, title: str = "Role") -> JDDecomposition:
    """
    Main entry: returns full JD decomposition.
    Tries AI first, falls back to heuristics - always returns a result.
    """
    ai_result = await analyze_jd_with_ai(jd_text, title)
    all_skills = extract_skills_heuristic(jd_text)

    if ai_result:
        required_skills = ai_result.get("required_skills", all_skills[:8])
        preferred_skills = ai_result.get("preferred_skills", all_skills[8:])
        implicit = ai_result.get("implicit_requirements", [])
        culture_signals = ai_result.get("culture_signals", detect_culture_signals(jd_text))
        seniority = ai_result.get("role_seniority", detect_seniority(jd_text))
        role_type = ai_result.get("role_type", detect_role_type(jd_text))
        domain_focus = ai_result.get("domain_focus", ["technology"])
        min_years = float(ai_result.get("min_years_experience", extract_years_experience(jd_text)))
        education = ai_result.get("education_requirement")
        enriched = ai_result.get("enriched_summary", jd_text[:500])
        
        # Blend enriched description for better embedding
        enriched_for_embed = (
            f"Job: {title}. Seniority: {seniority}. Type: {role_type}. "
            f"Required: {', '.join(required_skills[:10])}. "
            f"Culture: {', '.join(culture_signals)}. "
            f"Context: {enriched}"
        )
    else:
        # Pure heuristic fallback
        required_skills = all_skills[:8]
        preferred_skills = all_skills[8:15]
        implicit = []
        culture_signals = detect_culture_signals(jd_text)
        seniority = detect_seniority(jd_text)
        role_type = detect_role_type(jd_text)
        domain_focus = ["technology"]
        min_years = extract_years_experience(jd_text)
        education = None
        enriched_for_embed = f"{title}. {jd_text[:600]}"

    # Infer importance weights from JD characteristics
    skill_weight = 0.45 if role_type == "ic" else 0.30
    exp_weight = 0.30 if seniority in ("lead", "executive") else 0.25
    culture_weight = 1.0 - skill_weight - exp_weight

    return JDDecomposition(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_years_experience=min_years,
        education_requirement=education,
        role_seniority=seniority,
        role_type=role_type,
        culture_signals=culture_signals,
        domain_focus=domain_focus,
        skill_importance_weight=round(skill_weight, 2),
        experience_importance_weight=round(exp_weight, 2),
        culture_fit_weight=round(culture_weight, 2),
        enriched_description=enriched_for_embed,
    )