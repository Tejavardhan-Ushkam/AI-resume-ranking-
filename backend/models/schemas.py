from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timezone


# ─── Candidate Models ───────────────────────────────────────────────────────

class WorkExperience(BaseModel):
    title: str
    company: str
    start_date: str  # "YYYY-MM" format
    end_date: Optional[str] = None  # None = current
    description: str
    skills_used: List[str] = []


class Education(BaseModel):
    degree: str
    field: str
    institution: str
    year: int


class CareerDNA(BaseModel):
    """Our novel 6-axis career fingerprint"""
    skill_depth_score: float = Field(ge=0, le=1, description="Seniority within skill domains")
    growth_velocity: float = Field(ge=0, le=1, description="Rate of career progression")
    domain_breadth: float = Field(ge=0, le=1, description="Versatility across domains")
    tenure_stability: float = Field(ge=0, le=1, description="Job tenure reliability")
    recency_score: float = Field(ge=0, le=1, description="Relevance of recent experience")
    collaboration_signal: float = Field(ge=0, le=1, description="Leadership/team signals")


class BehavioralSignals(BaseModel):
    """Activity and behavioral metadata"""
    profile_completeness: float = Field(default=0.5, ge=0, le=1)
    last_active_days_ago: int = 0
    endorsements_count: int = 0
    certifications_count: int = 0
    projects_count: int = 0
    publications_count: int = 0
    open_to_work: bool = False
    response_rate: float = Field(ge=0, le=1, default=0.5)


class CandidateCreate(BaseModel):
    name: str
    email: str
    current_title: str
    current_company: Optional[str] = None
    location: str
    years_of_experience: float
    skills: List[str]
    summary: str
    work_experience: List[WorkExperience] = []
    education: List[Education] = []
    behavioral_signals: BehavioralSignals = BehavioralSignals()


class CandidateInDB(CandidateCreate):
    id: str
    career_dna: Optional[CareerDNA] = None
    embedding_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Job Description Models ─────────────────────────────────────────────────

class JDDecomposition(BaseModel):
    """Deep multi-axis JD analysis"""
    # Explicit requirements
    required_skills: List[str]
    preferred_skills: List[str]
    min_years_experience: float
    education_requirement: Optional[str]
    
    # Semantic / implicit
    role_seniority: str  # "junior" | "mid" | "senior" | "lead" | "executive"
    role_type: str  # "ic" | "manager" | "hybrid"
    culture_signals: List[str]  # ["fast-paced", "collaborative", "autonomous"]
    domain_focus: List[str]  # primary domains
    
    # Weights for this JD (auto-inferred)
    skill_importance_weight: float = 0.4
    experience_importance_weight: float = 0.3
    culture_fit_weight: float = 0.3

    # Full text for embedding
    enriched_description: str


class JobDescriptionCreate(BaseModel):
    title: str
    company: Optional[str] = None
    description: str
    location: Optional[str] = None


class JobDescriptionInDB(JobDescriptionCreate):
    id: str
    decomposition: Optional[JDDecomposition] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Search / Ranking Models ─────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    """Transparent per-dimension scores"""
    semantic_match: float
    skills_overlap: float
    career_dna_fit: float
    behavioral_score: float
    final_score: float


class CandidateRankResult(BaseModel):
    candidate: CandidateInDB
    rank: int
    score_breakdown: ScoreBreakdown
    match_percentage: float
    matched_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]  # 2-3 key strengths for this JD
    risks: List[str]      # honest risks/gaps
    ai_summary: str       # 1-paragraph AI-generated fit summary


class SearchRequest(BaseModel):
    job_description: str
    job_title: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    job_id: str
    jd_decomposition: JDDecomposition
    results: List[CandidateRankResult]
    total_candidates_evaluated: int
    processing_time_ms: float