const BASE_URL = '/api';

export interface WorkExperience {
  title: string;
  company: string;
  start_date: string;
  end_date: string | null;
  description: string;
  skills_used: string[];
}

export interface CareerDNA {
  skill_depth_score: number;
  growth_velocity: number;
  domain_breadth: number;
  tenure_stability: number;
  recency_score: number;
  collaboration_signal: number;
}

export interface BehavioralSignals {
  profile_completeness: number;
  last_active_days_ago: number;
  endorsements_count: number;
  certifications_count: number;
  projects_count: number;
  publications_count: number;
  open_to_work: boolean;
  response_rate: number;
}

export interface Candidate {
  id: string;
  name: string;
  email: string;
  current_title: string;
  current_company: string | null;
  location: string;
  years_of_experience: number;
  skills: string[];
  summary: string;
  work_experience: WorkExperience[];
  career_dna: CareerDNA | null;
  behavioral_signals: BehavioralSignals;
}

export interface ScoreBreakdown {
  semantic_match: number;
  skills_overlap: number;
  career_dna_fit: number;
  behavioral_score: number;
  final_score: number;
}

export interface CandidateRankResult {
  candidate: Candidate;
  rank: number;
  score_breakdown: ScoreBreakdown;
  match_percentage: number;
  matched_skills: string[];
  missing_skills: string[];
  strengths: string[];
  risks: string[];
  ai_summary: string;
}

export interface JDDecomposition {
  required_skills: string[];
  preferred_skills: string[];
  min_years_experience: number;
  education_requirement: string | null;
  role_seniority: string;
  role_type: string;
  culture_signals: string[];
  domain_focus: string[];
  skill_importance_weight: number;
  experience_importance_weight: number;
  culture_fit_weight: number;
  enriched_description: string;
}

export interface SearchResponse {
  job_id: string;
  jd_decomposition: JDDecomposition;
  results: CandidateRankResult[];
  total_candidates_evaluated: number;
  processing_time_ms: number;
}

export interface SearchRequest {
  job_description: string;
  job_title?: string;
  top_k?: number;
  filters?: Record<string, unknown>;
}

export async function searchCandidates(request: SearchRequest): Promise<SearchResponse> {
  const res = await fetch(`${BASE_URL}/search/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Search failed: ${res.status}`);
  }

  return res.json();
}

export async function getCandidateCount(): Promise<number> {
  const res = await fetch(`${BASE_URL}/candidates/count`);
  if (!res.ok) return 0;
  const data = await res.json();
  return data.count;
}

export async function getHealth(): Promise<{
  status: string;
  candidates_indexed: number;
  embedding_model: string;
  ai_enabled: boolean;
}> {
  const res = await fetch('/health');
  if (!res.ok) throw new Error('Backend not available');
  return res.json();
}

export async function bulkUploadCandidates(candidates: unknown[]): Promise<{
  created_count: number;
  error_count: number;
}> {
  const res = await fetch(`${BASE_URL}/candidates/bulk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(candidates),
  });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}