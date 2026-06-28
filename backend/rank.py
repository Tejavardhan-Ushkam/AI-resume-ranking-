#!/usr/bin/env python3
"""
rank.py — Standalone ranker for Redrob Hackathon
Reads candidates.jsonl (or .jsonl.gz), ranks top 100 for the Senior AI Engineer JD.

Usage:
  python rank.py --candidates candidates.jsonl --out submission.csv
  python rank.py --candidates candidates.jsonl.gz --out submission.csv

Constraints: ≤5 min, CPU-only, ≤16 GB RAM, no LLM API calls.
"""

import argparse
import csv
import gzip
import json
import math
import re
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TODAY = date.today()

# ─── JD Definition ───────────────────────────────────────────────────────────

JD_TEXT = """
Senior AI Engineer founding team Redrob AI Series A talent intelligence platform
Pune Noida India hybrid relocation
5 to 9 years experience applied machine learning production
embeddings retrieval ranking search recommendation systems
sentence transformers BGE E5 OpenAI embeddings production deployment
vector database pinecone weaviate qdrant milvus faiss opensearch elasticsearch
hybrid search BM25 dense retrieval
evaluation framework NDCG MRR MAP A/B testing offline online
strong Python code quality production systems
LLM fine tuning LoRA QLoRA PEFT learning to rank XGBoost neural ranking
product company startup not consulting not research only
information retrieval NLP natural language processing
candidate job matching talent intelligence recruiting platform
distributed systems large scale inference optimization
open source contributions AI ML
scrappy product engineering ship fast iterate learn users
"""

# ─── Skill Lists (exact names from actual dataset) ────────────────────────────

CORE_REQUIRED = {
    # Vector / retrieval
    "embeddings", "sentence transformers", "sentence-transformers",
    "faiss", "pinecone", "qdrant", "weaviate", "milvus", "opensearch",
    "elasticsearch", "vector search", "information retrieval",
    "hybrid search", "bm25", "retrieval", "semantic search",
    # Ranking / eval
    "ranking", "recommendation systems", "ndcg", "mrr", "map",
    "learning to rank", "re-ranking",
    # LLM / ML
    "hugging face transformers", "huggingface transformers",
    "langchain", "haystack", "fine-tuning llms", "fine tuning",
    "lora", "qlora", "peft", "mlops", "mlflow",
    "pytorch", "tensorflow", "machine learning", "deep learning",
    "nlp", "natural language processing",
    # Infrastructure
    "python", "production ml", "bentoml", "kubeflow",
}

NICE_TO_HAVE = {
    "distributed systems", "large scale", "inference optimization",
    "open source", "a/b testing", "spark", "kafka", "docker", "kubernetes",
    "aws", "gcp", "azure", "data pipelines", "airflow",
    "go", "rust", "scala",
}

# Title-based negative signals (these candidates are unlikely fits)
NEGATIVE_TITLE_KEYWORDS = {
    "marketing manager", "sales", "content writer", "accountant",
    "civil engineer", "mechanical engineer", "hr manager", "customer support",
    "graphic design", "operations manager",
}

# Consulting-only signal (penalize if ENTIRE career is consulting)
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "l&t infotech", "persistent systems", "cyient",
}

# India locations preferred by JD
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "ncr", "bangalore", "bengaluru", "new delhi",
}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_candidates(path: str):
    """Load candidates from .jsonl or .jsonl.gz file."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: File not found: {path}")

    candidates = []
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"

    print(f"Loading candidates from {p.name} ...", flush=True)
    with opener(p, mode, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            if (i + 1) % 20000 == 0:
                print(f"  Loaded {i+1:,} ...", flush=True)

    print(f"  Total: {len(candidates):,} candidates loaded.", flush=True)
    return candidates


# ─── Feature Extraction ───────────────────────────────────────────────────────

def parse_date_safe(s):
    """Parse date string safely, return date or None."""
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def days_since(d):
    """Days between a date and today. Returns large number if None."""
    if d is None:
        return 999
    return max(0, (TODAY - d).days)


def skill_score(skills: list) -> tuple:
    """
    Returns (score 0-1, matched_core, matched_nice, skill_text_for_reasoning).
    Uses actual skill names from the dataset.
    """
    prof_weights = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.65, "beginner": 0.4}

    matched_core = []
    matched_nice = []
    all_skill_names = []

    for s in skills:
        name_lower = s["name"].lower()
        name_orig = s["name"]
        prof = prof_weights.get(s.get("proficiency", "beginner"), 0.4)
        dur = s.get("duration_months", 0)
        end = s.get("endorsements", 0)

        # Duration weight (12 months = 1.0, 0 months = 0.3)
        dur_w = 0.3 + 0.7 * min(dur / 12, 1.0)

        all_skill_names.append(name_orig)

        # Match against core required
        for core in CORE_REQUIRED:
            if core in name_lower or name_lower in core:
                matched_core.append((name_orig, prof * dur_w, end))
                break
        else:
            for nice in NICE_TO_HAVE:
                if nice in name_lower or name_lower in nice:
                    matched_nice.append((name_orig, prof * dur_w * 0.5))
                    break

    # Score: core hits weighted by proficiency+duration, nice-to-have at half weight
    core_score = min(sum(v for _, v, _ in matched_core) / max(len(CORE_REQUIRED) * 0.3, 1), 1.0)
    nice_score = min(sum(v for _, v in matched_nice) / 5, 1.0)

    final = 0.75 * core_score + 0.25 * nice_score

    top_skills = sorted(matched_core, key=lambda x: -x[1])[:4]
    skill_names = [n for n, _, _ in top_skills] or all_skill_names[:3]

    return round(final, 4), [n for n, _, _ in matched_core], skill_names


def career_dna_score(candidate: dict) -> float:
    """
    6-axis Career DNA. Adapted for real dataset schema.
    Returns 0-1.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    yoe = profile.get("years_of_experience", 0)

    # 1. Skill Depth: YOE + seniority signal from title
    title = profile.get("current_title", "").lower()
    seniority_bonus = 0.0
    if any(k in title for k in ["principal", "staff", "architect", "head", "director", "vp", "lead"]):
        seniority_bonus = 0.2
    elif any(k in title for k in ["senior", "sr."]):
        seniority_bonus = 0.1
    elif any(k in title for k in ["junior", "jr.", "intern", "trainee"]):
        seniority_bonus = -0.15
    depth = min((yoe / 10) * 0.8 + 0.2 + seniority_bonus, 1.0)

    # 2. Growth velocity: number of companies, title progression
    if career:
        unique_co = len(set(c["company"] for c in career))
        tenures = [c.get("duration_months", 12) for c in career]
        avg_tenure = sum(tenures) / len(tenures)
        velocity = min(unique_co / 4, 1.0) * 0.5 + (
            1.0 if 18 <= avg_tenure <= 36 else 0.6 if avg_tenure >= 12 else 0.3
        ) * 0.5
    else:
        velocity = 0.4

    # 3. Product company experience (key JD requirement)
    is_consulting = lambda co: any(c in co.lower() for c in CONSULTING_COMPANIES)
    if career:
        product_jobs = sum(1 for c in career if not is_consulting(c["company"]))
        product_ratio = product_jobs / len(career)
    else:
        product_ratio = 0.5

    # 4. Tenure stability
    if career:
        tenures = [c.get("duration_months", 12) for c in career]
        short = sum(1 for t in tenures if t < 12)
        stability = max(0.0, 1.0 - (short / len(tenures)) * 0.8)
    else:
        stability = 0.5

    # 5. Recency: is current role AI/ML related?
    current = next((c for c in career if c.get("is_current")), None)
    if current:
        desc_lower = (current.get("description", "") + current.get("title", "")).lower()
        ai_terms = ["ml", "machine learning", "ai", "nlp", "embedding", "retrieval",
                    "ranking", "vector", "llm", "recommendation", "search"]
        recency = 0.8 if any(t in desc_lower for t in ai_terms) else 0.4
    else:
        recency = 0.4

    # 6. YOE range fit (JD wants 5-9 years, but judgment matters more)
    if 5 <= yoe <= 9:
        yoe_fit = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        yoe_fit = 0.8
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        yoe_fit = 0.6
    else:
        yoe_fit = 0.4

    score = (
        0.20 * depth
        + 0.20 * velocity
        + 0.25 * product_ratio  # most important: product vs consulting
        + 0.15 * stability
        + 0.10 * recency
        + 0.10 * yoe_fit
    )
    return round(min(max(score, 0.0), 1.0), 4)


def availability_score(signals: dict) -> float:
    """
    Compute availability from all 23 Redrob signals.
    Handles -1 sentinels for offer_acceptance_rate and github_activity_score.
    """
    # 1. Activity recency (exponential decay over 180 days)
    last_active = parse_date_safe(signals.get("last_active_date"))
    days_inactive = days_since(last_active)
    activity = max(0.0, 1.0 - days_inactive / 180)

    # 2. Open to work
    otw = 1.0 if signals.get("open_to_work_flag") else 0.4

    # 3. Recruiter response rate (0-1)
    rr = float(signals.get("recruiter_response_rate", 0.5))

    # 4. Notice period
    np_days = int(signals.get("notice_period_days", 90))
    if np_days <= 30:
        notice = 1.0
    elif np_days <= 60:
        notice = 0.8
    elif np_days <= 90:
        notice = 0.6
    else:
        notice = 0.3

    # 5. Interview completion rate
    ic = float(signals.get("interview_completion_rate", 0.5))

    # 6. Avg response time (lower = better; >48h = penalty)
    rt = float(signals.get("avg_response_time_hours", 24))
    response_speed = 1.0 if rt <= 4 else (0.8 if rt <= 24 else 0.6 if rt <= 72 else 0.3)

    # 7. Offer acceptance rate (-1 sentinel = no prior offers → neutral)
    oar = float(signals.get("offer_acceptance_rate", -1))
    offer_score = 0.5 if oar == -1 else oar  # neutral if no history

    # 8. Profile completeness (0-100 → 0-1)
    completeness = float(signals.get("profile_completeness_score", 50)) / 100

    score = (
        0.25 * activity
        + 0.20 * otw
        + 0.18 * rr
        + 0.15 * notice
        + 0.10 * ic
        + 0.06 * response_speed
        + 0.04 * offer_score
        + 0.02 * completeness
    )
    return round(min(max(score, 0.0), 1.0), 4)


def location_score(profile: dict) -> float:
    """
    JD prefers Pune/Noida, also accepts Hyderabad/Mumbai/Delhi NCR.
    Must check country=='India' first since location field is city-only.
    """
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()

    if country != "india":
        # Outside India: JD says "case-by-case, no visa sponsorship"
        return 0.25

    for preferred in PREFERRED_LOCATIONS:
        if preferred in location:
            return 1.0 if any(p in location for p in ["pune", "noida"]) else 0.85

    # Other Indian cities
    return 0.65


def is_honeypot(candidate: dict) -> bool:
    """
    Detect candidates with impossible profiles.
    Returns True if candidate should be scored 0.
    Dataset has ~80 honeypots out of 100K.
    """
    profile = candidate["profile"]
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    yoe = profile.get("years_of_experience", 0)

    # Check 1: Expert/advanced skill with 0 months duration
    zero_dur_expert = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 1) == 0
    )
    if zero_dur_expert >= 3:
        return True

    # Check 2: Many expert/advanced skills AND several have 0 duration = impossible
    zero_dur_adv = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 1) == 0
    )
    expert_count = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
    if expert_count > 10 and zero_dur_adv >= 2:
        return True

    # Check 3: YOE much less than total career months (impossible timeline)
    if career:
        total_career_months = sum(c.get("duration_months", 0) for c in career)
        career_years = total_career_months / 12
        if career_years > yoe + 3:  # more than 3 year discrepancy
            return True

    # Check 4: Duration > years of experience by large margin
    if yoe > 0:
        max_possible_months = yoe * 12 + 6  # small buffer for overlaps
        total_months = sum(c.get("duration_months", 0) for c in career)
        if total_months > max_possible_months * 1.5:
            return True

    return False


def build_candidate_text(candidate: dict) -> str:
    """Build rich text representation for TF-IDF matching."""
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]

    # Skills — repeat high-proficiency ones for TF-IDF weight
    for s in skills:
        name = s["name"]
        prof = s.get("proficiency", "beginner")
        parts.append(name)
        if prof in ("expert", "advanced"):
            parts.append(name)  # double weight
        if prof == "expert":
            parts.append(name)  # triple weight for expert

    # Career descriptions (most recent 3)
    sorted_career = sorted(career, key=lambda c: c.get("start_date") or "", reverse=True)
    for c in sorted_career[:3]:
        parts.append(c.get("title", ""))
        parts.append(c.get("description", "")[:300])

    # Education
    for e in education:
        parts.append(e.get("field_of_study", ""))

    return " ".join(filter(None, parts))


def build_reasoning(candidate: dict, skill_names: list, avail: float, loc: float, dna: float) -> str:
    """
    Build a specific 1-2 sentence reasoning. Never hallucinate.
    Only mentions skills actually in the candidate's profile.
    """
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = f"{profile.get('location','')}, {profile.get('country','')}"

    # Career note
    current = next((c for c in career if c.get("is_current")), None)
    if current:
        career_note = f"currently {current['title']} at {current['company']}"
    else:
        career_note = f"{title} at {company}"

    # Availability note
    otw = signals.get("open_to_work_flag", False)
    rr = signals.get("recruiter_response_rate", 0)
    notice = signals.get("notice_period_days", 90)
    last_active = signals.get("last_active_date", "")

    avail_parts = []
    if otw:
        avail_parts.append("open to work")
    if notice <= 30:
        avail_parts.append(f"{notice}d notice")
    elif notice <= 60:
        avail_parts.append(f"{notice}d notice")
    else:
        avail_parts.append(f"{notice}d notice (long)")
    if rr < 0.2:
        avail_parts.append(f"low response rate ({rr:.0%})")
    avail_str = "; ".join(avail_parts) if avail_parts else "availability unknown"

    # Skills
    top_skills = ", ".join(skill_names[:3]) if skill_names else "no strong AI/ML skill match"

    sentence1 = f"{yoe:.0f}y exp as {title}; {career_note}; core skills: {top_skills}."
    sentence2 = f"{avail_str}; loc: {location}."

    return f"{sentence1} {sentence2}"


# ─── Main Ranking Pipeline ────────────────────────────────────────────────────

def rank_candidates(candidates: list) -> list:
    """
    Full pipeline: returns list of dicts ready for CSV output.
    """
    n = len(candidates)
    print(f"\nScoring {n:,} candidates ...", flush=True)

    # ── Step 1: Build TF-IDF matrix ──────────────────────────────────────────
    print("  Step 1/4: Building TF-IDF index ...", flush=True)
    texts = [build_candidate_text(c) for c in candidates]
    texts.append(JD_TEXT)  # JD is last

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    jd_vector = tfidf_matrix[-1]
    candidate_matrix = tfidf_matrix[:-1]

    # Cosine similarities in batches (memory efficient)
    print("  Step 2/4: Computing semantic scores ...", flush=True)
    batch = 5000
    semantic_scores = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sims = cosine_similarity(candidate_matrix[start:end], jd_vector)
        semantic_scores[start:end] = sims.ravel()

    # ── Step 2: Per-candidate scoring ────────────────────────────────────────
    print("  Step 3/4: Computing ensemble scores ...", flush=True)
    results = []

    for i, candidate in enumerate(candidates):
        if (i + 1) % 25000 == 0:
            print(f"    {i+1:,}/{n:,} ...", flush=True)

        # Honeypot check first
        if is_honeypot(candidate):
            results.append({
                "candidate_id": candidate["candidate_id"],
                "score": 0.0,
                "reasoning": "Profile contains inconsistencies inconsistent with stated experience.",
                "skill_names": [],
            })
            continue

        profile = candidate["profile"]
        signals = candidate.get("redrob_signals", {})

        sem = float(semantic_scores[i])
        skill_s, matched_core, skill_names = skill_score(candidate.get("skills", []))
        dna_s = career_dna_score(candidate)
        avail_s = availability_score(signals)
        loc_s = location_score(profile)

        # Title mismatch penalty — if title is completely off-domain
        title_lower = profile.get("current_title", "").lower()
        if any(neg in title_lower for neg in NEGATIVE_TITLE_KEYWORDS):
            title_penalty = 0.55  # stronger penalty for truly off-domain titles
        elif any(k in title_lower for k in ["project manager", "program manager", "delivery manager"]):
            title_penalty = 0.80  # moderate penalty: PM can transition but is not ideal
        else:
            title_penalty = 1.0

        # Consulting-only penalty (entire career in consulting = penalize)
        career = candidate.get("career_history", [])
        if career:
            all_consulting = all(
                any(co in c["company"].lower() for co in CONSULTING_COMPANIES)
                for c in career
            )
            consulting_penalty = 0.75 if all_consulting else 1.0
        else:
            consulting_penalty = 1.0

        raw_score = (
            0.35 * sem
            + 0.25 * skill_s
            + 0.20 * dna_s
            + 0.15 * avail_s
            + 0.05 * loc_s
        )

        final_score = raw_score * title_penalty * consulting_penalty

        reasoning = build_reasoning(candidate, skill_names, avail_s, loc_s, dna_s)

        results.append({
            "candidate_id": candidate["candidate_id"],
            "score": round(final_score, 4),
            "reasoning": reasoning,
            "skill_names": skill_names,
        })

    # ── Step 3: Sort and top-100 ──────────────────────────────────────────────
    print("  Step 4/4: Selecting top 100 ...", flush=True)
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    top100 = results[:100]

    # Assign ranks and ensure score is non-increasing
    # Handle ties: same score → same (or next) rank, tie-break by candidate_id ascending
    ranked = []
    for rank, r in enumerate(top100, 1):
        ranked.append({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": r["score"],
            "reasoning": r["reasoning"],
        })

    return ranked


# ─── Output ──────────────────────────────────────────────────────────────────

def write_csv(ranked: list, out_path: str):
    """Write submission CSV."""
    p = Path(out_path)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(ranked)
    print(f"\n✅ Submission written: {p.resolve()}")
    print(f"   Rows: {len(ranked)}")
    if ranked:
        print(f"   Score range: {ranked[-1]['score']:.4f} – {ranked[0]['score']:.4f}")
        print(f"   Top 3:")
        for r in ranked[:3]:
            print(f"     #{r['rank']} {r['candidate_id']}  score={r['score']:.4f}  {r['reasoning'][:80]}...")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon — Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path (default: submission.csv)")
    args = parser.parse_args()

    t0 = datetime.now()
    candidates = load_candidates(args.candidates)
    ranked = rank_candidates(candidates)
    write_csv(ranked, args.out)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n⏱  Total time: {elapsed:.1f}s")

    if elapsed > 300:
        print("⚠️  WARNING: Exceeded 5-minute budget. Optimise batch size or feature extraction.")


if __name__ == "__main__":
    main()