"""
streamlit_app.py — Redrob Hackathon Sandbox Demo

Run: streamlit run streamlit_app.py
Accepts ≤100 candidates via file upload or paste, ranks them, outputs CSV.
"""

import csv
import io
import json
import math
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Import shared logic from rank.py ─────────────────────────────────────────
# (Inline reimplementation so the app is self-contained for HuggingFace Spaces)

JD_TEXT = """
Senior AI Engineer at Redrob AI Series A startup Pune Noida India hybrid.
5 to 9 years of experience required.
Must have: production embeddings retrieval sentence-transformers BGE E5 embeddings,
vector database Pinecone Weaviate Qdrant FAISS Milvus OpenSearch Elasticsearch,
hybrid search BM25, strong Python, ranking evaluation NDCG MRR MAP,
information retrieval recommendation systems search systems production ML deployment.
Nice to have: LLM fine-tuning LoRA QLoRA PEFT learning to rank LightGBM XGBoost
neural ranking HR tech recruiting distributed systems inference optimization open source RAG NLP transformers BERT MLOps.
Not wanted: pure consulting TCS Infosys Wipro Accenture Cognizant Capgemini,
computer vision only speech robotics without NLP, job hopping, marketing sales HR manager roles.
"""

CORE_REQUIRED = [
    "embeddings", "sentence-transformers", "vector database", "pinecone", "weaviate",
    "qdrant", "faiss", "milvus", "opensearch", "elasticsearch", "retrieval",
    "ranking", "ndcg", "mrr", "map", "hybrid search", "bm25",
    "python", "recommendation system", "search", "rag",
]
NICE_TO_HAVE = [
    "lora", "qlora", "fine-tuning", "peft", "learning to rank", "xgboost",
    "lightgbm", "hr tech", "distributed systems", "nlp", "bert", "huggingface",
    "mlops", "pytorch",
]
CONSULTING_ONLY = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
TODAY = date.today()


def parse_date_safe(s):
    if not s:
        return None
    try:
        return dateparser.parse(str(s)).date()
    except Exception:
        return None


def days_since(d):
    if d is None:
        return 9999
    return max(0, (TODAY - d).days)


def is_honeypot(c):
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    zero_dur_expert = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 1) == 0
    )
    if zero_dur_expert > 8:
        return True
    stated_yoe = float(profile.get("years_of_experience") or 0)
    total_months = sum(int(r.get("duration_months") or 0) for r in career)
    if total_months / 12 > stated_yoe + 4:
        return True
    for role in career:
        if int(role.get("duration_months") or 0) > 240:
            return True
    return False


def availability_score(signals):
    if not signals:
        return 0.2
    last_active = parse_date_safe(signals.get("last_active_date"))
    activity = max(0.0, 1.0 - days_since(last_active) / 180.0)
    otw = 1.0 if signals.get("open_to_work_flag") else 0.4
    rr = min(1.0, max(0.0, float(signals.get("recruiter_response_rate") or 0)))
    np_days = int(signals.get("notice_period_days") or 60)
    notice = 1.0 if np_days <= 30 else (0.8 if np_days <= 60 else (0.5 if np_days <= 90 else 0.2))
    ic = min(1.0, max(0.0, float(signals.get("interview_completion_rate") or 0.5)))
    return 0.30 * activity + 0.25 * otw + 0.20 * rr + 0.15 * notice + 0.10 * ic


def location_score(c):
    signals = c.get("redrob_signals") or {}
    profile = c.get("profile") or {}
    willing = signals.get("willing_to_relocate", False)
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    if any(p in loc for p in ["pune", "noida"]):
        return 1.0
    if any(p in loc for p in ["hyderabad", "mumbai", "delhi", "bengaluru", "bangalore", "gurgaon"]):
        return 0.85
    if "india" in country or "india" in loc:
        return 0.7 if willing else 0.5
    return 0.3 if willing else 0.1


def build_skill_map(c):
    skill_map = {}
    for s in c.get("skills") or []:
        name = (s.get("name") or "").lower()
        prof = s.get("proficiency") or "beginner"
        prof_w = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.6, "beginner": 0.3}.get(prof, 0.4)
        end_w = math.log1p(min(int(s.get("endorsements") or 0), 100)) / math.log1p(100)
        dur_w = math.log1p(min(int(s.get("duration_months") or 0), 120)) / math.log1p(120)
        skill_map[name] = max(skill_map.get(name, 0.0), 0.5 * prof_w + 0.25 * end_w + 0.25 * dur_w)
    for role in c.get("career_history") or []:
        desc = (role.get("description") or "").lower()
        for kw in CORE_REQUIRED:
            if kw in desc and kw not in skill_map:
                skill_map[kw] = 0.25
    return skill_map


def skill_match_score(skill_map):
    core_score = sum(
        max((w for sk, w in skill_map.items() if kw in sk or sk in kw), default=0.0)
        for kw in CORE_REQUIRED
    )
    nice_score = sum(
        max((w * 0.5 for sk, w in skill_map.items() if kw in sk or sk in kw), default=0.0)
        for kw in NICE_TO_HAVE
    )
    raw = (core_score / len(CORE_REQUIRED)) * 0.75 + (nice_score / len(NICE_TO_HAVE)) * 0.25
    return min(1.0, raw)


def career_dna_score(c):
    profile = c.get("profile") or {}
    career = c.get("career_history") or []
    skills = c.get("skills") or []
    signals = c.get("redrob_signals") or {}
    skill_depth = (sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced")) / max(len(skills), 1))
    growth_velocity = min(1.0, 0.4 + 0.1 * len(career))
    is_consulting_heavy = sum(
        1 for r in career if any(cn in (r.get("company") or "").lower() for cn in CONSULTING_ONLY)
    ) > len(career) * 0.7 if career else False
    domain_breadth = 0.2 if is_consulting_heavy else min(1.0, 0.5 + 0.1 * len(set(r.get("industry", "") for r in career)))
    tenures = [int(r.get("duration_months") or 0) for r in career if r.get("duration_months")]
    avg_tenure = sum(tenures) / len(tenures) if tenures else 24
    tenure_stability = 1.0 if 18 <= avg_tenure <= 48 else (0.3 if avg_tenure < 12 else 0.6)
    ct = (profile.get("current_title") or "").lower()
    cc = (profile.get("current_company") or "").lower()
    recency = min(1.0, (0.6 if any(t in ct for t in ["engineer", "scientist", "ml", "ai", "nlp", "search"]) else 0.2)
                  + (0.4 if not any(cn in cc for cn in CONSULTING_ONLY) else 0.1))
    collab = min(1.0, 0.4 * float(signals.get("github_activity_score") or 0) / 10.0
                 + 0.6 * min(1.0, int(signals.get("connection_count") or 0) / 500))
    yoe = float(profile.get("years_of_experience") or 0)
    yoe_factor = 1.0 if 5 <= yoe <= 9 else (0.85 if 4 <= yoe < 5 else (0.9 if 9 < yoe <= 12 else 0.6))
    raw = 0.20*skill_depth + 0.20*growth_velocity + 0.15*domain_breadth + 0.15*tenure_stability + 0.20*recency + 0.10*collab
    return min(1.0, raw * yoe_factor)


def build_text(c):
    profile = c.get("profile") or {}
    parts = ([profile.get("headline", "")] * 3 + [profile.get("current_title", "")] * 3
             + [profile.get("summary", "")])
    for s in c.get("skills") or []:
        parts += [s.get("name", "")] * (3 if s.get("proficiency") in ("expert", "advanced") else 1)
    for r in c.get("career_history") or []:
        parts += [r.get("title", ""), r.get("description", "")]
    return " ".join(filter(None, parts))


def run_ranking(candidates: List[Dict], jd_text: str) -> pd.DataFrame:
    n = len(candidates)
    if n == 0:
        return pd.DataFrame()

    corpus = [build_text(c) for c in candidates] + [jd_text]
    vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
    mat = vec.fit_transform(corpus)
    jd_vec = mat[-1]
    sims = cosine_similarity(mat[:-1], jd_vec).ravel()

    rows = []
    for i, c in enumerate(candidates):
        hp = is_honeypot(c)
        sm = build_skill_map(c)
        fs = (0.35 * sims[i] + 0.25 * skill_match_score(sm) + 0.20 * career_dna_score(c)
              + 0.15 * availability_score(c.get("redrob_signals") or {}) + 0.05 * location_score(c)) if not hp else 0.0
        profile = c.get("profile") or {}
        rows.append({
            "candidate_id": c.get("candidate_id", f"CAND_{i:07d}"),
            "score": round(float(fs), 6),
            "name": profile.get("anonymized_name", "—"),
            "title": profile.get("current_title", "—"),
            "company": profile.get("current_company", "—"),
            "yoe": profile.get("years_of_experience", 0),
            "location": profile.get("location", "—"),
            "honeypot": hp,
            "semantic": round(float(sims[i]), 4),
            "skill_match": round(skill_match_score(sm), 4),
            "career_dna": round(career_dna_score(c), 4),
            "availability": round(availability_score(c.get("redrob_signals") or {}), 4),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = io.StringIO()
    # submission format
    sub = df[["candidate_id", "rank", "score"]].copy()
    sub["reasoning"] = df.apply(
        lambda r: f"{r['yoe']:.0f}y exp as {r['title']} at {r['company']}; score={r['score']:.4f}.", axis=1
    )
    sub.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob — Intelligent Candidate Discovery",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Intelligent Candidate Discovery")
st.caption("Redrob Hackathon · Multi-Signal AI Ranking · CPU-only · No LLM calls during ranking")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📂 Upload Candidates")
    upload_mode = st.radio("Input mode", ["Upload JSONL file", "Paste JSON array"], horizontal=True)

    candidates = []
    if upload_mode == "Upload JSONL file":
        uploaded = st.file_uploader("Upload candidates.jsonl (≤100 candidates)", type=["jsonl", "json"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            candidates.extend(obj)
                        else:
                            candidates.append(obj)
                    except Exception:
                        pass
            if len(candidates) > 100:
                st.warning(f"Loaded {len(candidates)} candidates — trimming to first 100 for sandbox.")
                candidates = candidates[:100]
    else:
        pasted = st.text_area("Paste JSON array of candidates", height=200, placeholder='[{"candidate_id": "CAND_0000001", ...}]')
        if pasted.strip():
            try:
                parsed = json.loads(pasted)
                if isinstance(parsed, list):
                    candidates = parsed[:100]
                else:
                    candidates = [parsed]
            except Exception as e:
                st.error(f"JSON parse error: {e}")

    if candidates:
        st.success(f"✓ {len(candidates)} candidates loaded")

    st.subheader("📋 Job Description")
    jd_input = st.text_area("Job Description", value=JD_TEXT.strip(), height=200)

    run_btn = st.button("🚀 Run Ranking", type="primary", disabled=len(candidates) == 0)

with col_right:
    if run_btn and candidates:
        with st.spinner("Running ranking pipeline..."):
            results = run_ranking(candidates, jd_input)

        st.subheader(f"🏆 Top {min(len(results), 100)} Candidates")

        if not results.empty:
            honeypots = results[results["honeypot"]].shape[0]
            valid = results[~results["honeypot"]].shape[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Ranked", len(results))
            m2.metric("Valid", valid)
            m3.metric("Honeypots Detected", honeypots)
            m4.metric("Top Score", f"{results['score'].max():.4f}")

            display_cols = ["rank", "candidate_id", "name", "title", "company", "yoe", "location",
                            "score", "semantic", "skill_match", "career_dna", "availability", "honeypot"]
            st.dataframe(
                results[display_cols].style
                    .format({"score": "{:.4f}", "semantic": "{:.4f}", "skill_match": "{:.4f}",
                             "career_dna": "{:.4f}", "availability": "{:.4f}", "yoe": "{:.1f}"})
                    .background_gradient(subset=["score"], cmap="YlGn")
                    .map(lambda v: "background-color: #ffcccc" if v else "", subset=["honeypot"]),
                use_container_width=True,
                height=500,
            )

            csv_bytes = df_to_csv_bytes(results.head(100))
            st.download_button(
                "⬇️ Download submission.csv",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv",
            )

            with st.expander("📊 Score Distribution"):
                valid_scores = results[~results["honeypot"]]["score"]
                st.bar_chart(valid_scores.values[:50])
    elif not run_btn:
        st.info("⬅ Upload candidates and click **Run Ranking** to see results.")

st.divider()
st.caption(
    "Scoring: 35% TF-IDF semantic · 25% skill match · 20% Career DNA · 15% availability · 5% location  |  "
    "Honeypot detection: expert skills with 0 duration, YoE inconsistency, impossible tenures"
)