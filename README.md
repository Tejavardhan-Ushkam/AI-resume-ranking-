# 🧠 Intelligent Candidate Discovery

> **Hackathon Submission** — Data & AI Challenge: Intelligent Candidate Discovery  
> A novel multi-signal AI system that intelligently ranks candidates using semantic understanding, Career DNA fingerprinting, and ensemble scoring.

---

## 🏆 What Makes This Different

| Feature | Typical Approach | Our Approach |
|---|---|---|
| Job Understanding | Keyword extraction | **Deep JD decomposition** — skills, seniority, culture, implicit needs |
| Matching | TF-IDF cosine similarity | **Ensemble:** semantic + skills + Career DNA + behavioral |
| Candidate Representation | Resume text blob | **Career DNA** — 6-axis fingerprint (depth, velocity, breadth, stability, recency, leadership) |
| Ranking Weights | Fixed | **Adaptive** — weights shift based on JD type (IC vs Manager vs Lead) |
| Explainability | Score only | **Per-candidate AI reasoning** — strengths, risks, and narrative |
| Speed | Batch-only | **ANN vector search** (ChromaDB) for sub-second retrieval |

---

## 🧬 Career DNA — The Novel Differentiator

Every candidate gets a **6-axis Career DNA fingerprint**:

```
skill_depth_score   — Seniority within skill domains (0–1)
growth_velocity     — Career progression speed (0–1)  
domain_breadth      — Versatility across tech domains (0–1)
tenure_stability    — Job tenure reliability (0–1)
recency_score       — Freshness of relevant experience (0–1)
collaboration_signal — Leadership & teamwork signals (0–1)
```

This fingerprint is **matched against a JD-specific target vector** rather than a generic embedding, enabling axis-aware ranking (e.g., a Director role weights `collaboration_signal` more; an IC role weights `skill_depth_score` more).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    REACT FRONTEND                         │
│  Job Input → JD Analysis Panel → Ranked Candidate Cards  │
│  CareerDNA Radar Charts + Score Breakdowns + AI Summaries│
└─────────────────────────┬───────────────────────────────┘
                          │ REST API (proxied via Vite)
┌─────────────────────────▼───────────────────────────────┐
│                  FASTAPI BACKEND                          │
│                                                          │
│  POST /api/search/                                       │
│     1. JD Analyzer   → structured decomposition          │
│     2. Embed JD      → sentence-transformers (local)     │
│     3. ANN Search    → ChromaDB cosine similarity        │
│     4. Ensemble Rank → semantic+skills+DNA+behavioral    │
│     5. AI Explainer  → Claude Haiku (or rule fallback)   │
│                                                          │
│  Storage:                                                │
│    SQLite → candidate metadata + JD history              │
│    ChromaDB → embedding vectors (HNSW index)             │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
intelligent-candidate-discovery/
├── backend/
│   ├── main.py                    # FastAPI app, lifespan, CORS
│   ├── config.py                  # Settings (env-based)
│   ├── requirements.txt
│   ├── seed.py                    # Load demo candidates
│   ├── api/
│   │   └── routes/
│   │       ├── candidates.py      # CRUD, bulk upload
│   │       └── search.py          # Core ranking endpoint
│   ├── core/
│   │   ├── jd_analyzer.py         # Deep JD decomposition (AI + heuristics)
│   │   ├── signal_engine.py       # Embeddings, ChromaDB, skills overlap
│   │   ├── career_dna.py          # 6-axis candidate fingerprinting ★ NOVEL
│   │   ├── ranker.py              # Ensemble scoring + explanation generation
│   │   └── explainer.py           # AI-generated candidate summaries
│   ├── models/
│   │   ├── schemas.py             # Pydantic models
│   │   └── database.py            # SQLite + ChromaDB wrappers
│   └── data/
│       └── seed_candidates.json   # 20 realistic demo candidates
└── frontend/
    ├── package.json
    ├── vite.config.ts             # Dev server + API proxy
    ├── tailwind.config.js
    └── src/
        ├── App.tsx                # Main layout
        ├── index.css
        ├── services/
        │   └── api.ts             # Typed API client
        └── components/
            ├── JobInputPanel.tsx      # JD input + example JDs
            ├── CandidateCard.tsx      # Rich candidate result card
            ├── CareerDNAChart.tsx     # Radar chart (Recharts)
            ├── ScoreBreakdownBar.tsx  # 4-dimension score bars
            └── JDInsightsPanel.tsx    # JD decomposition display
```

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+
- Node.js 18+

### Step 1: Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set Anthropic API key for AI-powered JD analysis + summaries
# Without this, heuristic fallbacks are used — still works great!
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...
```

### Step 2: Seed Demo Candidates

```bash
# Still in backend/ with venv active
python seed.py
```

Expected output:
```
🌱 Seeding candidate database...
Loading embedding model...
  ✅ [1/20] Priya Sharma — Career DNA: depth=0.73, velocity=0.72, breadth=0.83
  ✅ [2/20] Rahul Verma — Career DNA: depth=0.85, velocity=0.78, breadth=0.73
  ...
✨ Done! 20 candidates indexed, 0 errors.
📊 ChromaDB has 20 vectors ready.
```

### Step 3: Start Backend

```bash
# Still in backend/
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000  
API docs (Swagger): http://localhost:8000/docs  
Health check: http://localhost:8000/health

### Step 4: Start Frontend

```bash
# New terminal
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:3000

---

## 🎯 Usage

1. Open http://localhost:3000
2. Click **"Examples"** to load a pre-written job description, OR paste your own
3. Click **"Discover Candidates"**
4. View ranked candidates with:
   - Match percentage + rank medal
   - Career DNA radar chart
   - 4-dimension score breakdown
   - Matched vs missing skills
   - AI-generated strengths, risks, and summary

---

## 🔌 API Reference

### Search Candidates
```bash
POST /api/search/
Content-Type: application/json

{
  "job_description": "We are looking for a Senior ML Engineer...",
  "job_title": "Senior ML Engineer",
  "top_k": 10,
  "filters": {
    "min_years_experience": 3
  }
}
```

### Bulk Upload Candidates
```bash
POST /api/candidates/bulk
Content-Type: application/json

[
  {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "current_title": "Senior Engineer",
    "location": "Bengaluru",
    "years_of_experience": 6,
    "skills": ["python", "aws", "kubernetes"],
    "summary": "...",
    "work_experience": [...],
    "education": [...],
    "behavioral_signals": {
      "profile_completeness": 0.9,
      "open_to_work": true,
      ...
    }
  }
]
```

---

## 🧪 Running Without API Key

The system works **fully offline without any API key**:
- Embeddings: `all-MiniLM-L6-v2` (local, ~80MB, downloaded once)
- JD Analysis: Heuristic regex + keyword extraction
- Candidate Summaries: Rule-based narrative generation

Add `ANTHROPIC_API_KEY` to `.env` to upgrade to Claude-powered analysis.

---

## 🌱 .env Example

```env
# Optional - enables Claude-powered JD analysis and AI summaries
ANTHROPIC_API_KEY=sk-ant-...

# Ranking weight tuning (optional)
WEIGHT_SEMANTIC=0.35
WEIGHT_SKILLS=0.25
WEIGHT_CAREER_DNA=0.25
WEIGHT_BEHAVIORAL=0.15
```

---

## 🏆 Novelty Summary

1. **Career DNA Fingerprinting** — Moves beyond text similarity to a structured multi-axis candidate model
2. **Adaptive Ensemble Weights** — Ranking weights dynamically adjust based on JD role type and seniority
3. **Semantic JD Enrichment** — JD is decomposed and re-enriched before embedding (not raw text → raw embedding)
4. **Explainable Rankings** — Every result comes with human-readable AI reasoning, not just a score
5. **Graceful Degradation** — Works offline with heuristics; upgrades with AI when API key available


# Intelligent Candidate Discovery

## Hackathon Submission — Quick Start

### Produce the submission CSV
```bash
cd backend
pip install -r requirements.txt
python rank.py --candidates /path/to/candidates.jsonl --out submission.csv
# Or with gzip:
python rank.py --candidates /path/to/candidates.jsonl.gz --out submission.csv
python validate_and_submit.py submission.csv
```

### Run the sandbox demo
```bash
cd backend
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### Full app (backend + frontend)
```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python seed.py          # first time only
uvicorn main:app --reload --port 8000
# API at http://localhost:8000  |  Docs at http://localhost:8000/docs

# Frontend (new terminal)
cd frontend
npm install && npm run dev
# Opens at http://localhost:3000
```

---

## About

A multi-signal AI ranking system for intelligent candidate discovery, built for the Redrob Hackathon. The system combines:

- **Semantic matching** (TF-IDF bigrams, cosine similarity against job description)
- **Skill overlap scoring** (24 core required + 14 nice-to-have skills, weighted by proficiency, endorsements, and duration)
- **Career DNA fingerprinting** (6-axis trajectory model: skill depth, growth velocity, domain breadth, tenure stability, recency, collaboration)
- **Behavioral signals** (23 Redrob platform signals grouped into activity, recruiter fit, profile quality, engagement)
- **Honeypot detection** (5-check system flagging impossible profiles to 0)

### Performance
- Ranks 100,000 candidates in **<3 minutes** on a CPU-only machine
- Peak memory: **~3–4 GB** (well within 16 GB constraint)
- Zero LLM API calls during ranking; no GPU required

### Scoring Formula
```
score = 0.35 × semantic_score
      + 0.25 × skill_match_score
      + 0.20 × career_dna_score
      + 0.15 × availability_score
      + 0.05 × location_score
```

### Project Structure
```
intelligent-candidate-discovery/
├── README.md
├── submission_metadata.yaml
├── backend/
│   ├── rank.py                 ← Standalone ranker (100K candidates → top-100 CSV)
│   ├── streamlit_app.py        ← Sandbox demo (≤100 candidates)
│   ├── validate_and_submit.py  ← Submission validator
│   ├── requirements.txt
│   ├── main.py                 ← FastAPI app
│   ├── seed.py
│   ├── config.py
│   ├── api/
│   ├── core/
│   │   ├── career_dna.py
│   │   ├── ranker.py
│   │   ├── signal_engine.py
│   │   ├── jd_analyzer.py
│   │   └── explainer.py
│   └── models/
└── presentation/
    └── slides.tex              ← Compile with: pdflatex slides.tex
```

---

## Submission Checklist

```
[ ] python rank.py runs in <5 min on CPU
[ ] python validate_and_submit.py submission.csv → "Submission is valid."
[ ] submission.csv has exactly 100 rows + 1 header
[ ] Scores are non-increasing
[ ] No duplicate candidate_ids or ranks
[ ] streamlit run streamlit_app.py works locally
[ ] Deployed to HuggingFace Spaces or Streamlit Cloud
[ ] pdflatex slides.tex compiles without errors
[ ] submission_metadata.yaml updated with real team name, IDs, sandbox URL
```