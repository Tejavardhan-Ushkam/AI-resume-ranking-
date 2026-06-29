# 🧠 Intelligent Candidate Discovery

> **Hackathon Submission** — Data & AI Challenge: Intelligent Candidate Discovery  
> A novel multi-signal AI system that intelligently ranks candidates using semantic understanding, Career DNA fingerprinting, and ensemble scoring.

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+
- Node.js 18+
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


## 🏆 Novelty Summary

1. **Career DNA Fingerprinting** — Moves beyond text similarity to a structured multi-axis candidate model
2. **Adaptive Ensemble Weights** — Ranking weights dynamically adjust based on JD role type and seniority
3. **Semantic JD Enrichment** — JD is decomposed and re-enriched before embedding (not raw text → raw embedding)
4. **Explainable Rankings** — Every result comes with human-readable AI reasoning, not just a score
5. **Graceful Degradation** — Works offline with heuristics; upgrades with AI when API key available

