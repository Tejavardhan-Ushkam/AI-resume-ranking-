#!/usr/bin/env python3
"""
Seed script: loads demo candidates into the system.
Run: python seed.py
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.database import init_db, get_chroma
from models.schemas import CandidateCreate


def seed():
    print("🌱 Seeding candidate database...")
    
    # Init DB
    init_db()
    get_chroma()
    
    # Load seed data
    seed_path = os.path.join(os.path.dirname(__file__), "data", "seed_candidates.json")
    with open(seed_path) as f:
        candidates_raw = json.load(f)

    print(f"Loading embedding model...")
    from core.signal_engine import embed_text
    embed_text("warmup")  # warm up
    
    from api.routes.candidates import _create_candidate_full

    success = 0
    errors = 0
    
    for i, cdata in enumerate(candidates_raw):
        try:
            candidate = CandidateCreate(**cdata)
            result = _create_candidate_full(candidate)
            print(f"  ✅ [{i+1}/{len(candidates_raw)}] {result.name} — Career DNA: "
                  f"depth={result.career_dna.skill_depth_score:.2f}, "
                  f"velocity={result.career_dna.growth_velocity:.2f}, "
                  f"breadth={result.career_dna.domain_breadth:.2f}")
            success += 1
        except Exception as e:
            print(f"  ❌ [{i+1}] {cdata.get('name', '?')}: {e}")
            errors += 1

    print(f"\n✨ Done! {success} candidates indexed, {errors} errors.")
    print(f"📊 ChromaDB has {get_chroma().count()} vectors ready.")
    print(f"\nNow run: uvicorn main:app --reload")


if __name__ == "__main__":
    seed()