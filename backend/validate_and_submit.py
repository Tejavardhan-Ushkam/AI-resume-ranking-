"""
validate_and_submit.py — Validates submission.csv before portal submission.

Usage:
    python validate_and_submit.py submission.csv
"""

import csv
import re
import sys
from collections import Counter


EXPECTED_COLUMNS = {"candidate_id", "rank", "score", "reasoning"}
CAND_ID_PATTERN = re.compile(r"^CAND_\d{7}$")


def validate(path: str) -> bool:
    print(f"\n{'='*55}")
    print(f"  Redrob Hackathon — Submission Validator")
    print(f"{'='*55}")
    print(f"  File: {path}\n")

    errors = []
    warnings = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])

            # Check columns
            missing_cols = EXPECTED_COLUMNS - columns
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")
                print("[FAIL] Missing required columns:", missing_cols)
                return False

            rows = list(reader)
    except FileNotFoundError:
        print(f"[FAIL] File not found: {path}")
        return False
    except Exception as e:
        print(f"[FAIL] Could not read file: {e}")
        return False

    # ── Row count ─────────────────────────────────────────────────────────────
    n = len(rows)
    if n != 100:
        errors.append(f"Expected 100 rows, got {n}")
    else:
        print(f"[PASS] Row count: {n}")

    # ── Candidate ID format ───────────────────────────────────────────────────
    bad_ids = [r["candidate_id"] for r in rows if not CAND_ID_PATTERN.match(r.get("candidate_id", ""))]
    if bad_ids:
        errors.append(f"Invalid candidate_id format ({len(bad_ids)} rows): e.g. {bad_ids[:3]}")
    else:
        print(f"[PASS] All candidate_ids match CAND_XXXXXXX format")

    # ── Duplicate candidate IDs ───────────────────────────────────────────────
    cid_counts = Counter(r["candidate_id"] for r in rows)
    dupes = [k for k, v in cid_counts.items() if v > 1]
    if dupes:
        errors.append(f"Duplicate candidate_ids: {dupes[:5]}")
    else:
        print(f"[PASS] No duplicate candidate_ids")

    # ── Ranks ─────────────────────────────────────────────────────────────────
    try:
        ranks = [int(r["rank"]) for r in rows]
        expected_ranks = set(range(1, n + 1))
        actual_ranks = set(ranks)
        if actual_ranks != expected_ranks:
            missing = expected_ranks - actual_ranks
            extra = actual_ranks - expected_ranks
            errors.append(f"Rank issues — missing: {missing}, extra: {extra}")
        else:
            print(f"[PASS] Ranks 1–{n} all present, no duplicates")
    except ValueError as e:
        errors.append(f"Non-integer rank value: {e}")

    # ── Scores non-increasing ─────────────────────────────────────────────────
    try:
        # Sort by rank to check order
        sorted_rows = sorted(rows, key=lambda r: int(r["rank"]))
        scores = [float(r["score"]) for r in sorted_rows]
        violations = [(i + 1, i + 2, scores[i], scores[i + 1]) for i in range(len(scores) - 1) if scores[i] < scores[i + 1]]
        if violations:
            errors.append(f"Scores not non-increasing — {len(violations)} violations. First: rank {violations[0][0]} ({violations[0][2]:.4f}) < rank {violations[0][1]} ({violations[0][3]:.4f})")
        else:
            print(f"[PASS] Scores are non-increasing (rank 1 has highest score)")
    except ValueError as e:
        errors.append(f"Non-numeric score: {e}")

    # ── Scores in valid range ─────────────────────────────────────────────────
    try:
        out_of_range = [float(r["score"]) for r in rows if not (0.0 <= float(r["score"]) <= 1.0)]
        if out_of_range:
            warnings.append(f"Scores outside [0,1]: {out_of_range[:5]}")
        else:
            print(f"[PASS] All scores in [0.0, 1.0]")
    except ValueError:
        pass

    # ── Reasoning not empty ───────────────────────────────────────────────────
    empty_reasoning = [r["candidate_id"] for r in rows if not r.get("reasoning", "").strip()]
    if empty_reasoning:
        warnings.append(f"Empty reasoning for {len(empty_reasoning)} candidates: {empty_reasoning[:3]}")
    else:
        print(f"[PASS] All reasoning strings non-empty")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  SUMMARY STATISTICS")
    print(f"{'─'*55}")
    if rows:
        try:
            all_scores = [float(r["score"]) for r in rows]
            sorted_by_rank = sorted(rows, key=lambda r: int(r["rank"]))
            print(f"  Score range  : {min(all_scores):.6f} – {max(all_scores):.6f}")
            print(f"  Score mean   : {sum(all_scores)/len(all_scores):.6f}")
            print(f"\n  TOP 5 CANDIDATES:")
            for r in sorted_by_rank[:5]:
                print(f"    #{r['rank']:>3}  {r['candidate_id']}  score={float(r['score']):.4f}")
                print(f"          {r['reasoning'][:90]}")
        except Exception:
            pass

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print(f"    ⚠  {w}")
    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗  {e}")
        print(f"\n  RESULT: ✗ SUBMISSION IS INVALID — fix errors above before submitting.")
        print(f"{'='*55}\n")
        return False
    else:
        print("  RESULT: ✓ Submission is valid.")
        print(f"{'='*55}\n")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_and_submit.py <submission.csv>")
        sys.exit(1)
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)