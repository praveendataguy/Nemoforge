"""Build 10 agent teams (easy, medium, complex) inside one total budget and print a scorecard.

    python demo/batch_test.py              # $1.00 total
    python demo/batch_test.py --total 0.5  # smaller budget
    python demo/batch_test.py --lite       # Super does Ultra's jobs (about 3x cheaper)

Builds land in the normal builds folder, so they also show up in the web dashboard.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemoforge.config import settings  # noqa: E402
from nemoforge.events import registry  # noqa: E402
from nemoforge.forge import build  # noqa: E402
from nemoforge.llm import NemotronRouter  # noqa: E402

GOALS = [
    ("easy", "Convert a temperature given in Celsius to Fahrenheit and Kelvin"),
    ("easy", "Extract every email address from a block of text and return them as a list"),
    ("easy", "Count the words and sentences in a paragraph and estimate the reading time in minutes"),
    ("medium", "Classify a customer support ticket as billing, technical or account, and rate its urgency as low, medium or high"),
    ("medium", "Read an invoice and check the total, flag any line that does not add up"),
    ("medium", "Turn a messy postal address into structured JSON with street, city, postcode and country"),
    ("medium", "Summarise a product review into pros, cons and a star rating from 1 to 5"),
    ("complex", "Check a job description for biased or exclusionary language, explain each issue and rewrite the text"),
    ("complex", "Read a customer support email, work out what they need, check our refund policy rules and draft a reply"),
    ("complex", "Research a company by name on the web and write a one paragraph brief with its sources"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--total", type=float, default=1.0, help="total USD for all builds")
    p.add_argument("--lite", action="store_true", help="send Ultra's jobs to Super (cheaper)")
    p.add_argument("--only", choices=["easy", "medium", "complex"], help="run one difficulty only")
    args = p.parse_args()

    base = settings()
    goals = [g for g in GOALS if not args.only or g[0] == args.only]
    spent, rows = 0.0, []
    for i, (level, goal) in enumerate(goals):
        left = args.total - spent
        if left < 0.02:
            print(f"\nBudget used up, skipping the last {len(goals) - i} builds.")
            break
        # Share what's left evenly across the builds still to run, so one hungry build can't starve the rest.
        cap = round(left / (len(goals) - i), 4)
        cfg = dataclasses.replace(base, budget_usd=cap, lite=args.lite or base.lite)
        print(f"\n[{i + 1}/{len(goals)}] {level.upper()}  budget ${cap:.3f}  {goal}", flush=True)
        run = registry.new("build", goal)
        t = time.time()
        try:
            report = build(NemotronRouter(run, cfg), goal, run, cfg)
        except Exception as e:  # keep going, one broken build shouldn't stop the batch
            report = {"error": f"{type(e).__name__}: {e}"}
        cost = float(report.get("cost_usd") or 0)
        spent += cost
        rows.append({"level": level, "goal": goal, "status": run.status, "name": report.get("name", ""),
                     "train": report.get("train_score"), "holdout": report.get("holdout_score"),
                     "cost": cost, "seconds": round(time.time() - t, 1), "error": report.get("error", "")})
        r = rows[-1]
        print(f"    -> {r['status']}  train {r['train']}  holdout {r['holdout']}  ${cost:.4f}  {r['seconds']}s {r['error']}", flush=True)

    passed = sum(r["status"] == "passed" for r in rows)
    print(f"\n{'level':8} {'status':8} {'holdout':>7} {'cost':>8}  goal")
    for r in rows:
        h = "-" if r["holdout"] is None else f"{r['holdout']:.2f}"
        print(f"{r['level']:8} {r['status']:8} {h:>7} ${r['cost']:7.4f}  {r['goal'][:70]}")
    print(f"\n{passed}/{len(rows)} passed, total spent ${spent:.4f} of ${args.total:.2f}")
    out = base.builds_dir / "batch_report.json"
    out.write_text(json.dumps({"spent": spent, "passed": passed, "builds": rows}, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
