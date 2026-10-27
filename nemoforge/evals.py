"""Test cases for a built system, and a judge that grades each answer."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import Settings
from .llm import LLM, BudgetExceeded, chat_json
from .runtime import run_system
from .spec import EvalCase, SystemSpec

DESIGN = """You are the Test Designer in NemoForge. Write an evaluation set for an AI system.

Write {n} test cases a demanding user would actually send. Mix:
- typical requests (about half),
- edge cases: missing details, ambiguous or messy input, unusual formats,
- at least one request the system should refuse or push back on, if any exist for this goal.
Each case states, in "criteria", what a correct answer MUST do, concretely enough that a
strict grader can decide pass/fail. Use "must_include" only for exact strings that any
correct answer is certain to contain (often leave it empty).

JSON shape: {{"cases": [{{"id": "c1", "input": "...", "criteria": "...", "must_include": [], "must_not_include": []}}]}}"""

JUDGE = """You are a strict grader. Decide whether the AI system's answer meets the criteria.
Judge only against the criteria and the request; ignore style unless the criteria mention it.
Hallucinated facts, ignored instructions or missing required parts are failures.

JSON shape: {"pass": true|false, "score": 0.0-1.0, "reason": "<one or two sentences>"}"""


def design_cases(llm: LLM, spec: SystemSpec, n: int) -> list[EvalCase]:
    user = f"System goal: {spec.goal}\nSummary: {spec.summary}\nExpected input: {spec.input_hint}"
    data = chat_json(llm, "ultra", DESIGN.format(n=n), user, max_tokens=5000)
    cases = []
    for i, c in enumerate(data.get("cases", [])[:n]):
        c.setdefault("id", f"c{i + 1}")
        cases.append(EvalCase.model_validate(c))
    return cases


def grade(llm: LLM, case: EvalCase, answer: str) -> dict[str, Any]:
    low = answer.lower()
    missing = [s for s in case.must_include if s.lower() not in low]
    forbidden = [s for s in case.must_not_include if s.lower() in low]
    if missing or forbidden:
        why = []
        if missing:
            why.append(f"missing required text: {missing}")
        if forbidden:
            why.append(f"contains forbidden text: {forbidden}")
        return {"pass": False, "score": 0.0, "reason": "; ".join(why), "by": "rule"}
    user = f"# Request\n{case.input}\n\n# Criteria\n{case.criteria}\n\n# Answer\n{answer[:8000]}"
    verdict = chat_json(llm, "super", JUDGE, user, max_tokens=800)
    return {
        "pass": bool(verdict.get("pass")),
        "score": float(verdict.get("score", 1.0 if verdict.get("pass") else 0.0)),
        "reason": str(verdict.get("reason", "")),
        "by": "judge",
    }


def evaluate(llm: LLM, spec: SystemSpec, cases: list[EvalCase], cfg: Settings, workers: int = 4) -> list[dict[str, Any]]:
    def one(case: EvalCase) -> dict[str, Any]:
        try:
            res = run_system(llm, spec, case.input, cfg)
            verdict = grade(llm, case, res["output"])
        except BudgetExceeded:
            raise
        except Exception as err:  # one broken case should not sink the whole round
            res = {"output": "", "trace": [], "ms": 0}
            verdict = {"pass": False, "score": 0.0, "reason": f"run crashed: {type(err).__name__}: {err}", "by": "error"}
        return {"case": case.model_dump(), "output": res["output"], "trace": res["trace"], "ms": res["ms"], **verdict}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, cases))
