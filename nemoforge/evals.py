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
correct answer is certain to contain (often leave it empty). Never put the wording of a
refusal or any phrase the system could not know in advance in "must_include"; describe it
in "criteria" instead. Tag each case with "kind": "typical", "edge" or "refuse".
Criteria must follow from the goal and the input alone: never require a business rule, format
detail or policy the goal does not state, and when an input is ambiguous accept any reasonable
reading. Only write a "refuse" case when the goal clearly implies something to refuse.

JSON shape: {{"cases": [{{"id": "c1", "kind": "typical", "input": "...", "criteria": "...", "must_include": [], "must_not_include": []}}]}}"""

JUDGE = """You are a strict grader. Decide whether the AI system's answer meets the criteria.
Judge only against the criteria and the request; ignore style unless the criteria mention it.
Hallucinated facts, ignored instructions or missing required parts are failures.

JSON shape: {"pass": true|false, "score": 0.0-1.0, "reason": "<one or two sentences>"}"""


APPEAL = """

A first grader failed this answer, saying: {reason}
Check that claim against the criteria yourself. Fail it only if the answer really misses a stated
criterion; pass it if the first grader asked for something the criteria do not require."""


def design_cases(llm: LLM, spec: SystemSpec, n: int) -> list[EvalCase]:
    user = f"System goal: {spec.goal}\nSummary: {spec.summary}\nExpected input: {spec.input_hint}"
    data = chat_json(llm, "ultra", DESIGN.format(n=n), user, max_tokens=5000)
    cases = []
    for i, c in enumerate(data.get("cases", [])[:n]):
        c.setdefault("id", f"c{i + 1}")
        cases.append(EvalCase.model_validate(c))
    return cases


def split(cases: list[EvalCase]) -> tuple[list[EvalCase], list[EvalCase]]:
    """Hold out about a third of the cases, spread across kinds, keeping every kind in training."""
    n_hold = len(cases) // 3 if len(cases) >= 3 else 0
    groups: dict[str, list[EvalCase]] = {}
    for c in cases:
        groups.setdefault(c.kind, []).append(c)
    held: list[EvalCase] = []
    # Take the last case of each kind in turn, never a kind's only case.
    while len(held) < n_hold:
        took = False
        for group in groups.values():
            if len(held) < n_hold and len(group) > 1:
                held.append(group.pop())
                took = True
        if not took:
            break
    if len(held) < n_hold:  # every kind has one case left: hold out from the largest kind
        rest = [c for c in cases if c not in held]
        held += rest[len(rest) - (n_hold - len(held)) :]
    return [c for c in cases if c not in held], [c for c in cases if c in held]


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
    by = "judge"
    if not verdict.get("pass"):
        # Second opinion: Ultra re-grades every failure, so one harsh Super verdict can't sink a good answer.
        verdict = chat_json(llm, "ultra", JUDGE + APPEAL.format(reason=verdict.get("reason", "")), user, max_tokens=800)
        by = "judge+ultra"
    return {
        "pass": bool(verdict.get("pass")),
        "score": float(verdict.get("score", 1.0 if verdict.get("pass") else 0.0)),
        "reason": str(verdict.get("reason", "")),
        "by": by,
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
