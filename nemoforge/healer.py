"""The Healer: when a build gets stuck or crashes, Nemotron Ultra works out why and repairs it.

Two cases:
  stuck - every improvement round is used up and the team is still below target. The Healer
          reviews ALL rounds at once (scores, every change the Optimizer made, tests that never
          passed), finds the root cause the round-by-round Optimizer missed, and redesigns
          from the best version so far. NemoForge then runs a bonus round to prove the fix.
  crash - a stage raised an error. The Healer reads the error and the last steps, explains it
          in plain words, and writes notes so the next attempt avoids it.

Held-out tests stay hidden: the Healer only sees their scores, never their inputs.
"""
from __future__ import annotations

import json
from typing import Any

from . import optimizer
from .llm import LLM, chat_json
from .spec import SystemSpec
from .tools import BUILTINS, describe

STUCK = """You are the Healer in NemoForge. An agent team has used all its improvement rounds
and is still below target. A round-by-round Optimizer already tried; look at the WHOLE history
to find what it kept missing.

Check for these patterns:
- the same tests fail in every round (a structural problem: wrong agent split, missing tool,
  tier too weak for the reasoning needed, an output format the grader rejects),
- held-out score falls while training rises (overfitting: rules that only fit the visible tests;
  replace them with general rules),
- a change made things worse (undo it),
- a custom tool that keeps crashing (rewrite it, or replace it with a built-in).
Start from the BEST version shown. You may raise an agent's tier when the evidence says it lacks
reasoning, but keep the cheapest tier that will pass. Changes must generalise to new requests.

Built-in tools available: {builtins}

JSON shape:
{{
  "diagnosis": "<2-4 sentences, plain words: why the build is stuck>",
  "root_causes": ["<each root cause, one line>"],
  "overfitting": true|false,
  "changes": ["<short description of each change>"],
  "agents": [ ...full updated agent list, same shape as input... ],
  "pipeline": ["..."],
  "new_tools": [{{"name": "...", "description": "...", "parameters": {{...}}}}],
  "rewrite_tools": [{{"name": "...", "note": "<what is wrong with it>"}}]
}}"""

CRASH = """You are the Healer in NemoForge. A build of an AI agent team crashed.
Read the error and the last steps. Explain the cause in plain words, then write short notes for
the Architect that will avoid the crash on the next attempt (for example: fewer agents, no custom
tool for X, simpler output format, smaller test inputs). If no design change can fix it (a missing
or invalid API key, a network outage, an exhausted budget), set "retry" to false.

JSON shape: {"diagnosis": "<1-3 sentences>", "fix_notes": "<notes for the Architect>", "retry": true|false}"""

# Errors that a new design cannot fix: retrying would only spend more credits.
NOT_HEALABLE = ("BudgetExceeded", "AuthenticationError", "PermissionDeniedError", "APIConnectionError", "APITimeoutError", "NotFoundError")


def _rounds_table(history: list[dict[str, Any]]) -> str:
    lines = []
    for h in history:
        changes = "; ".join(h.get("optimizer", {}).get("changes", [])) or "-"
        lines.append(f"round {h['round']} v{h['version']}: training {h['train']:.0%}, held-out {h['holdout']:.0%}; changes after it: {changes}")
    return "\n".join(lines)


def always_failing(train_by_round: list[list[dict[str, Any]]]) -> list[str]:
    """Training cases that failed in every round so far."""
    failing: set[str] | None = None
    for results in train_by_round:
        ids = {r["case"]["id"] for r in results if not r["pass"]}
        failing = ids if failing is None else failing & ids
    return sorted(failing or [])


def heal_stuck(
    llm: LLM, best: SystemSpec, history: list[dict[str, Any]], train_by_round: list[list[dict[str, Any]]], best_train: list[dict[str, Any]], version: int
) -> tuple[SystemSpec, dict[str, Any]]:
    stuck = always_failing(train_by_round)
    passed = sum(r["pass"] for r in best_train)
    user = (
        f"Goal: {best.goal}\n\nEvery round so far:\n{_rounds_table(history)}\n\n"
        f"Training tests that failed in EVERY round: {', '.join(stuck) or 'none'}\n\n"
        f"Best version so far is v{best.version} ({passed}/{len(best_train)} training tests passed). Its design:\n"
        f"{json.dumps(optimizer.current_design(best), indent=2)}\n\n"
        f"Failures of the best version:\n{optimizer._failure_report(best_train, limit=6)}"
    )
    data = chat_json(llm, "ultra", STUCK.format(builtins=describe([b[0] for b in BUILTINS.values()])), user, max_tokens=8000)
    info = {
        "diagnosis": str(data.get("diagnosis", "")),
        "root_causes": [str(c) for c in data.get("root_causes", []) or []],
        "overfitting": bool(data.get("overfitting")),
        "changes": [str(c) for c in data.get("changes", []) or []],
        "always_failing": stuck,
        "from_version": best.version,
    }
    return optimizer.apply(best, data, version=version), info


def heal_crash(llm: LLM, err: BaseException, trace: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if type(err).__name__ in NOT_HEALABLE or "API_KEY" in str(err):
        return {"diagnosis": f"{type(err).__name__}: {err}", "fix_notes": "", "retry": False}
    steps = "\n".join(f"[{e['stage']}] {e['message']}" for e in events if e.get("stage") not in ("model", "ping"))[-3000:]
    user = f"Error: {type(err).__name__}: {err}\n\nTraceback (end):\n{trace[-2000:]}\n\nLast steps:\n{steps}"
    data = chat_json(llm, "ultra", CRASH, user, max_tokens=1500)
    return {"diagnosis": str(data.get("diagnosis", "")), "fix_notes": str(data.get("fix_notes", "")), "retry": bool(data.get("retry", True))}
