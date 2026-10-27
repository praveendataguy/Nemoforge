"""Nemotron Ultra reads failing eval traces and rewrites the parts of the spec that caused them."""
from __future__ import annotations

import json
from typing import Any

from .llm import LLM, chat_json
from .spec import SystemSpec
from .tools import BUILTINS, describe

SYSTEM = """You are the Optimizer in NemoForge. A team of agents failed some of its tests.
Find the ROOT CAUSE of each failure in the traces and fix the design, not the symptoms.

You may: rewrite system prompts (be specific: add output formats, rules, examples),
change which tools an agent has, change an agent's tier, add/remove/reorder agents,
or ask for a custom tool to be rewritten by listing it in "rewrite_tools" with a note.
Do not special-case individual test inputs; changes must generalise to new requests.
Do not touch what already works. Prefer the cheapest tier that will pass.

Built-in tools available: {builtins}

JSON shape:
{{
  "diagnosis": "<2-4 sentences: what went wrong and why>",
  "changes": ["<short description of each change>"],
  "agents": [ ...full updated agent list, same shape as input... ],
  "pipeline": ["..."],
  "new_tools": [{{"name": "...", "description": "...", "parameters": {{...}}}}],
  "rewrite_tools": [{{"name": "...", "note": "<what is wrong with it>"}}]
}}"""


def _failure_report(results: list[dict[str, Any]], limit: int = 5) -> str:
    out = []
    for r in [r for r in results if not r["pass"]][:limit]:
        steps = []
        for t in r["trace"]:
            calls = "; ".join(f"{s['tool']}({json.dumps(s['args'])[:150]}) -> {json.dumps(s['result'])[:300]}" for s in t["steps"])
            steps.append(f"  [{t['agent']}] tools: {calls or 'none'}\n  [{t['agent']}] output: {t['output'][:900]}")
        out.append(
            f"### Case {r['case']['id']}\nRequest: {r['case']['input']}\nCriteria: {r['case']['criteria']}\n"
            f"Grader: {r['reason']}\nTrace:\n" + "\n".join(steps)
        )
    return "\n\n".join(out)


def improve(llm: LLM, spec: SystemSpec, results: list[dict[str, Any]]) -> tuple[SystemSpec, dict[str, Any]]:
    current = spec.model_dump(include={"agents", "pipeline"})
    current["custom_tools"] = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in spec.tools]
    passed = sum(r["pass"] for r in results)
    user = (
        f"Goal: {spec.goal}\n\nCurrent design:\n{json.dumps(current, indent=2)}\n\n"
        f"Score: {passed}/{len(results)} tests passed.\n\nFailures:\n{_failure_report(results)}"
    )
    data = chat_json(llm, "ultra", SYSTEM.format(builtins=describe([b[0] for b in BUILTINS.values()])), user, max_tokens=8000)
    update: dict[str, Any] = {"version": spec.version + 1}
    if data.get("agents"):
        update["agents"] = data["agents"]
    if data.get("pipeline"):
        update["pipeline"] = data["pipeline"]
    tools = [t.model_dump() for t in spec.tools]
    known = {t["name"] for t in tools}
    for t in data.get("new_tools", []) or []:
        if t.get("name") and t["name"] not in known:
            tools.append({**t, "kind": "python", "code": ""})
    rewrite = {str(t.get("name")): str(t.get("note", "")) for t in data.get("rewrite_tools", []) or []}
    for t in tools:
        if t["name"] in rewrite:
            t["code"] = ""  # blank code tells NemoForge to re-forge it
            t["description"] += f" (Fix: {rewrite[t['name']]})" if rewrite[t["name"]] else ""
    update["tools"] = tools
    new_spec = SystemSpec.model_validate({**spec.model_dump(), **update})
    return new_spec, {"diagnosis": data.get("diagnosis", ""), "changes": data.get("changes", [])}
