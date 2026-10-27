"""Nemotron Super writes each custom Python tool, plus self-checks, and keeps fixing the
code until the self-checks pass in the sandbox."""
from __future__ import annotations

import json
from typing import Any

from .events import Run
from .llm import LLM, chat_json
from .spec import ToolSpec
from .tools import run_python_tool

SYSTEM = """You are the Toolsmith in Agent Foundry. You write one small, correct Python tool.

Rules:
- Define `def run(<parameters>) -> dict | list | str | int | float` as the entry point.
  Parameter names must match the spec exactly; give optional ones defaults.
- Python standard library only. No network, no files outside the working dir, no input().
- Validate inputs and return {"error": "..."} for bad input instead of raising.
- Keep it short and readable.

Also write 3 self-checks that prove it works:
  {"args": {...}, "expect_contains": "<substring that must appear in json.dumps(result)>"}

JSON shape: {"code": "<full python source>", "tests": [ ... ]}"""


def _check(code: str, tests: list[dict[str, Any]], timeout: int) -> list[str]:
    failures = []
    for i, t in enumerate(tests):
        out = run_python_tool(code, t.get("args") or {}, timeout=timeout)
        if "error" in out and "result" not in out:
            failures.append(f"check {i + 1} args={t.get('args')} crashed: {out['error'][-600:]}")
            continue
        blob = json.dumps(out.get("result"), default=str)
        want = str(t.get("expect_contains", ""))
        if want and want not in blob:
            failures.append(f"check {i + 1} args={t.get('args')}: expected '{want}' in {blob[:400]}")
    return failures


def forge(llm: LLM, tool: ToolSpec, goal: str, run: Run | None = None, attempts: int = 3, timeout: int = 20) -> ToolSpec:
    brief = (
        f"Team goal: {goal}\n\nTool to write:\nname: {tool.name}\ndescription: {tool.description}\n"
        f"parameters: {tool.parameters}"
    )
    feedback = ""
    for attempt in range(1, attempts + 1):
        data = chat_json(llm, "super", SYSTEM, brief + feedback, max_tokens=4000)
        code, tests = str(data.get("code", "")), list(data.get("tests", []))
        failures = ["no `def run(` function"] if "def run(" not in code else _check(code, tests, timeout)
        if run:
            run.emit(
                "toolsmith",
                f"{tool.name}: attempt {attempt}, {len(tests) - len(failures)}/{len(tests)} self-checks passed",
                tool=tool.name,
                attempt=attempt,
                passed=not failures,
                code=code,
                failures=failures,
            )
        if not failures:
            return tool.model_copy(update={"code": code, "tests": tests})
        feedback = "\n\nYour previous code failed:\n" + "\n".join(failures) + f"\n\nPrevious code:\n```python\n{code}\n```\nFix it."
    # Keep the last attempt so the optimiser can still see and repair it.
    return tool.model_copy(update={"code": code, "tests": tests})
