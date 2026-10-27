"""A scripted stand-in for Nemotron, used only to test Foundry plumbing offline.

It answers each Foundry role with a fixed, valid reply. It does not judge quality;
real builds always run on Nemotron via Nebius Token Factory.
"""
from __future__ import annotations

import json
import threading

SPEC = {
    "name": "invoice_checker",
    "goal": "Check an invoice total",
    "summary": "Adds up invoice lines and flags mistakes.",
    "input_hint": "Paste invoice lines like 'Widget 2 x 5.00'",
    "agents": [
        {"name": "parser", "role": "extract lines", "tier": "nano", "tools": ["sum_lines"], "system_prompt": "Parse lines."},
        {"name": "writer", "role": "write verdict", "tier": "nano", "tools": [], "system_prompt": "Write the verdict."},
    ],
    "pipeline": ["parser", "writer"],
    "tools": [{"name": "sum_lines", "description": "Sum qty*price", "parameters": {"lines": {"type": "array"}}}],
}

TOOL = {
    "code": "def run(lines):\n    return {'total': round(sum(q*p for q, p in lines), 2)}\n",
    "tests": [{"args": {"lines": [[2, 5.0]]}, "expect_contains": "10.0"}],
}


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.rounds = 0
        self._lock = threading.Lock()

    def chat(self, tier, system, user, *, max_tokens=4096):
        with self._lock:
            self.calls.append((tier, system[:40]))
        if system.startswith("You are the Architect"):
            return "```json\n" + json.dumps(SPEC) + "\n```"
        if system.startswith("You are the Toolsmith"):
            return json.dumps(TOOL)
        if system.startswith("You are the Test Designer"):
            return json.dumps({"cases": [{"id": f"c{i}", "input": "Widget 2 x 5.00", "criteria": "total is 10.00"} for i in range(1, 7)]})
        if system.startswith("You are a strict grader"):
            ok = "10.0" in user and self.rounds >= 1
            return json.dumps({"pass": ok, "score": 1.0 if ok else 0.2, "reason": "ok" if ok else "total missing"})
        if system.startswith("You are the Optimizer"):
            with self._lock:
                self.rounds += 1
            return json.dumps({"diagnosis": "Writer ignored the tool output.", "changes": ["Writer must quote the total"], "agents": SPEC["agents"], "pipeline": SPEC["pipeline"]})
        if "'parser'" in system:
            if "# Result" not in user:
                return '{"action": "tool", "tool": "sum_lines", "args": {"lines": [[2, 5.0]]}}'
            return '{"action": "final", "output": "lines parsed, total 10.0"}'
        return '<think>hmm</think>{"action": "final", "output": "Invoice total is 10.0"}'
