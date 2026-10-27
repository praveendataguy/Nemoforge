"""The blueprint Agent Foundry designs, tests, improves and deploys.

A SystemSpec is plain data (JSON on disk), so every build is inspectable and diffable
between optimisation rounds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Tier = Literal["nano", "super", "ultra"]

_SLUG = re.compile(r"[^a-z0-9_]+")


def slug(text: str) -> str:
    s = _SLUG.sub("_", text.lower()).strip("_")
    return s or "item"


class ToolSpec(BaseModel):
    name: str
    description: str
    # JSON-schema-like mapping of argument name -> {"type": ..., "description": ...}
    parameters: dict[str, Any] = Field(default_factory=dict)
    # "builtin" tools ship with the runtime; "python" tools are written by the Toolsmith.
    kind: Literal["builtin", "python"] = "python"
    code: str = ""
    # Small self-checks the Toolsmith writes alongside the code: {"args": {...}, "expect_contains": "..."}
    tests: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return slug(v)


class AgentSpec(BaseModel):
    name: str
    role: str
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    tier: Tier = "nano"

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return slug(v)


class EvalCase(BaseModel):
    id: str
    input: str
    # What a correct answer must do, in plain words, for the judge.
    criteria: str
    # Optional cheap deterministic checks run before the judge.
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)


class SystemSpec(BaseModel):
    name: str
    goal: str
    summary: str
    agents: list[AgentSpec]
    # Agents run in this order; each sees the user input plus earlier agents' outputs.
    pipeline: list[str]
    tools: list[ToolSpec] = Field(default_factory=list)
    input_hint: str = ""
    version: int = 1

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return slug(v)

    def agent(self, name: str) -> AgentSpec:
        for a in self.agents:
            if a.name == name:
                return a
        raise KeyError(name)

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    def check(self, builtin_names: set[str]) -> list[str]:
        """Return a list of structural problems (empty when the spec is runnable)."""
        problems: list[str] = []
        names = {a.name for a in self.agents}
        if not self.pipeline:
            problems.append("pipeline is empty")
        for step in self.pipeline:
            if step not in names:
                problems.append(f"pipeline step '{step}' is not a defined agent")
        tool_names = {t.name for t in self.tools} | builtin_names
        for a in self.agents:
            for t in a.tools:
                if t not in tool_names:
                    problems.append(f"agent '{a.name}' uses unknown tool '{t}'")
        for t in self.tools:
            if t.kind == "python" and "def run(" not in t.code:
                problems.append(f"tool '{t.name}' has no `def run(` function yet")
        return problems

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "SystemSpec":
        return cls.model_validate(json.loads(path.read_text()))
