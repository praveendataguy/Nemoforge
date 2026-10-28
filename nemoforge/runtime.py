"""Runs a SystemSpec: each agent in the pipeline gets the request plus earlier agents'
outputs, and can call its tools in a JSON action loop before giving its answer.

The same runtime powers evals during a build and the deployed endpoint afterwards.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .config import Settings, settings as load_settings
from . import router
from .llm import LLM, parse_json
from .spec import AgentSpec, SystemSpec
from .tools import BUILTINS, call_tool, describe

_PROTOCOL = """
You can use tools. Each turn, reply with exactly one JSON object and nothing else:
  {"action": "tool", "tool": "<tool name>", "args": {...}}   to call a tool, or
  {"action": "final", "output": "<your answer>"}             when you are done.
Tool results come back in the next message. Never invent tool results.
""".strip()

_NO_TOOLS = 'Reply with exactly one JSON object: {"action": "final", "output": "<your answer>"}.'


def _agent_tools(spec: SystemSpec, agent: AgentSpec):
    out = []
    for name in agent.tools:
        t = spec.tool(name) or (BUILTINS[name][0] if name in BUILTINS else None)
        if t is not None:
            out.append(t)
    return out


def _clip(value: Any, limit: int = 4000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def run_agent(
    llm: LLM, spec: SystemSpec, agent: AgentSpec, request: str, context: list[dict[str, str]], cfg: Settings, tier: str | None = None
) -> dict[str, Any]:
    tier = tier or agent.tier
    tools = _agent_tools(spec, agent)
    system = (
        f"{agent.system_prompt}\n\nYou are '{agent.name}' ({agent.role}) in a team of agents. "
        f"Overall goal of the team: {spec.goal}\n\n"
        + (f"Tools:\n{describe(tools)}\n\n{_PROTOCOL}" if tools else _NO_TOOLS)
    )
    prior = "\n\n".join(f"## Output from {c['agent']}\n{c['output']}" for c in context)
    transcript = f"# Request\n{request}" + (f"\n\n# Work so far\n{prior}" if prior else "")
    steps: list[dict[str, Any]] = []
    for _ in range(cfg.max_agent_steps):
        reply = llm.chat(tier, system, transcript, max_tokens=4096)
        try:
            action = parse_json(reply)
        except ValueError:
            # Treat a plain-text reply as the final answer rather than failing the run.
            return {"output": reply.strip(), "steps": steps}
        if not isinstance(action, dict):
            return {"output": _clip(action), "steps": steps}
        if action.get("action") == "tool" and tools:
            name = str(action.get("tool", ""))
            args = action.get("args") or {}
            started = time.time()
            result = call_tool(spec.tool(name), name, args if isinstance(args, dict) else {}, cfg)
            steps.append({"tool": name, "args": args, "result": result, "ms": int((time.time() - started) * 1000)})
            transcript += f"\n\n# You called {name}({_clip(args, 500)})\n# Result\n{_clip(result)}"
            continue
        output = action.get("output", reply)
        return {"output": output if isinstance(output, str) else json.dumps(output, indent=2), "steps": steps}
    transcript += "\n\nYou have used all your tool calls. Give your final answer now."
    reply = llm.chat(tier, system, transcript, max_tokens=4096)
    try:
        action = parse_json(reply)
        output = action.get("output", reply) if isinstance(action, dict) else reply
    except ValueError:
        output = reply
    return {"output": output if isinstance(output, str) else json.dumps(output), "steps": steps}


def run_system(llm: LLM, spec: SystemSpec, request: str, cfg: Settings | None = None) -> dict[str, Any]:
    cfg = cfg or load_settings()
    context: list[dict[str, str]] = []
    trace: list[dict[str, Any]] = []
    started = time.time()
    for name in spec.pipeline:
        agent = spec.agent(name)
        t0 = time.time()
        # Smart routing: Nano rates this request for this agent and picks the tier.
        decision = router.route(llm, agent, request, context) if cfg.smart_routing else None
        tier = decision["tier"] if decision else agent.tier
        res = run_agent(llm, spec, agent, request, context, cfg, tier)
        context.append({"agent": name, "output": res["output"]})
        trace.append({"agent": name, "tier": tier, "route": decision, "output": res["output"], "steps": res["steps"], "ms": int((time.time() - t0) * 1000)})
    return {"output": context[-1]["output"] if context else "", "trace": trace, "ms": int((time.time() - started) * 1000)}
