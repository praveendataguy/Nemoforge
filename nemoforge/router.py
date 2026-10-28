"""Smart router: before each agent runs, Nemotron Nano rates how hard THIS request is for
THIS agent's job, and the agent runs on the cheapest tier that can handle it.

Easy requests go to Nano, harder ones to Super, and only the hardest reach Ultra. The
Architect's tier is treated as the plan: the router may step down at most one tier from
it (so a step designed for deep reasoning is never sent to Nano), and may step up as far
as the request needs.
"""
from __future__ import annotations

from typing import Any

from .llm import LLM, chat_json
from .spec import AgentSpec

TIERS = ("nano", "super", "ultra")

SYSTEM = """You are the Router in NemoForge. Rate how hard one request is for one agent's job,
so it runs on the cheapest NVIDIA Nemotron model that will do it well.

1 = trivial (copy, extract an obvious field, short format change)
2 = easy (short classification, simple summary of a short clear text)
3 = moderate (several steps, some judgement, medium-length or messy input)
4 = hard (long or ambiguous input, multi-step reasoning, careful synthesis)
5 = very hard (deep reasoning, conflicting information, high stakes if wrong)

Rate the request, not the agent: the same agent can get easy and hard requests.

JSON shape: {"difficulty": 1-5, "why": "<one short sentence>"}"""


def tier_for(difficulty: int) -> str:
    return "nano" if difficulty <= 2 else "super" if difficulty <= 4 else "ultra"


def choose(planned: str, difficulty: int) -> str:
    """The tier the difficulty asks for, but never more than one step below the plan."""
    want = TIERS.index(tier_for(difficulty))
    floor = max(TIERS.index(planned) - 1, 0)
    return TIERS[max(want, floor)]


def route(llm: LLM, agent: AgentSpec, request: str, context: list[dict[str, str]]) -> dict[str, Any]:
    prior = "\n".join(f"- {c['agent']}: {c['output'][:300]}" for c in context)
    clip = request if len(request) <= 2500 else request[:2000] + f"\n... [{len(request) - 2500} chars omitted] ...\n" + request[-500:]
    user = (
        f"Agent: {agent.name}\nAgent's job: {agent.role}\n"
        f"Request length: {len(request):,} characters\n\n# Request\n{clip}"
        + (f"\n\n# Earlier agents' output (short)\n{prior}" if prior else "")
    )
    try:
        data = chat_json(llm, "nano", SYSTEM, user, max_tokens=300)
        if not isinstance(data, dict) or "difficulty" not in data:
            raise ValueError("no difficulty in reply")
        difficulty = min(max(int(data.get("difficulty", 0)), 1), 5)
        why = str(data.get("why", ""))[:200]
    except Exception as err:  # a broken rating must never break the agent: fall back to the plan
        return {"agent": agent.name, "planned": agent.tier, "tier": agent.tier, "difficulty": None, "why": f"router fallback ({type(err).__name__})"}
    return {"agent": agent.name, "planned": agent.tier, "tier": choose(agent.tier, difficulty), "difficulty": difficulty, "why": why}
