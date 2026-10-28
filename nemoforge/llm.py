"""Model router for Nemotron on Nebius Token Factory.

Three tiers, picked per task:
  nano  - cheap, fast: log triage, classification, short summaries
  super - mid: investigation, picking files, reviewing diffs
  ultra - heavy reasoning: root cause + writing multi-file patches
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, Protocol

from .config import Settings, settings as load_settings
from .events import Run

Tier = Literal["nano", "super", "ultra"]

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.S)


class BudgetExceeded(RuntimeError):
    pass


class LLM(Protocol):
    def chat(self, tier: Tier, system: str, user: str, *, max_tokens: int = 4096) -> str: ...


class NemotronRouter:
    def __init__(self, run: Run | None = None, cfg: Settings | None = None) -> None:
        from openai import OpenAI

        self.cfg = cfg or load_settings()
        if not self.cfg.nebius_api_key:
            raise RuntimeError("NEBIUS_API_KEY is not set. Get a key from Nebius Token Factory.")
        self.client = OpenAI(base_url=self.cfg.nebius_base_url, api_key=self.cfg.nebius_api_key, timeout=300)
        self.run = run

    def model_for(self, tier: Tier) -> str:
        return getattr(self.cfg.models, tier)

    def chat(self, tier: Tier, system: str, user: str, *, max_tokens: int = 4096) -> str:
        if self.cfg.lite and tier == "ultra":
            tier = "super"
        if self.run is not None and self.cfg.budget_usd and self.run.cost_usd >= self.cfg.budget_usd:
            raise BudgetExceeded(f"budget of ${self.cfg.budget_usd:.2f} used up (spent ${self.run.cost_usd:.4f})")
        model = self.model_for(tier)
        think = self.cfg.thinking.get(tier, True)
        started = time.time()
        resp = self._create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.2,
            think=think,
        )
        msg = resp.choices[0].message
        text = msg.content or ""
        thought = reasoning_of(msg, text)
        if self.run is not None and resp.usage is not None:
            p_in, p_out = self.cfg.prices[tier]
            cost = (resp.usage.prompt_tokens * p_in + resp.usage.completion_tokens * p_out) / 1_000_000
            self.run.add_usage(tier, model, resp.usage.prompt_tokens, resp.usage.completion_tokens, cost)
            self.run.emit(
                "model",
                f"{tier} ({model.split('/')[-1]}) answered in {time.time() - started:.1f}s, ${cost:.4f}",
                tier=tier,
                model=model,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                cost_usd=cost,
                total_cost_usd=self.run.cost_usd,
                thinking=think and self._thinking_param,
                # Rough count: about 4 characters per token.
                thought_tokens=len(thought) // 4,
                thought=_clip(thought, 700),
            )
        return _THINK.sub("", text).strip()

    # Nebius serves Nemotron with its chat template switch for reasoning. If an endpoint rejects it,
    # stop sending it and let the model use its default.
    _thinking_param = True

    def _create(self, *, think: bool, **kwargs: Any):
        if self._thinking_param:
            try:
                return self.client.chat.completions.create(
                    **kwargs, extra_body={"chat_template_kwargs": {"enable_thinking": think}}
                )
            except Exception as err:
                if getattr(err, "status_code", None) != 400:
                    raise
                # Only stop sending the switch when the same request works without it.
                resp = self.client.chat.completions.create(**kwargs)
                NemotronRouter._thinking_param = False
                return resp
        return self.client.chat.completions.create(**kwargs)


def reasoning_of(msg: Any, text: str) -> str:
    """Nemotron's reasoning for one reply: a separate reasoning field, or the <think> block in the text."""
    extra = getattr(msg, "model_extra", None) or {}
    thought = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning") or ""
    if not thought:
        m = re.search(r"<think>(.*?)(?:</think>|$)", text, re.S)
        thought = m.group(1) if m else ""
    return thought.strip()


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rstrip() + "..."


def embed(cfg: Settings, texts: list[str]) -> list[list[float]]:
    """Embed texts with the Nebius embedding model (used to rank agent memory)."""
    from openai import OpenAI

    client = OpenAI(base_url=cfg.nebius_base_url, api_key=cfg.nebius_api_key, timeout=60)
    resp = client.embeddings.create(model=cfg.embed_model, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def parse_json(text: str) -> Any:
    """Pull the first JSON object out of a model reply (fenced or bare)."""
    text = _THINK.sub("", text).strip()
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (candidate.find("{"), candidate.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"No JSON in model reply: {text[:300]}")
    opener = candidate[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start : i + 1])
    raise ValueError(f"Unbalanced JSON in model reply: {text[:300]}")


def chat_json(llm: LLM, tier: Tier, system: str, user: str, *, max_tokens: int = 4096, retries: int = 1) -> Any:
    reply = llm.chat(tier, system + "\n\nReply with a single JSON object only.", user, max_tokens=max_tokens)
    for attempt in range(retries + 1):
        try:
            return parse_json(reply)
        except ValueError as err:
            if attempt == retries:
                raise
            reply = llm.chat(
                tier,
                system + "\n\nReply with a single valid JSON object only. No prose.",
                user + f"\n\nYour previous reply was not valid JSON ({err}). Try again.",
                max_tokens=max_tokens,
            )
    raise AssertionError("unreachable")
