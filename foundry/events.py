"""In-memory run registry and event stream that feeds the live dashboard."""
from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Run:
    id: str
    kind: str  # "build" or "invoke"
    target: str
    status: str = "running"  # running | passed | below_target | failed
    created: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    # (event loop, queue) per live dashboard stream; builds emit from worker threads.
    _waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = field(default_factory=list, repr=False)

    def emit(self, stage: str, message: str, **data: Any) -> None:
        event = {"seq": len(self.events), "t": time.time(), "stage": stage, "message": message, "data": data}
        self.events.append(event)
        for loop, q in list(self._waiters):
            loop.call_soon_threadsafe(q.put_nowait, event)
        # Also print, so the CLI shows the same story as the dashboard.
        print(f"[{stage:>12}] {message}", flush=True)

    def add_usage(self, tier: str, model: str, prompt: int, completion: int) -> None:
        u = self.usage.setdefault(tier, {"model": model, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        u["calls"] += 1
        u["prompt_tokens"] += prompt
        u["completion_tokens"] += completion

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "created": self.created,
            "usage": self.usage,
            "result": self.result,
            "events": len(self.events),
        }

    async def stream(self):
        q: asyncio.Queue = asyncio.Queue()
        waiter = (asyncio.get_running_loop(), q)
        self._waiters.append(waiter)
        try:
            for e in list(self.events):
                yield e
            seen = len(self.events)
            while self.status == "running" or not q.empty():
                try:
                    e = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"stage": "ping"}
                    continue
                if e["seq"] >= seen:
                    yield e
        finally:
            self._waiters.remove(waiter)


class Registry:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self._counter = itertools.count(1)

    def new(self, kind: str, target: str) -> Run:
        run = Run(id=f"{next(self._counter):03d}-{uuid.uuid4().hex[:6]}", kind=kind, target=target)
        self.runs[run.id] = run
        return run

    def list(self) -> list[dict[str, Any]]:
        return [r.summary() for r in sorted(self.runs.values(), key=lambda r: r.created, reverse=True)]


registry = Registry()
