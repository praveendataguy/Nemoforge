"""In-memory run registry and event stream that feeds the live dashboard."""
from __future__ import annotations

import asyncio
import itertools
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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
    # When set, every event is appended here so finished builds survive a restart.
    log_path: Path | None = None
    # (event loop, queue) per live dashboard stream; builds emit from worker threads.
    _waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = field(default_factory=list, repr=False)

    def emit(self, stage: str, message: str, **data: Any) -> None:
        event = {"seq": len(self.events), "t": time.time(), "stage": stage, "message": message, "data": data}
        self.events.append(event)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        for loop, q in list(self._waiters):
            loop.call_soon_threadsafe(q.put_nowait, event)
        # Also print, so the CLI shows the same story as the dashboard.
        print(f"[{stage:>12}] {message}", flush=True)

    def add_usage(self, tier: str, model: str, prompt: int, completion: int, cost: float = 0.0) -> None:
        u = self.usage.setdefault(tier, {"model": model, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        u["calls"] += 1
        u["prompt_tokens"] += prompt
        u["completion_tokens"] += completion
        u["cost_usd"] = round(u.get("cost_usd", 0.0) + cost, 6)

    @property
    def cost_usd(self) -> float:
        return round(sum(u.get("cost_usd", 0.0) for u in self.usage.values()), 6)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "created": self.created,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
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

    def load(self, builds_dir: Path) -> None:
        """Restore finished builds from their event logs (newest last)."""
        for log in sorted(builds_dir.glob("*/events.jsonl")):
            run_id = log.parent.name
            if run_id in self.runs:
                continue
            events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            if not events:
                continue
            done = next((e for e in reversed(events) if e["stage"] in ("done", "error")), None)
            report = (done or {}).get("data", {}).get("report", {})
            status = "failed" if done is None or done["stage"] == "error" else (
                "passed" if (log.parent / "status").exists() and (log.parent / "status").read_text().strip() == "passed" else "below_target"
            )
            run = Run(id=run_id, kind="build", target=events[0].get("data", {}).get("goal", run_id), status=status,
                      created=events[0]["t"], events=events, usage=report.get("usage", {}), result=report)
            self.runs[run_id] = run
            try:
                n = int(run_id.split("-")[0])
                self._counter = itertools.count(max(n + 1, next(self._counter)))
            except ValueError:
                pass

    def list(self) -> list[dict[str, Any]]:
        return [r.summary() for r in sorted(self.runs.values(), key=lambda r: r.created, reverse=True)]


registry = Registry()
