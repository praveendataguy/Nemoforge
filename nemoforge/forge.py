"""The build loop: design -> forge tools -> write tests -> evaluate -> improve -> repeat.

Test cases are split into a training set (the Optimizer sees these failures) and a
held-out set (never shown to the Optimizer). The held-out score is the honest number:
it shows whether improvements generalise or just overfit the visible tests.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import architect, evals, optimizer, toolsmith
from .config import Settings, settings as load_settings
from .events import Run
from .llm import LLM, BudgetExceeded
from .spec import EvalCase, SystemSpec
from .tools import builtin_names


def _score(results: list[dict[str, Any]]) -> float:
    return sum(r["pass"] for r in results) / len(results) if results else 0.0


def _forge_missing(llm: LLM, spec: SystemSpec, run: Run, cfg: Settings) -> SystemSpec:
    tools = []
    for t in spec.tools:
        if t.kind == "python" and not t.code:
            run.emit("toolsmith", f"Writing tool '{t.name}' with Nemotron Super", tool=t.name)
            t = toolsmith.forge(llm, t, spec.goal, run, timeout=cfg.tool_timeout)
        tools.append(t)
    return spec.model_copy(update={"tools": tools})


def _repair(spec: SystemSpec) -> SystemSpec:
    """Drop references the model got wrong (unknown tools, missing agents) instead of crashing."""
    names = builtin_names() | {t.name for t in spec.tools}
    agents = [a.model_copy(update={"tools": [t for t in a.tools if t in names]}) for a in spec.agents]
    agent_names = {a.name for a in agents}
    pipeline = [p for p in spec.pipeline if p in agent_names] or [a.name for a in agents]
    return spec.model_copy(update={"agents": agents, "pipeline": pipeline})


def build(llm: LLM, goal: str, run: Run, cfg: Settings | None = None, notes: str = "") -> dict[str, Any]:
    cfg = cfg or load_settings()
    big = "Nemotron Super (lite mode)" if cfg.lite else "Nemotron Ultra"
    started = time.time()
    out_dir: Path = cfg.builds_dir / run.id
    out_dir.mkdir(parents=True, exist_ok=True)
    if run.log_path is None:
        run.log_path = out_dir / "events.jsonl"

    run.emit("architect", f"{big} is designing the agent team", goal=goal)
    spec = _repair(architect.design(llm, goal, notes))
    run.emit("architect", f"Designed '{spec.name}': {len(spec.agents)} agents, {len(spec.tools)} custom tools", spec=spec.model_dump())

    spec = _repair(_forge_missing(llm, spec, run, cfg))
    spec.save(out_dir / f"spec_v{spec.version}.json")

    run.emit("tests", f"{big} is writing {cfg.eval_cases} test cases")
    cases = evals.design_cases(llm, spec, cfg.eval_cases)
    train, holdout = evals.split(cases)
    (out_dir / "evals.json").write_text(json.dumps({"train": [c.model_dump() for c in train], "holdout": [c.model_dump() for c in holdout]}, indent=2))
    run.emit("tests", f"{len(train)} training cases, {len(holdout)} held out", train=[c.model_dump() for c in train], holdout=[c.model_dump() for c in holdout])

    history: list[dict[str, Any]] = []
    best: tuple[float, float, SystemSpec] | None = None
    for round_no in range(1, cfg.max_iterations + 1):
        run.emit("eval", f"Round {round_no}: running v{spec.version} on all tests (agents on Nemotron)", round=round_no)
        try:
            res_train = evals.evaluate(llm, spec, train, cfg)
            res_hold = evals.evaluate(llm, spec, holdout, cfg) if holdout else []
        except BudgetExceeded as err:
            if best is None:
                raise
            run.emit("budget", f"Stopping early: {err}. Keeping the best version so far.")
            break
        s_train, s_hold = _score(res_train), _score(res_hold)
        entry = {"round": round_no, "version": spec.version, "train": s_train, "holdout": s_hold, "results": res_train + res_hold}
        history.append(entry)
        run.emit(
            "eval",
            f"Round {round_no}: training {s_train:.0%}, held-out {s_hold:.0%}",
            round=round_no,
            version=spec.version,
            train_score=s_train,
            holdout_score=s_hold,
            results=[
                {k: r[k] for k in ("pass", "score", "reason", "ms")}
                | {"id": r["case"]["id"], "input": r["case"]["input"], "output": r["output"][:1500], "holdout": held}
                for held, group in ((False, res_train), (True, res_hold))
                for r in group
            ],
        )
        rank = (s_train + s_hold, s_hold)
        if best is None or rank > (best[0], best[1]):
            best = (rank[0], rank[1], spec)
        if s_train >= cfg.pass_threshold and (not holdout or s_hold >= cfg.pass_threshold):
            run.emit("eval", "Target reached, stopping")
            break
        if round_no == cfg.max_iterations:
            break
        run.emit("optimizer", f"{big} is diagnosing the failures")
        try:
            spec, info = optimizer.improve(llm, spec, res_train)
            spec = _repair(_forge_missing(llm, spec, run, cfg))
        except BudgetExceeded as err:
            run.emit("budget", f"Stopping early: {err}. Keeping the best version so far.")
            break
        spec.save(out_dir / f"spec_v{spec.version}.json")
        entry["optimizer"] = info
        run.emit("optimizer", info["diagnosis"] or "Updated the design", changes=info["changes"], spec=spec.model_dump())

    assert best is not None
    final = best[2]
    final.save(out_dir / "spec.json")
    last = next(h for h in reversed(history) if h["version"] == final.version)
    report = {
        "goal": goal,
        "name": final.name,
        "version": final.version,
        "train_score": last["train"],
        "holdout_score": last["holdout"],
        "rounds": [{k: h[k] for k in ("round", "version", "train", "holdout")} | ({"changes": h["optimizer"]["changes"]} if "optimizer" in h else {}) for h in history],
        "usage": run.usage,
        "cost_usd": run.cost_usd,
        "seconds": round(time.time() - started, 1),
        "build_dir": str(out_dir),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    run.result = report
    passed = last["train"] >= cfg.pass_threshold and (not holdout or last["holdout"] >= cfg.pass_threshold)
    run.emit("done", f"Built '{final.name}' v{final.version}: training {last['train']:.0%}, held-out {last['holdout']:.0%}", report=report, spec=final.model_dump())
    run.status = "passed" if passed else "below_target"
    (out_dir / "status").write_text(run.status)
    return report


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text())
    return [EvalCase.model_validate(c) for c in data["train"] + data["holdout"]]
