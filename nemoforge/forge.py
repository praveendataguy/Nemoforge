"""The build loop: design -> forge tools -> write tests -> evaluate -> improve -> repeat -> heal.

Test cases are split into a training set (the Optimizer sees these failures) and a
held-out set (never shown to the Optimizer). The held-out score is the honest number:
it shows whether improvements generalise or just overfit the visible tests.

When the rounds run out below target, or a stage crashes, the Healer (healer.py) works out
why and repairs the build, then a bonus round checks that the repair really helped.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from . import architect, evals, healer, optimizer, toolsmith
from .config import Settings, settings as load_settings
from .events import Run
from .llm import LLM, BudgetExceeded
from .spec import EvalCase, SystemSpec
from .tools import builtin_names


def _score(results: list[dict[str, Any]]) -> float:
    return sum(r["pass"] for r in results) / len(results) if results else 0.0


def _routes(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The smart router's decision for each agent in one run (empty when routing is off)."""
    return [t["route"] for t in trace if t.get("route")]


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
    """Run a build; if it crashes, the Healer diagnoses the crash and retries with a fix."""
    cfg = cfg or load_settings()
    big = "Nemotron Super (lite mode)" if cfg.lite else "Nemotron Ultra"
    heals: list[dict[str, Any]] = []
    for attempt in range(cfg.heal_retries + 1):
        try:
            return _build(llm, goal, run, cfg, notes, heals)
        except BudgetExceeded:
            raise
        except Exception as err:
            if attempt == cfg.heal_retries:
                raise
            trace = traceback.format_exc()
            run.emit("healer", f"Build crashed ({type(err).__name__}). {big} is diagnosing the crash", kind="crash", error=f"{type(err).__name__}: {err}")
            try:
                fix = healer.heal_crash(llm, err, trace, run.events[-60:])
            except Exception:
                raise err
            if not fix["retry"]:
                run.emit("healer", f"Redesigning cannot fix this: {fix['diagnosis']}", kind="crash")
                raise
            heals.append({"kind": "crash", "attempt": attempt + 1, "error": f"{type(err).__name__}: {err}", **fix})
            run.emit("healer", f"{fix['diagnosis']} Rebuilding with a fix.", kind="crash", diagnosis=fix["diagnosis"], fix_notes=fix["fix_notes"])
            notes = (notes + "\n\n" if notes else "") + "Avoid what crashed the last attempt: " + fix["fix_notes"]
    raise AssertionError("unreachable")


def _build(llm: LLM, goal: str, run: Run, cfg: Settings, notes: str, heals: list[dict[str, Any]]) -> dict[str, Any]:
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
    (out_dir / "evals.json").write_text(json.dumps({"train": [c.model_dump() for c in train], "holdout": [c.model_dump() for c in holdout]}, indent=2), encoding="utf-8")
    run.emit("tests", f"{len(train)} training cases, {len(holdout)} held out", train=[c.model_dump() for c in train], holdout=[c.model_dump() for c in holdout])

    history: list[dict[str, Any]] = []
    train_by_round: list[list[dict[str, Any]]] = []
    # (train + held-out, held-out, spec, its training results)
    best: tuple[float, float, SystemSpec, list[dict[str, Any]]] | None = None
    top_version, limit, heals_left, round_no = spec.version, cfg.max_iterations, cfg.heal_rounds, 0
    checking_heal = False
    while True:
        round_no += 1
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
        train_by_round.append(res_train)
        run.emit(
            "eval",
            f"Round {round_no}: training {s_train:.0%}, held-out {s_hold:.0%}",
            round=round_no,
            version=spec.version,
            train_score=s_train,
            holdout_score=s_hold,
            results=[
                {k: r[k] for k in ("pass", "score", "reason", "ms")}
                | {"id": r["case"]["id"], "input": r["case"]["input"], "output": r["output"][:1500], "holdout": held, "routes": _routes(r["trace"])}
                for held, group in ((False, res_train), (True, res_hold))
                for r in group
            ],
        )
        passed = s_train >= cfg.pass_threshold and (not holdout or s_hold >= cfg.pass_threshold)
        if checking_heal:
            checking_heal = False
            h = heals[-1]
            h["after"] = {"train": s_train, "holdout": s_hold}
            h["fixed"] = passed
            h["improved"] = (s_train + s_hold) > (h["before"]["train"] + h["before"]["holdout"])
            verdict = "Self-heal worked" if passed else "Self-heal improved the score" if h["improved"] else "Self-heal did not help"
            run.emit(
                "healer",
                f"{verdict}: training {h['before']['train']:.0%} to {s_train:.0%}, held-out {h['before']['holdout']:.0%} to {s_hold:.0%}",
                kind="result",
                fixed=passed,
                improved=h["improved"],
                before=h["before"],
                after=h["after"],
            )
        rank = (s_train + s_hold, s_hold)
        if best is None or rank > (best[0], best[1]):
            best = (rank[0], rank[1], spec, res_train)
        if passed:
            run.emit("eval", "Target reached, stopping")
            break
        if round_no < limit:
            run.emit("optimizer", f"{big} is diagnosing the failures")
            try:
                spec, info = optimizer.improve(llm, spec, res_train)
                spec = _repair(_forge_missing(llm, spec, run, cfg))
            except BudgetExceeded as err:
                run.emit("budget", f"Stopping early: {err}. Keeping the best version so far.")
                break
            except Exception as err:  # a broken Optimizer reply should not sink the build
                run.emit("healer", f"The Optimizer failed ({type(err).__name__}: {str(err)[:200]}). The Healer takes over.", kind="optimizer")
            else:
                top_version = max(top_version, spec.version)
                spec.save(out_dir / f"spec_v{spec.version}.json")
                entry["optimizer"] = info
                run.emit("optimizer", info["diagnosis"] or "Updated the design", changes=info["changes"], spec=spec.model_dump())
                continue
        if heals_left <= 0:
            break
        heals_left -= 1
        run.emit("healer", f"Still below target after {round_no} rounds. {big} is reviewing every round to find the root cause", kind="stuck", round=round_no)
        try:
            spec, info = healer.heal_stuck(llm, best[2], history, train_by_round, best[3], top_version + 1)
            spec = _repair(_forge_missing(llm, spec, run, cfg))
        except BudgetExceeded as err:
            run.emit("budget", f"Stopping early: {err}. Keeping the best version so far.")
            break
        except Exception as err:
            run.emit("healer", f"The Healer could not repair this build ({type(err).__name__}: {str(err)[:200]}). Keeping the best version.", kind="stuck")
            break
        top_version = spec.version
        spec.save(out_dir / f"spec_v{spec.version}.json")
        best_round = next(h for h in history if h["version"] == best[2].version)
        heals.append({"kind": "stuck", "round": round_no, "to_version": spec.version, "before": {"train": best_round["train"], "holdout": best_round["holdout"]}, **info})
        entry["healer"] = info
        run.emit(
            "healer",
            info["diagnosis"] or "Redesigned the team",
            kind="stuck",
            root_causes=info["root_causes"],
            changes=info["changes"],
            overfitting=info["overfitting"],
            always_failing=info["always_failing"],
            from_version=info["from_version"],
            spec=spec.model_dump(),
        )
        checking_heal = True
        limit = round_no + 1  # one bonus round to prove the fix

    assert best is not None
    final = best[2]
    final.save(out_dir / "spec.json")
    last = next(h for h in reversed(history) if h["version"] == final.version)
    passed = last["train"] >= cfg.pass_threshold and (not holdout or last["holdout"] >= cfg.pass_threshold)
    healed = passed and any(h.get("fixed") or h["kind"] == "crash" for h in heals)
    report = {
        "goal": goal,
        "name": final.name,
        "version": final.version,
        "train_score": last["train"],
        "holdout_score": last["holdout"],
        "rounds": [
            {k: h[k] for k in ("round", "version", "train", "holdout")}
            | ({"changes": h["optimizer"]["changes"]} if "optimizer" in h else {})
            | ({"healed": h["healer"]["changes"]} if "healer" in h else {})
            for h in history
        ],
        "healing": heals,
        "self_healed": healed,
        "usage": run.usage,
        "cost_usd": run.cost_usd,
        "seconds": round(time.time() - started, 1),
        "build_dir": str(out_dir),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    run.result = report
    tag = " (self-healed)" if healed else ""
    run.emit("done", f"Built '{final.name}' v{final.version}{tag}: training {last['train']:.0%}, held-out {last['holdout']:.0%}", report=report, spec=final.model_dump())
    run.status = "passed" if passed else "below_target"
    (out_dir / "status").write_text(run.status, encoding="utf-8")
    return report


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(c) for c in data["train"] + data["holdout"]]
