"""Observability logs and dashboard KPIs, computed from the builds stored on disk.

Every number here is derived from recorded events (model calls, eval rounds, the final
report); nothing is estimated. Severity comes from fixed rules, and each record carries
the reasons that set it, so the dashboard can show why.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import settings
from .events import Run

SEVERITIES = ["critical", "high", "medium", "low", "info"]
_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def _worst(reasons: list[tuple[str, str]]) -> str:
    return min((s for s, _ in reasons), key=_RANK.__getitem__, default="info")


def _by_stage(run: Run, stage: str) -> list[dict[str, Any]]:
    return [e for e in run.events if e.get("stage") == stage]


def _rounds(run: Run) -> list[dict[str, Any]]:
    return [e["data"] for e in _by_stage(run, "eval") if "results" in e.get("data", {})]


def _report(run: Run) -> dict[str, Any] | None:
    done = _by_stage(run, "done")
    return done[-1]["data"].get("report") if done else None


def _shipped_round(run: Run) -> dict[str, Any] | None:
    """The eval round of the version that shipped (the best one), else the latest round."""
    rounds = _rounds(run)
    report = _report(run)
    if report is not None:
        for r in reversed(rounds):
            if r.get("version") == report.get("version"):
                return r
    return rounds[-1] if rounds else None


def heals(run: Run) -> list[dict[str, Any]]:
    """Each repair the Healer made in this build, and whether it fixed the build."""
    out = []
    for e in _by_stage(run, "healer"):
        d = e.get("data", {})
        if d.get("kind") == "stuck" and "root_causes" in d:
            out.append({"kind": "stuck", "message": e["message"], "fixed": None})
        elif d.get("kind") == "result" and out and out[-1]["kind"] == "stuck":
            out[-1]["fixed"] = bool(d.get("fixed"))
            out[-1]["result"] = e["message"]
        elif d.get("kind") == "crash" and "fix_notes" in d:
            out.append({"kind": "crash", "message": e["message"], "fixed": None})
    # A crash repair worked if the build went on to finish.
    for h in out:
        if h["kind"] == "crash":
            h["fixed"] = run.status in ("passed", "below_target")
    return out


def overfit_events(run: Run) -> list[tuple[int, int]]:
    """Rounds where training held or rose but held-out dropped: overfitting the held-out tests caught."""
    rounds = [r for r in _rounds(run) if r.get("holdout_score") is not None]
    return [
        (a["version"], b["version"])
        for a, b in zip(rounds, rounds[1:])
        if b["train_score"] >= a["train_score"] and b["holdout_score"] < a["holdout_score"]
    ]


def _duration(run: Run) -> float | None:
    report = _report(run)
    if report and report.get("seconds") is not None:
        return float(report["seconds"])
    if run.status != "running" and len(run.events) > 1:
        return round(run.events[-1]["t"] - run.events[0]["t"], 1)
    return None


def _build_record(run: Run) -> dict[str, Any]:
    reasons: list[tuple[str, str]] = []
    report = _report(run)
    errors = _by_stage(run, "error")
    if run.status == "running":
        reasons.append(("info", "Build is still running."))
    elif run.status == "interrupted":
        reasons.append(("medium", "Build was interrupted: the server stopped before it finished. Rebuild to get a result."))
    elif run.status == "failed" or report is None:
        msg = errors[-1]["message"] if errors else "the build stopped before it finished"
        reasons.append(("critical", f"Build crashed: {msg}"))
    else:
        hold, train = report.get("holdout_score"), report.get("train_score", 0.0)
        if run.status == "below_target":
            reasons.append(("medium", f"Shipped v{report['version']} is below the pass target."))
        if hold is not None and hold < 0.5:
            reasons.append(("high", f"Held-out score of the shipped version is {hold:.0%}: it may not work on new inputs."))
        if hold is not None and train - hold >= 0.5:
            reasons.append(("high", f"Training {train:.0%} vs held-out {hold:.0%}: a gap this big suggests overfitting."))
        if hold is not None and hold - train >= 0.5:
            reasons.append(("medium", f"Held-out {hold:.0%} vs training {train:.0%}: too few held-out tests to trust the gap."))
        for a, b in overfit_events(run):
            kept = "so NemoForge shipped an earlier version" if b != report["version"] else "but it still ranked best overall"
            reasons.append(("low", f"Overfitting caught: v{b} did as well or better on training but worse on held-out than v{a}, {kept}."))
        for e in _by_stage(run, "budget"):
            reasons.append(("low", e["message"]))
        for h in heals(run):
            if h["kind"] == "crash":
                reasons.append(("low", f"Self-healed a crash: {h['message']}"))
            elif h["fixed"]:
                reasons.append(("info", f"{h['result']}."))
            else:
                reasons.append(("low", f"Healer tried a repair: {h.get('result') or h['message']}."))
        if run.status == "passed" and not reasons:
            reasons.append(("info", "Passed training and held-out targets."))
    return {
        "id": f"{run.id}:build",
        "kind": "build",
        "build": run.id,
        "t": run.created,
        "input": run.target,
        "output": (report or {}).get("name", ""),
        "detail": {
            "status": run.status,
            "version": (report or {}).get("version"),
            "train_score": (report or {}).get("train_score"),
            "holdout_score": (report or {}).get("holdout_score"),
            "cost_usd": _cost(run),
            "seconds": _duration(run),
            "model_calls": len(_model_calls(run)),
        },
        "severity": _worst(reasons),
        "reasons": [{"severity": s, "text": t} for s, t in reasons],
    }


def _test_records(run: Run) -> list[dict[str, Any]]:
    shipped = _shipped_round(run)
    if shipped is None:
        return []
    t = next((e["t"] for e in _by_stage(run, "eval") if e.get("data") is shipped), run.created)
    out = []
    for r in shipped["results"]:
        kind = "held-out" if r.get("holdout") else "training"
        if not r.get("pass"):
            sev = "high" if r.get("holdout") else "medium"
            why = f"Failed a {kind} test (score {r.get('score', 0):.0%})." + (
                " Held-out tests are never shown to the optimizer, so this is how the agent does on new inputs." if r.get("holdout") else ""
            )
        elif (r.get("score") or 0) < 1:
            sev, why = "low", f"Passed the {kind} test with a partial score of {r.get('score', 0):.0%}."
        else:
            sev, why = "info", f"Passed the {kind} test."
        reasons = [{"severity": sev, "text": why}]
        if r.get("reason"):
            reasons.append({"severity": "info", "text": f"Grader: {r['reason']}"})
        out.append({
            "id": f"{run.id}:test:{r.get('id')}",
            "kind": "test",
            "build": run.id,
            "t": t,
            "input": r.get("input", ""),
            "output": r.get("output", ""),
            "detail": {"case": r.get("id"), "set": kind, "version": shipped.get("version"), "score": r.get("score"), "ms": r.get("ms")},
            "severity": sev,
            "reasons": reasons,
        })
    return out


def invocations_path(builds_dir: Path, run_id: str) -> Path:
    return builds_dir / run_id / "invocations.jsonl"


def _try_records(run: Run, builds_dir: Path) -> list[dict[str, Any]]:
    path = invocations_path(builds_dir, run.id)
    if not path.exists():
        return []
    out = []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i, line in enumerate(lines, 1):
        rec = json.loads(line)
        if rec.get("error"):
            reasons = [{"severity": "critical", "text": f"The agent team crashed: {rec['error']}"}]
        elif not (rec.get("output") or "").strip():
            reasons = [{"severity": "medium", "text": "The agent team returned an empty answer."}]
        else:
            reasons = [{"severity": "info", "text": "Answered without errors. Playground runs are not graded."}]
        out.append({
            "id": f"{run.id}:try:{i}",
            "kind": "try",
            "build": run.id,
            "t": rec.get("t", run.created),
            "input": rec.get("input", ""),
            "output": rec.get("output") or rec.get("error") or "",
            "detail": {"ms": rec.get("ms"), "agents": rec.get("agents")},
            "severity": reasons[0]["severity"],
            "reasons": reasons,
        })
    return out


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "..."


def memory(run: Run, builds_dir: Path, recent: int = 5) -> dict[str, Any]:
    """What an agent remembers: lessons from its own build, and its latest playground runs."""
    lessons: list[dict[str, Any]] = []
    for e in run.events:
        d = e.get("data", {})
        if e.get("stage") == "optimizer" and "changes" in d:
            lessons.append({"source": "optimizer", "text": e["message"], "changes": d.get("changes", [])})
        elif e.get("stage") == "healer" and d.get("root_causes"):
            lessons.append({"source": "healer", "text": e["message"], "changes": d.get("root_causes", [])})
    runs = []
    path = invocations_path(builds_dir, run.id)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("output") and not rec.get("error"):
                runs.append({"t": rec.get("t"), "input": _clip(rec.get("input", ""), 600), "output": _clip(rec["output"], 1200)})
    return {"lessons": lessons, "runs": runs[-recent:][::-1], "total_runs": len(runs)}


def memory_context(mem: dict[str, Any], n: int = 3) -> str:
    """Earlier runs written as context for the next run (oldest first)."""
    if not mem["runs"]:
        return ""
    parts = [f"## Earlier run {i}\nInput: {r['input']}\nAnswer: {r['output']}" for i, r in enumerate(reversed(mem["runs"][:n]), 1)]
    return "# Memory: this agent's earlier runs. Use them only when relevant (for example, follow up on open items from a previous meeting).\n\n" + "\n\n".join(parts)


def logs(runs: Iterable[Run], builds_dir: Path) -> list[dict[str, Any]]:
    """One record per input NemoForge handled: build goals, graded test cases and playground runs."""
    runs = list(runs)
    records: list[dict[str, Any]] = []
    for run in runs:
        if run.kind != "build":
            continue
        records.append(_build_record(run))
        records += _test_records(run)
        records += _try_records(run, builds_dir)
    # Newest build first; inside a build, the goal then its tests then playground runs.
    order = {"build": 0, "test": 1, "try": 2}
    created = {r.id: r.created for r in runs if r.kind == "build"}
    records.sort(key=lambda r: (-created.get(r["build"], r["t"]), order[r["kind"]], r["t"]))
    return records


def _model_calls(run: Run) -> list[dict[str, Any]]:
    return [e["data"] for e in _by_stage(run, "model") if "prompt_tokens" in e.get("data", {})]


def _cost(run: Run) -> float:
    calls = _model_calls(run)
    if calls:
        return round(sum(c.get("cost_usd", 0.0) for c in calls), 6)
    return run.cost_usd


def stats(runs: Iterable[Run], builds_dir: Path) -> dict[str, Any]:
    runs = [r for r in runs if r.kind == "build"]
    finished = [r for r in runs if r.status not in ("running", "interrupted")]
    reports = [rep for rep in (_report(r) for r in runs) if rep]
    calls = [c for r in runs for c in _model_calls(r)]
    by_tier: dict[str, int] = {}
    for c in calls:
        by_tier[c.get("tier", "?")] = by_tier.get(c.get("tier", "?"), 0) + 1
    graded = [res for r in runs for rnd in _rounds(r) for res in rnd["results"]]
    holds = [rep["holdout_score"] for rep in reports if rep.get("holdout_score") is not None]
    trains = [rep["train_score"] for rep in reports if rep.get("train_score") is not None]
    durations = [d for d in (_duration(r) for r in finished) if d is not None]
    records = logs(runs, builds_dir)
    sev_counts = {s: sum(1 for x in records if x["severity"] == s) for s in SEVERITIES}
    status = {s: sum(1 for r in runs if r.status == s) for s in ("passed", "below_target", "failed", "interrupted", "running")}
    return {
        "builds": len(runs),
        "status": status,
        "pass_rate": status["passed"] / len(finished) if finished else None,
        "avg_holdout": sum(holds) / len(holds) if holds else None,
        "avg_train": sum(trains) / len(trains) if trains else None,
        "model_calls": len(calls),
        "calls_by_tier": by_tier,
        "tokens": sum(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0) for c in calls),
        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
        "completion_tokens": sum(c.get("completion_tokens", 0) for c in calls),
        "tests_graded": len(graded),
        "tests_passed": sum(1 for g in graded if g.get("pass")),
        "total_cost_usd": round(sum(_cost(r) for r in runs), 6),
        "avg_seconds": sum(durations) / len(durations) if durations else None,
        "overfit_caught": sum(len(overfit_events(r)) for r in runs),
        "overfit_builds": sum(1 for r in runs if overfit_events(r)),
        "heals": sum(len(heals(r)) for r in runs),
        "heals_fixed": sum(1 for r in runs for h in heals(r) if h["fixed"]),
        "alerts": sev_counts["critical"] + sev_counts["high"],
        "severity": sev_counts,
        "records": len(records),
        "routing": routing(calls) | {"smart": smart_routes(runs, builds_dir)},
    }


def smart_routes(runs: Iterable[Run], builds_dir: Path) -> dict[str, Any]:
    """What the smart router decided, across every test run and playground run."""
    decisions: list[dict[str, Any]] = []
    for run in runs:
        for e in _by_stage(run, "eval"):
            for r in e.get("data", {}).get("results", []) or []:
                decisions += r.get("routes", []) or []
        path = invocations_path(builds_dir, run.id)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    decisions += json.loads(line).get("routes", []) or []
                except (ValueError, AttributeError):
                    continue
    order = {"nano": 0, "super": 1, "ultra": 2}
    rated = [d for d in decisions if d.get("difficulty")]
    return {
        "decisions": len(decisions),
        "by_tier": {t: sum(1 for d in decisions if d.get("tier") == t) for t in ("nano", "super", "ultra")},
        "down": sum(1 for d in decisions if order.get(d.get("tier"), 0) < order.get(d.get("planned"), 0)),
        "up": sum(1 for d in decisions if order.get(d.get("tier"), 0) > order.get(d.get("planned"), 0)),
        "avg_difficulty": sum(d["difficulty"] for d in rated) / len(rated) if rated else None,
        "recent": [{k: d.get(k) for k in ("agent", "planned", "tier", "difficulty", "why")} for d in decisions[-6:]][::-1],
    }


def routing(calls: list[dict[str, Any]], prices: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    """What tier routing saved: the recorded cost against the same tokens all sent to Ultra at list price."""
    prices = prices or settings().prices
    p_in, p_out = prices["ultra"]
    tiers: dict[str, dict[str, Any]] = {}
    for c in calls:
        t = tiers.setdefault(c.get("tier", "?"), {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        t["calls"] += 1
        t["tokens"] += c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)
        t["cost_usd"] += c.get("cost_usd", 0.0)
    actual = sum(t["cost_usd"] for t in tiers.values())
    all_ultra = sum(c.get("prompt_tokens", 0) * p_in + c.get("completion_tokens", 0) * p_out for c in calls) / 1e6
    return {
        "tiers": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in tiers.items()},
        "actual_usd": round(actual, 6),
        "all_ultra_usd": round(all_ultra, 6),
        "saved_usd": round(max(all_ultra - actual, 0.0), 6),
        "saved_pct": (all_ultra - actual) / all_ultra if all_ultra else None,
    }


def agents(runs: Iterable[Run]) -> list[dict[str, Any]]:
    """The agent registry: one entry per build, newest first, with the team that shipped."""
    out = []
    for run in sorted((r for r in runs if r.kind == "build"), key=lambda r: r.created, reverse=True):
        report = _report(run) or {}
        done = _by_stage(run, "done")
        spec = done[-1]["data"].get("spec") if done else None
        if spec is None:  # still running or crashed: show the Architect's latest design, if any
            spec = next((e["data"]["spec"] for e in reversed(run.events) if "spec" in e.get("data", {})), None)
        team = []
        if spec:
            tiers = {a["name"]: a.get("tier", "nano") for a in spec.get("agents", [])}
            team = [{"name": n, "tier": tiers.get(n, "nano")} for n in spec.get("pipeline", [])]
        out.append({
            "build": run.id,
            "name": report.get("name") or (spec or {}).get("name") or "(designing)",
            "goal": run.target,
            "summary": (spec or {}).get("summary", ""),
            "status": run.status,
            "version": report.get("version"),
            "train_score": report.get("train_score"),
            "holdout_score": report.get("holdout_score"),
            "cost_usd": _cost(run),
            "created": run.created,
            "team": team,
            "tools": len((spec or {}).get("tools", [])),
        })
    return out
