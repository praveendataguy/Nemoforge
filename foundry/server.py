"""Agent Foundry web app: start builds, watch them live, try the result, deploy it.

    uvicorn foundry.server:app --port 8000
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import deploy as deployer
from .config import settings
from .events import Run, registry
from .foundry import build
from .llm import LLM, NemotronRouter
from .runtime import run_system
from .spec import SystemSpec

WEB = Path(__file__).resolve().parent.parent / "web"

# Swappable so tests can run the app without network access.
llm_factory: Callable[[Run | None], LLM] = lambda run: NemotronRouter(run)

app = FastAPI(title="Agent Foundry")


class BuildRequest(BaseModel):
    goal: str = Field(min_length=8, max_length=2000)
    notes: str = Field(default="", max_length=4000)


class InvokeRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20000)


class DeployRequest(BaseModel):
    registry: str
    subnet_id: str = ""
    project_id: str = ""
    dry_run: bool = True


def _build_dir(run_id: str) -> Path:
    d = settings().builds_dir / run_id
    if not (d / "spec.json").exists():
        raise HTTPException(404, "build not finished or not found")
    return d


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/config")
def config() -> dict:
    cfg = settings()
    return {
        "models": {"nano": cfg.models.nano, "super": cfg.models.super, "ultra": cfg.models.ultra},
        "nebius_key": bool(cfg.nebius_api_key),
        "tavily_key": bool(cfg.tavily_api_key),
        "pass_threshold": cfg.pass_threshold,
        "max_iterations": cfg.max_iterations,
    }


@app.post("/api/builds")
def start_build(req: BuildRequest) -> dict:
    run = registry.new("build", req.goal)

    def work() -> None:
        try:
            build(llm_factory(run), req.goal, run, notes=req.notes)
        except Exception as err:
            run.emit("error", f"{type(err).__name__}: {err}", traceback=traceback.format_exc()[-3000:])
            run.status = "failed"

    threading.Thread(target=work, daemon=True, name=f"build-{run.id}").start()
    return run.summary()


@app.get("/api/builds")
def list_builds() -> list[dict]:
    return registry.list()


@app.get("/api/builds/{run_id}")
def get_build(run_id: str) -> dict:
    run = registry.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown build")
    return {**run.summary(), "events": run.events}


@app.get("/api/builds/{run_id}/events")
async def stream(run_id: str) -> StreamingResponse:
    run = registry.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown build")

    async def gen():
        async for e in run.stream():
            yield f"data: {json.dumps(e, default=str)}\n\n"
        yield f"data: {json.dumps({'stage': 'end', 'status': run.status})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/builds/{run_id}/invoke")
def invoke(run_id: str, req: InvokeRequest) -> dict:
    spec = SystemSpec.load(_build_dir(run_id) / "spec.json")
    out = run_system(llm_factory(None), spec, req.input, settings())
    return out


@app.post("/api/builds/{run_id}/deploy")
def deploy(run_id: str, req: DeployRequest) -> dict:
    d = _build_dir(run_id)
    try:
        cmds = deployer.deploy(d, req.registry, req.subnet_id, req.project_id, dry_run=req.dry_run)
    except Exception as err:
        raise HTTPException(500, f"deploy failed: {err}")
    return {"dry_run": req.dry_run, "commands": cmds, "image_context": str(d / "image")}
