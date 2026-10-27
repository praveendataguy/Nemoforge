"""Standalone server for ONE built agent system. This is what gets deployed to a
Nebius Serverless Endpoint: the generic runtime plus the build's spec.json.

    AF_SPEC=builds/<id>/spec.json uvicorn foundry.serve:app --port 8080
"""
from __future__ import annotations

import html
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import settings
from .llm import NemotronRouter
from .runtime import run_system
from .spec import SystemSpec

SPEC_PATH = Path(os.environ.get("AF_SPEC", "spec.json"))
spec = SystemSpec.load(SPEC_PATH)
app = FastAPI(title=spec.name, description=spec.summary)


class Invoke(BaseModel):
    input: str
    trace: bool = False


@app.get("/health")
def health() -> dict:
    return {"ok": True, "system": spec.name, "version": spec.version}


@app.get("/spec")
def get_spec() -> dict:
    return spec.model_dump(exclude={"tools": {"__all__": {"tests"}}})


@app.post("/invoke")
def invoke(body: Invoke) -> dict:
    if not body.input.strip():
        raise HTTPException(400, "input is empty")
    router = NemotronRouter()
    out = run_system(router, spec, body.input, settings())
    return {"output": out["output"], "ms": out["ms"], **({"trace": out["trace"]} if body.trace else {})}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    e = html.escape
    agents = "".join(f"<li><b>{e(a.name)}</b> ({a.tier}): {e(a.role)}</li>" for a in spec.agents)
    return f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{e(spec.name)}</title>
<style>body{{font:16px system-ui;max-width:720px;margin:2rem auto;padding:0 16px}}textarea{{width:100%;height:8rem}}
pre{{white-space:pre-wrap;background:#f4f4f2;padding:1rem;border-radius:8px}}</style>
<h1>{e(spec.name.replace('_', ' '))}</h1><p>{e(spec.summary)}</p><ul>{agents}</ul>
<p><small>Built by Agent Foundry · runs on NVIDIA Nemotron via Nebius Token Factory</small></p>
<textarea id=q placeholder="{e(spec.input_hint)}"></textarea><br><button id=go>Run</button><pre id=out></pre>
<script>go.onclick=async()=>{{out.textContent='Running...';const r=await fetch('invoke',{{method:'POST',
headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{input:q.value}})}});
const j=await r.json();out.textContent=j.output||JSON.stringify(j)}}</script>"""
