"""Package a finished build as a container and deploy it to a Nebius Serverless Endpoint.

Steps: copy the runtime + spec.json into a build context, `docker build`, `docker push`
to your registry, then `nebius ai endpoint create` on a CPU preset (the agents call
Nemotron through Token Factory, so the endpoint itself needs no GPU).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .config import settings

ROOT = Path(__file__).resolve().parent.parent

DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic openai httpx
COPY nemoforge/ nemoforge/
COPY spec.json spec.json
ENV NF_SPEC=/app/spec.json
EXPOSE 8080
CMD ["uvicorn", "nemoforge.serve:app", "--host", "0.0.0.0", "--port", "8080"]
"""


def package(build_dir: Path) -> Path:
    spec = build_dir / "spec.json"
    if not spec.exists():
        raise FileNotFoundError(f"{spec} not found: finish a build first")
    ctx = build_dir / "image"
    if ctx.exists():
        shutil.rmtree(ctx)
    shutil.copytree(ROOT / "nemoforge", ctx / "nemoforge", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(spec, ctx / "spec.json")
    (ctx / "Dockerfile").write_text(DOCKERFILE)
    return ctx


def commands(build_dir: Path, registry: str, subnet_id: str = "", project_id: str = "") -> list[list[str]]:
    name = json.loads((build_dir / "spec.json").read_text())["name"].replace("_", "-")[:40]
    image = f"{registry.rstrip('/')}/nf-{name}:{build_dir.name}"
    create = [
        "nebius", "ai", "endpoint", "create",
        "--name", f"nf-{name}",
        "--image", image,
        "--platform", os.environ.get("NF_NEBIUS_PLATFORM", "cpu-d3"),
        "--preset", os.environ.get("NF_NEBIUS_PRESET", "4vcpu-16gb"),
        "--container-port", "8080",
        "--auth", "token",
        "--token", "${NF_ENDPOINT_TOKEN}",
        "--env", "NEBIUS_API_KEY=${NEBIUS_API_KEY}",
        "--env", "TAVILY_API_KEY=${TAVILY_API_KEY}",
    ]
    if subnet_id:
        create += ["--subnet-id", subnet_id]
    if project_id:
        create += ["--parent-id", project_id]
    ctx = str(build_dir / "image")
    return [["docker", "build", "-t", image, ctx], ["docker", "push", image], create]


def deploy(build_dir: Path, registry: str, subnet_id: str = "", project_id: str = "", dry_run: bool = True) -> list[str]:
    package(build_dir)
    cmds = commands(build_dir, registry, subnet_id, project_id)
    printable = [" ".join(c) for c in cmds]
    if dry_run:
        return printable
    cfg = settings()
    env = {**os.environ, "NEBIUS_API_KEY": cfg.nebius_api_key, "TAVILY_API_KEY": cfg.tavily_api_key}
    for c in cmds:
        expanded = [os.path.expandvars(part) if "${" in part else part for part in c]
        subprocess.run(expanded, check=True, env=env)
    return printable
