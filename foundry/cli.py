"""Command line: `python -m foundry build "goal"`, `run`, `deploy`, `serve`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="foundry", description="Agent Foundry: one sentence in, a tested agent team out.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="design, test and improve an agent team for a goal")
    b.add_argument("goal")
    b.add_argument("--notes", default="")

    r = sub.add_parser("run", help="run a finished build on one input")
    r.add_argument("build_id")
    r.add_argument("input")
    r.add_argument("--trace", action="store_true")

    d = sub.add_parser("deploy", help="package a build and deploy it to a Nebius Serverless Endpoint")
    d.add_argument("build_id")
    d.add_argument("--registry", required=True, help="e.g. cr.eu-north1.nebius.cloud/<registry-id>")
    d.add_argument("--subnet-id", default="")
    d.add_argument("--project-id", default="")
    d.add_argument("--yes", action="store_true", help="actually run the commands (default prints them)")

    s = sub.add_parser("serve", help="start the Agent Foundry web app")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")

    args = p.parse_args(argv)
    cfg = settings()

    if args.cmd == "build":
        from .events import registry
        from .foundry import build
        from .llm import NemotronRouter

        run = registry.new("build", args.goal)
        report = build(NemotronRouter(run), args.goal, run, cfg, notes=args.notes)
        print(json.dumps(report, indent=2))
        return 0 if run.status == "passed" else 2

    if args.cmd == "run":
        from .llm import NemotronRouter
        from .runtime import run_system
        from .spec import SystemSpec

        spec = SystemSpec.load(cfg.builds_dir / args.build_id / "spec.json")
        out = run_system(NemotronRouter(), spec, args.input, cfg)
        print(json.dumps(out, indent=2) if args.trace else out["output"])
        return 0

    if args.cmd == "deploy":
        from .deploy import deploy

        for line in deploy(cfg.builds_dir / args.build_id, args.registry, args.subnet_id, args.project_id, dry_run=not args.yes):
            print(line)
        return 0

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("foundry.server:app", host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
