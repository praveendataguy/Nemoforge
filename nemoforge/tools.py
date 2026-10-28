"""Tools agents can call: a few built-ins, plus Python tools the Toolsmith writes.

Generated tool code never runs inside the NemoForge process. Each call starts a fresh
`python -I` subprocess with a stripped environment (no API keys), a temp working dir,
CPU/memory limits and a wall-clock timeout. On Linux, if `unshare` is allowed, the
subprocess also gets no network.
"""
from __future__ import annotations

import ast
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

import httpx

from .config import Settings, settings as load_settings
from .spec import ToolSpec

# ---------------------------------------------------------------- built-ins


def _web_search(args: dict[str, Any], cfg: Settings) -> Any:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "query is required"}
    if not cfg.tavily_api_key:
        return {"error": "web search unavailable: TAVILY_API_KEY not set"}
    resp = httpx.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {cfg.tavily_api_key}"},
        json={"query": query, "max_results": int(args.get("max_results", 5)), "include_answer": True},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data.get("answer"),
        "results": [
            {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:800]}
            for r in data.get("results", [])
        ],
    }


def _fetch_url(args: dict[str, Any], cfg: Settings) -> Any:
    url = str(args.get("url", ""))
    if not url.startswith(("http://", "https://")):
        return {"error": "url must start with http:// or https://"}
    if cfg.tavily_api_key:
        resp = httpx.post(
            "https://api.tavily.com/extract",
            headers={"Authorization": f"Bearer {cfg.tavily_api_key}"},
            json={"urls": [url]},
            timeout=40,
        )
        if resp.status_code == 200 and resp.json().get("results"):
            return {"url": url, "content": resp.json()["results"][0].get("raw_content", "")[:6000]}
    resp = httpx.get(url, timeout=20, follow_redirects=True)
    return {"url": url, "status": resp.status_code, "content": resp.text[:6000]}


_MATH_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
)


def _calculator(args: dict[str, Any], cfg: Settings) -> Any:
    expr = str(args.get("expression", ""))
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return {"error": f"unsupported syntax: {type(node).__name__}"}
        if isinstance(node, ast.Name) and node.id not in _MATH_NAMES:
            return {"error": f"unknown name: {node.id}"}
    return {"result": eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _MATH_NAMES)}


BUILTINS: dict[str, tuple[ToolSpec, Callable[[dict[str, Any], Settings], Any]]] = {
    "web_search": (
        ToolSpec(
            name="web_search",
            kind="builtin",
            description="Search the web (Tavily). Returns a short answer and the top results with URLs.",
            parameters={"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}},
        ),
        _web_search,
    ),
    "fetch_url": (
        ToolSpec(
            name="fetch_url",
            kind="builtin",
            description="Fetch the readable text of a web page (Tavily extract, falls back to plain GET).",
            parameters={"url": {"type": "string"}},
        ),
        _fetch_url,
    ),
    "calculator": (
        ToolSpec(
            name="calculator",
            kind="builtin",
            description="Evaluate an arithmetic expression exactly, e.g. '1200*0.15 + sqrt(16)'.",
            parameters={"expression": {"type": "string"}},
        ),
        _calculator,
    ),
}


def builtin_names() -> set[str]:
    return set(BUILTINS)


# ------------------------------------------------------- sandboxed python tools

_HARNESS = r"""
import json, sys
sys.path.insert(0, ".")
args = json.loads(sys.stdin.read() or "{}")
import tool
out = tool.run(**args)
sys.stdout.write("\n__NF_RESULT__" + json.dumps(out, default=str))
"""


try:
    import resource
except ImportError:  # Windows: no rlimits, the timeout still applies
    resource = None  # type: ignore[assignment]


def _limits() -> None:  # runs in the child before exec
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    os.setsid()


_NO_NET: list[str] | None = None


def _no_net_prefix() -> list[str]:
    """Use `unshare -rn` (no network namespace) when the host allows it."""
    global _NO_NET
    if _NO_NET is None:
        _NO_NET = []
        if shutil.which("unshare"):
            probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True)
            if probe.returncode == 0:
                _NO_NET = ["unshare", "-rn"]
    return _NO_NET


def run_python_tool(code: str, args: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nf-tool-") as tmp:
        with open(os.path.join(tmp, "tool.py"), "w", encoding="utf-8") as f:
            f.write(code)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name == "nt":  # Python will not start on Windows without SYSTEMROOT
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        cmd = [*_no_net_prefix(), sys.executable, "-I", "-c", _HARNESS]
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(args),
                capture_output=True,
                text=True,
                cwd=tmp,
                env=env,
                timeout=timeout,
                preexec_fn=_limits if resource is not None else None,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"tool timed out after {timeout}s"}
    if "__NF_RESULT__" in proc.stdout:
        printed, _, payload = proc.stdout.rpartition("\n__NF_RESULT__")
        try:
            return {"result": json.loads(payload), **({"stdout": printed[-1000:]} if printed.strip() else {})}
        except json.JSONDecodeError:
            return {"error": "tool returned non-JSON output"}
    return {"error": (proc.stderr or proc.stdout or f"exit code {proc.returncode}")[-2000:]}


def call_tool(tool: ToolSpec | None, name: str, args: dict[str, Any], cfg: Settings | None = None) -> dict[str, Any]:
    cfg = cfg or load_settings()
    if name in BUILTINS:
        try:
            out = BUILTINS[name][1](args, cfg)
        except Exception as err:  # surface tool errors to the agent instead of crashing the run
            return {"error": f"{type(err).__name__}: {err}"}
        return out if isinstance(out, dict) and "error" in out else {"result": out}
    if tool is None or tool.kind != "python":
        return {"error": f"unknown tool '{name}'"}
    return run_python_tool(tool.code, args, timeout=cfg.tool_timeout)


def describe(tools: list[ToolSpec]) -> str:
    lines = []
    for t in tools:
        params = ", ".join(f"{k}: {v.get('type', 'any') if isinstance(v, dict) else v}" for k, v in t.parameters.items())
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines) or "(no tools)"
