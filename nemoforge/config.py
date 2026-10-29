"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from .env so keys work in any new terminal. Real environment variables win."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and not os.environ.get(key):
            os.environ[key] = value


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Models:
    """Nemotron tiers on Nebius Token Factory. Override with env vars if the catalog IDs change."""

    nano: str = field(default_factory=lambda: _env("NF_MODEL_NANO", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"))
    super: str = field(default_factory=lambda: _env("NF_MODEL_SUPER", "nvidia/nemotron-3-super-120b-a12b"))
    ultra: str = field(default_factory=lambda: _env("NF_MODEL_ULTRA", "nvidia/Nemotron-3-Ultra-550b-a55b"))


def _price(tier: str, default_in: float, default_out: float) -> tuple[float, float]:
    """USD per 1M tokens (input, output). Defaults are approximate list prices; override per tier."""
    return (float(_env(f"NF_PRICE_{tier}_IN", str(default_in))), float(_env(f"NF_PRICE_{tier}_OUT", str(default_out))))


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str = field(default_factory=lambda: _env("NEBIUS_API_KEY"))
    nebius_base_url: str = field(default_factory=lambda: _env("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"))
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    builds_dir: Path = field(default_factory=lambda: Path(_env("NF_BUILDS_DIR", "builds")))
    # Optimise until this share of eval cases pass, or until max_iterations rounds. 0.85 allows one miss in 8.
    pass_threshold: float = field(default_factory=lambda: float(_env("NF_PASS_THRESHOLD", "0.85")))
    max_iterations: int = field(default_factory=lambda: int(_env("NF_MAX_ITERATIONS", "4")))
    # Self-healing: bonus Healer rounds when a build is stuck below target, and rebuilds after a crash.
    heal_rounds: int = field(default_factory=lambda: int(_env("NF_HEAL_ROUNDS", "1")))
    heal_retries: int = field(default_factory=lambda: int(_env("NF_HEAL_RETRIES", "1")))
    # Smart routing: Nano rates each request before every agent runs and picks Nano, Super or Ultra.
    smart_routing: bool = field(default_factory=lambda: _env("NF_SMART_ROUTING", "1") in ("1", "true", "yes"))
    # Nemotron reasoning per tier: Ultra and Super think step by step before answering, Nano answers directly.
    thinking: dict[str, bool] = field(
        default_factory=lambda: {
            t: _env(f"NF_THINK_{t.upper()}", on) in ("1", "true", "yes") for t, on in (("nano", "0"), ("super", "1"), ("ultra", "1"))
        }
    )
    # Agent memory recalls the most related earlier runs, ranked by a Nebius embedding model.
    semantic_memory: bool = field(default_factory=lambda: _env("NF_SEMANTIC_MEMORY", "1") in ("1", "true", "yes"))
    embed_model: str = field(default_factory=lambda: _env("NF_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B"))
    eval_cases: int = field(default_factory=lambda: int(_env("NF_EVAL_CASES", "8")))
    tool_timeout: int = field(default_factory=lambda: int(_env("NF_TOOL_TIMEOUT", "20")))
    max_agent_steps: int = field(default_factory=lambda: int(_env("NF_MAX_AGENT_STEPS", "6")))
    models: Models = field(default_factory=Models)
    prices: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "nano": _price("NANO", 0.06, 0.24),
            "super": _price("SUPER", 0.30, 0.90),
            "ultra": _price("ULTRA", 1.00, 3.00),
        }
    )
    # Hard stop for one build or request, in USD. 0 disables the cap.
    budget_usd: float = field(default_factory=lambda: float(_env("NF_BUDGET_USD", "0.50")))
    # Lite mode sends Ultra's jobs to Super: much cheaper, for trial credits.
    lite: bool = field(default_factory=lambda: _env("NF_LITE", "0") in ("1", "true", "yes"))
    # Shown as the owner of every agent in the dashboard's agent registry.
    owner: str = field(default_factory=lambda: _env("NF_OWNER", "Praveen"))


def settings() -> Settings:
    return Settings()
