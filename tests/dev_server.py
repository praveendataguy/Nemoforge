"""Run the web app against the scripted FakeLLM, for UI work without API keys.

    python -m tests.dev_server   (never use this for real builds or demos)
"""
import time

import uvicorn

from foundry import server

from .fake_llm import FakeLLM


class SlowFake(FakeLLM):
    def __init__(self, run):
        super().__init__()
        self.run = run

    def chat(self, tier, system, user, *, max_tokens=4096):
        time.sleep(0.3)
        out = super().chat(tier, system, user, max_tokens=max_tokens)
        if self.run is not None:
            self.run.add_usage(tier, f"fake-{tier}", len(system + user) // 4, len(out) // 4)
        return out


_fakes = {}
server.llm_factory = lambda run: _fakes.setdefault(id(run), SlowFake(run))

if __name__ == "__main__":
    uvicorn.run(server.app, port=8765)
