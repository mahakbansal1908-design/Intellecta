"""Bridge to llm_gatewayV7.

V7 is V3 plus a single new endpoint, `POST /v1/embed`. The session-version
mapping (V7 for Session 7) lets us evolve the gateway forward without
touching prior versions. V3 remains available for Session 6 agents.

Auto-starts the gateway on port 8107 if it is not already up, then
re-exports the V7 `LLM` client and a module-level `embed()` helper. Every
layer in this agent imports from here so the boot logic lives in one place.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

GATEWAY_V7_DIR = Path(__file__).resolve().parents[1] / "llm_gatewayV7"
GATEWAY_URL = "http://localhost:8107"


def _is_up() -> bool:
    try:
        httpx.get(f"{GATEWAY_URL}/v1/routers", timeout=2.0)
        return True
    except Exception:
        return False


def ensure_gateway() -> None:
    """Start V7 if it is not already running. Idempotent."""
    if _is_up():
        return
    if not GATEWAY_V7_DIR.exists():
        raise RuntimeError(
            f"Gateway V7 directory not found at {GATEWAY_V7_DIR}. "
            "Build llm_gatewayV7 (Session 7 prerequisite) before running S7 code."
        )
    print(f"[gateway] launching llm_gatewayV7 from {GATEWAY_V7_DIR}")
    log_dir = Path(__file__).parent / "state"
    log_dir.mkdir(exist_ok=True)
    log_file = open(log_dir / "gateway.log", "w")
    subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(GATEWAY_V7_DIR),
        stdout=log_file,
        stderr=log_file,
    )
    for _ in range(45):
        time.sleep(1)
        if _is_up():
            print(f"[gateway] up on {GATEWAY_URL}")
            log_file.close()
            return
    log_file.close()
    raise RuntimeError(f"Gateway V7 failed to start within 45s. Check {GATEWAY_V7_DIR}")


# Load V7's client.py without polluting sys.path. The gateway dir has its
# own `schemas.py`, which would shadow ours if we put it on the path.
import importlib.util as _importlib_util

_client_path = GATEWAY_V7_DIR / "client.py"
if _client_path.exists():
    _spec = _importlib_util.spec_from_file_location("llm_gatewayV7_client", _client_path)
    _mod = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    LLM = _mod.LLM
else:
    LLM = None  # populated once V7 is built; importers should ensure_gateway() first


def embed(text: str, task_type: str = "retrieval_document") -> dict:
    """Compute an embedding for `text` via the gateway's V7 embed endpoint.

    Returns the full response dict: `{embedding, dim, model, provider,
    latency_ms, ...}`. The chosen embedding model is fixed at the gateway
    level. Changing it invalidates every FAISS index built against the old
    vectors, so callers should treat the model as a project-level constant.
    """
    ensure_gateway()
    if LLM is None:
        raise RuntimeError(
            "Gateway V7 client unavailable. Confirm llm_gatewayV7/client.py exists."
        )
    return LLM().embed(text, task_type=task_type)


__all__ = ["ensure_gateway", "LLM", "GATEWAY_URL", "GATEWAY_V7_DIR", "embed"]
