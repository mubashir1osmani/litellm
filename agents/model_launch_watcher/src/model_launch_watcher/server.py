"""FastAPI application serving the agent over A2A JSON-RPC.

The card is served at the well-known path so other agents can discover the skills, and
LiteLLM's own proxy can register this URL as an upstream A2A agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.utils import DEFAULT_RPC_URL
from fastapi import FastAPI

from .card import build_agent_card
from .catalog import Catalog
from .executor import ModelLaunchWatcherExecutor
from .graph import Dependencies
from .memory import Memory
from .pricing import LiteLLMTableExtractor

DEFAULT_CATALOG_PATH: Final = Path(__file__).resolve().parents[4] / "model_prices_and_context_window.json"


DEFAULT_MEMORY_PATH: Final = Path(__file__).resolve().parents[2] / "corrections.jsonl"


DEFAULT_EXTRACTION_MODEL: Final = "claude-sonnet-5"


def resolve_extraction_model() -> str:
    return os.environ.get("WATCHER_EXTRACTION_MODEL") or DEFAULT_EXTRACTION_MODEL


def resolve_memory_path() -> Path:
    override: Final = os.environ.get("WATCHER_MEMORY_PATH")
    return Path(override) if override else DEFAULT_MEMORY_PATH


def resolve_catalog_path() -> Path:
    override: Final = os.environ.get("LITELLM_COST_MAP_PATH")
    return Path(override) if override else DEFAULT_CATALOG_PATH


def build_app(deps: Dependencies, public_url: str) -> FastAPI:
    card: Final = build_agent_card(public_url)
    handler: Final = DefaultRequestHandlerV2(
        agent_executor=ModelLaunchWatcherExecutor(deps=deps),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app: Final = FastAPI(title=card.name, version=card.version, description=card.description)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, DEFAULT_RPC_URL),
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "catalog_entries": len(deps.catalog.entries), "agent": card.name}

    return app


def create_app() -> FastAPI:
    public_url: Final = os.environ.get("WATCHER_PUBLIC_URL", "http://localhost:8080")
    deps: Final = Dependencies(
        catalog=Catalog.load(resolve_catalog_path()),
        memory=Memory.load(resolve_memory_path()),
        extractor=LiteLLMTableExtractor(model=resolve_extraction_model()),
    )
    return build_app(deps, public_url)


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("WATCHER_HOST", "127.0.0.1"),
        port=int(os.environ.get("WATCHER_PORT", "8080")),
    )
