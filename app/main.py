from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.routes.messages import router
from app.storage.sqlite import SQLiteStore


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    store = SQLiteStore(resolved_settings.db_path)
    store.init_db()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="DeepSeek Anthropic Thinking Repair Proxy", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.store = store
    app.include_router(router)

    return app


app = create_app()
