from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.services.transcript import (
    ConversationKey,
    SSEAccumulator,
    request_hash,
    resolve_storage_conversation_key,
    should_store_content,
)
from app.storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

DROP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
}


def build_upstream_headers(incoming_headers: dict[str, str], settings: Settings) -> dict[str, str]:
    headers = {
        key: value
        for key, value in incoming_headers.items()
        if key.lower() not in DROP_HEADERS
    }
    headers["host"] = upstream_host_header(settings.messages_url)
    headers["accept-encoding"] = "identity"
    return headers


def upstream_host_header(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.hostname:
        raise ValueError(f"upstream URL must include a host: {url!r}")

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    default_port = {"http": 80, "https": 443}.get(parsed.scheme)
    if parsed.port is not None and parsed.port != default_port:
        return f"{host}:{parsed.port}"

    return host


class UpstreamProxy:
    def __init__(self, settings: Settings, store: SQLiteStore):
        self.settings = settings
        self.store = store

    async def post_messages(
        self, body: dict[str, Any], incoming_headers: dict[str, str]
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            return await client.post(
                self.settings.messages_url,
                json=body,
                headers=build_upstream_headers(incoming_headers, self.settings),
            )

    async def request(
        self,
        method: str,
        path: str,
        query_string: bytes,
        body: bytes,
        incoming_headers: dict[str, str],
    ) -> httpx.Response:
        url = self.settings.upstream_url(path)
        if query_string:
            url = f"{url}?{query_string.decode('ascii')}"

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            return await client.request(
                method,
                url,
                content=body,
                headers=build_upstream_headers(incoming_headers, self.settings),
            )

    async def stream_messages(
        self,
        body: dict[str, Any],
        incoming_headers: dict[str, str],
        conversation: ConversationKey,
    ) -> tuple[httpx.Response, AsyncIterator[bytes]]:
        client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        request = client.build_request(
            "POST",
            self.settings.messages_url,
            json=body,
            headers=build_upstream_headers(incoming_headers, self.settings),
        )
        response = await client.send(request, stream=True)
        if response.status_code >= 400:
            content = await response.aread()
            await response.aclose()
            await client.aclose()
            return response, single_chunk(content)

        accumulator = SSEAccumulator()
        body_hash = request_hash(body)

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield accumulator.feed(chunk)
            finally:
                await response.aclose()
                await client.aclose()
                if accumulator.complete and should_store_content(
                    accumulator.content, self.settings.store_all_assistant_content
                ):
                    storage_conversation = resolve_storage_conversation_key(
                        self.store, conversation, body, accumulator.content
                    )
                    self.store.save_assistant_response(
                        storage_conversation.value,
                        accumulator.message_id,
                        accumulator.content,
                        body_hash,
                    )
                elif not accumulator.complete:
                    logger.warning("stream ended before message_stop; not storing partial content")

        return response, iterator()


async def single_chunk(content: bytes) -> AsyncIterator[bytes]:
    yield content
