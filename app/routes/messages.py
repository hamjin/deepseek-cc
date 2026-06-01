from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.config import Settings
from app.services.proxy import UpstreamProxy
from app.services.request_logging import (
    RequestLogRecorder,
    StreamLogCollector,
    build_log_payload,
    json_preview,
    response_body_to_json,
    response_preview,
)
from app.services.transcript import (
    ConversationKey,
    TranscriptRepairer,
    UnknownToolUseError,
    request_hash,
    resolve_conversation_key,
    resolve_storage_conversation_key,
    should_store_content,
)
from app.storage.sqlite import SQLiteStore

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


@router.post("/v1/messages")
async def messages(request: Request) -> Response:
    settings: Settings = request.app.state.settings
    store: SQLiteStore = request.app.state.store
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "body must be object"}})

    conversation = resolve_conversation_key(dict(request.headers), body)
    repaired_body = strip_output_config_effort_when_thinking_disabled(body)
    if settings.enable_thinking_repair:
        try:
            repaired_body = TranscriptRepairer(store, settings.repair_strict).repair_body(
                conversation.value, repaired_body
            )
        except UnknownToolUseError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "unknown_tool_use_id", "message": str(exc)}},
                headers={"x-conversation-key-source": conversation.source},
            )

    storage_conversation = resolve_storage_conversation_key(store, conversation, repaired_body)

    proxy = UpstreamProxy(settings, store)
    incoming_headers = dict(request.headers)
    log_recorder = RequestLogRecorder(settings.log_dir)
    log_request_preview(storage_conversation, repaired_body)

    if bool(repaired_body.get("stream")):
        return await stream_response(
            proxy,
            repaired_body,
            incoming_headers,
            storage_conversation,
            settings,
            log_recorder,
        )
    return await non_stream_response(
        proxy,
        repaired_body,
        incoming_headers,
        storage_conversation,
        settings,
        log_recorder,
    )


def strip_output_config_effort_when_thinking_disabled(body: dict[str, Any]) -> dict[str, Any]:
    thinking = body.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "disabled":
        return body

    output_config = body.get("output_config")
    if not isinstance(output_config, dict) or "effort" not in output_config:
        return body

    normalized = dict(body)
    normalized_output_config = dict(output_config)
    normalized_output_config.pop("effort", None)
    if normalized_output_config:
        normalized["output_config"] = normalized_output_config
    else:
        normalized.pop("output_config", None)
    return normalized


@router.get("/v1/{upstream_path:path}")
@router.post("/v1/{upstream_path:path}")
@router.put("/v1/{upstream_path:path}")
@router.patch("/v1/{upstream_path:path}")
@router.delete("/v1/{upstream_path:path}")
async def passthrough(upstream_path: str, request: Request) -> Response:
    settings: Settings = request.app.state.settings
    store: SQLiteStore = request.app.state.store
    proxy = UpstreamProxy(settings, store)
    upstream = await proxy.request(
        request.method,
        f"/v1/{upstream_path}",
        request.scope.get("query_string", b""),
        await request.body(),
        dict(request.headers),
    )

    response_drop_headers = {
        "connection",
        "content-encoding",
        "content-length",
        "content-type",
        "transfer-encoding",
    }
    headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in response_drop_headers
    }
    media_type = upstream.headers.get("content-type", None)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=media_type,
        headers=headers,
    )


async def non_stream_response(
    proxy: UpstreamProxy,
    body: dict[str, Any],
    headers: dict[str, str],
    conversation: ConversationKey,
    settings: Settings,
    log_recorder: RequestLogRecorder,
) -> Response:
    upstream = await proxy.post_messages(body, headers)
    response_headers = {"x-conversation-key-source": conversation.source}
    media_type = upstream.headers.get("content-type", "application/json")
    content = upstream.content
    log_response_preview(conversation, content)

    if 200 <= upstream.status_code < 300:
        conversation = maybe_store_json_response(proxy.store, conversation, body, upstream)

    if settings.log_full_request:
        log_recorder.write(
            conversation.source,
            conversation.value,
            build_log_payload(body, response_body_to_json(content), stream_events=[]),
        )

    return Response(
        content=content,
        status_code=upstream.status_code,
        media_type=media_type,
        headers=response_headers,
    )


async def stream_response(
    proxy: UpstreamProxy,
    body: dict[str, Any],
    headers: dict[str, str],
    conversation: ConversationKey,
    settings: Settings,
    log_recorder: RequestLogRecorder,
) -> Response:
    upstream, iterator = await proxy.stream_messages(body, headers, conversation)
    response_headers = {"x-conversation-key-source": conversation.source}
    media_type = upstream.headers.get("content-type", "text/event-stream")
    if upstream.status_code >= 400:
        content = b""
        async for chunk in iterator:
            content += chunk
        log_response_preview(conversation, content)
        if settings.log_full_request:
            log_conversation = resolve_storage_conversation_key(
                proxy.store, conversation, body
            )
            log_recorder.write(
                log_conversation.source,
                log_conversation.value,
                build_log_payload(body, response_body_to_json(content), stream_events=[]),
            )
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=media_type,
            headers=response_headers,
        )
    logged_iterator = logging_stream_iterator(
        iterator,
        body,
        conversation,
        settings,
        log_recorder,
        proxy.store,
    )
    return StreamingResponse(
        logged_iterator,
        status_code=upstream.status_code,
        media_type="text/event-stream",
        headers=response_headers,
    )


async def logging_stream_iterator(
    iterator: AsyncIterator[bytes],
    body: dict[str, Any],
    conversation: ConversationKey,
    settings: Settings,
    log_recorder: RequestLogRecorder,
    store: SQLiteStore,
) -> AsyncIterator[bytes]:
    collector = StreamLogCollector()
    try:
        async for chunk in iterator:
            yield collector.feed(chunk)
    finally:
        logger.info(
            "response_preview conversation_key_source=%s body=%s",
            conversation.source,
            collector.preview_text(),
        )
        if settings.log_full_request:
            log_conversation = resolve_storage_conversation_key(
                store,
                conversation,
                body,
                collector.accumulator.content if collector.accumulator.complete else None,
            )
            log_recorder.write(
                log_conversation.source,
                log_conversation.value,
                build_log_payload(
                    body,
                    collector.full_response(),
                    stream_events=collector.events,
                ),
            )


def log_request_preview(conversation: ConversationKey, body: dict[str, Any]) -> None:
    model = body.get("model", "unknown")
    logger.info(
        "request_preview conversation_key_source=%s model=%s body=%s",
        conversation.source,
        model,
        json_preview(body),
    )


def log_response_preview(conversation: ConversationKey, content: bytes) -> None:
    parsed = response_body_to_json(content)
    model = "unknown"
    text_snippet = ""

    if isinstance(parsed, dict) and "raw" not in parsed:
        model = parsed.get("model", model)
        choices = parsed.get("choices", [])
        if choices:
            msg = choices[0].get("message") or choices[0].get("delta") or {}
            raw = msg.get("content")
            if isinstance(raw, str):
                text_snippet = raw[:80]
            elif isinstance(raw, list):
                text_snippet = str(raw[0])[:80] if raw else "[empty content]"
        if not text_snippet and "error" in parsed:
            text_snippet = str(parsed["error"])[:80]

    logger.info(
        "response_preview conversation_key_source=%s model=%s content=%s",
        conversation.source,
        model,
        text_snippet or response_preview(content, limit=80),
    )


def maybe_store_json_response(
    store: SQLiteStore, conversation: ConversationKey, body: dict[str, Any], upstream: Any
) -> ConversationKey:
    try:
        payload = upstream.json()
    except ValueError:
        return conversation
    if not isinstance(payload, dict):
        return conversation
    content = payload.get("content")
    if not isinstance(content, list):
        return conversation
    storage_conversation = resolve_storage_conversation_key(
        store, conversation, body, content
    )
    if should_store_content(content, store_all=True):
        store.save_assistant_response(
            storage_conversation.value,
            payload.get("id") if isinstance(payload.get("id"), str) else None,
            content,
            request_hash(body),
        )
    return storage_conversation
