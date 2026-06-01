import json
import logging
from pathlib import Path

import httpx
import pytest
import respx
from httpx import ASGITransport

from app.config import Settings
from app.main import create_app
from app.storage.sqlite import SQLiteStore


def sse_frame(event: str, payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


@pytest.mark.asyncio
async def test_removes_output_config_effort_when_thinking_is_disabled(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={"id": "msg_1", "role": "assistant", "content": []},
            )
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "output_config": {"effort": "high"},
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    sent = json.loads(route.calls[0].request.content)
    assert "output_config" not in sent
    assert sent["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_keeps_disabled_thinking_without_output_config_effort(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={"id": "msg_1", "role": "assistant", "content": []},
            )
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    sent = json.loads(route.calls[0].request.content)
    assert "output_config" not in sent
    assert sent["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_keeps_output_config_effort_when_thinking_is_not_disabled(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    with respx.mock(assert_all_called=True) as router:
        route = router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={"id": "msg_1", "role": "assistant", "content": []},
            )
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "output_config": {"effort": "high"},
                    "thinking": {"type": "enabled"},
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert response.status_code == 200
    sent = json.loads(route.calls[0].request.content)
    assert sent["output_config"] == {"effort": "high"}
    assert sent["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_non_stream_passthrough_stores_and_repairs_followup(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
        log_full_request=False,
        enable_thinking_repair=True,
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    assistant_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Need tool", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
        ],
    }
    final_response = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
    }

    with respx.mock(assert_all_called=True) as router:
        first = router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(200, json=assistant_response)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                headers={
                    "x-conversation-id": "conv-1",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": "client-secret",
                    "x-client-trace-id": "trace-1",
                },
                json={"model": "deepseek-v4-pro", "max_tokens": 64, "messages": []},
            )

        assert response.status_code == 200
        assert response.json() == assistant_response
        assert first.calls[0].request.headers["x-api-key"] == "client-secret"
        assert first.calls[0].request.headers["anthropic-version"] == "2023-06-01"
        assert first.calls[0].request.headers["x-client-trace-id"] == "trace-1"
        assert first.calls[0].request.headers["host"] == "upstream.test"

    with respx.mock(assert_all_called=True) as router:
        second = router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(200, json=final_response)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                headers={"x-conversation-id": "conv-1"},
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_1",
                                    "content": "ok",
                                }
                            ],
                        }
                    ],
                },
            )

        assert response.status_code == 200
        sent = json.loads(second.calls[0].request.content)
        assert sent["messages"][0] == {
            "role": "assistant",
            "content": assistant_response["content"],
        }
        assert sent["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_repair_uses_tool_id_without_conversation_header(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
        enable_thinking_repair=True,
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    assistant_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Need tool", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_no_header", "name": "lookup", "input": {}},
        ],
    }

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(200, json=assistant_response)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/v1/messages",
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "first"}],
                },
            )

    with respx.mock(assert_all_called=True) as router:
        second = router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={"id": "msg_2", "role": "assistant", "content": []},
            )
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/v1/messages",
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_no_header",
                                    "content": "ok",
                                }
                            ],
                        }
                    ],
                },
            )

    sent = json.loads(second.calls[0].request.content)
    assert sent["messages"][0] == {
        "role": "assistant",
        "content": assistant_response["content"],
    }


@pytest.mark.asyncio
async def test_followup_save_reuses_tool_id_conversation_without_header(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
        enable_thinking_repair=True,
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    first_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Need tool", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_same_dialog", "name": "lookup", "input": {}},
        ],
    }
    second_response = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Need next tool", "signature": "sig2"},
            {"type": "tool_use", "id": "toolu_next", "name": "lookup", "input": {}},
        ],
    }

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(200, json=first_response)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/v1/messages",
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "first"}],
                },
            )

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(200, json=second_response)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/v1/messages",
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_same_dialog",
                                    "content": "ok",
                                }
                            ],
                        }
                    ],
                },
            )

    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    first_stored = store.find_by_tool_use_ids("unrelated", ["toolu_same_dialog"])[0]
    second_stored = store.find_by_tool_use_ids("unrelated", ["toolu_next"])[0]
    assert second_stored.conversation_key == first_stored.conversation_key


@pytest.mark.asyncio
async def test_upstream_error_passthrough_does_not_store(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad"}})
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                headers={"x-conversation-id": "conv-1"},
                json={"model": "deepseek-v4-pro", "max_tokens": 64, "messages": []},
            )

    assert response.status_code == 400
    assert response.json() == {"error": {"message": "bad"}}
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    assert store.find_by_tool_use_ids("conv-1", ["toolu_1"]) == []


@pytest.mark.asyncio
async def test_local_dotenv_log_full_request_does_not_write_route_test_logs(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    Path(".env").write_text("LOG_FULL_REQUEST=true\n", encoding="utf-8")
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad"}})
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                headers={"x-conversation-id": "conv-dotenv-leak"},
                json={"model": "deepseek-v4-pro", "max_tokens": 64, "messages": []},
            )

    assert response.status_code == 400
    assert not (tmp_path / "log" / "conv-dotenv-leak").exists()


@pytest.mark.asyncio
async def test_log_full_request_writes_full_exchange_file_when_enabled(tmp_path, caplog):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
        log_full_request=True,
        log_dir=tmp_path / "logs",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={"id": "msg_1", "role": "assistant", "content": []},
            )
        )
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/v1/messages",
                    headers={"x-conversation-id": "conv-log"},
                    json={
                        "model": "deepseek-v4-pro",
                        "max_tokens": 64,
                        "metadata": {
                            "user_id": '{"device_id":"abc","session_id":"turn-1"}',
                        },
                        "messages": [{"role": "user", "content": "log me"}],
                    },
                )

    assert "request_preview" in caplog.text
    assert "response_preview" in caplog.text
    assert "log me" not in caplog.text
    assert "secret" not in caplog.text
    log_file = tmp_path / "logs" / "conv-log" / "1.json"
    assert log_file.exists()
    payload = json.loads(log_file.read_text(encoding="utf-8"))
    assert payload["client"]["metadata"]["user_id"] == {
        "device_id": "abc",
        "session_id": "turn-1",
    }
    assert payload["client"]["messages"] == [{"role": "user", "content": "log me"}]
    assert payload["response"] == {
        "sse": {},
        "full": {"id": "msg_1", "role": "assistant", "content": []},
    }


@pytest.mark.asyncio
async def test_log_full_request_disabled_writes_only_console_preview(tmp_path, caplog):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
        log_full_request=False,
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={"id": "msg_1", "role": "assistant", "content": []},
            )
        )
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/v1/messages",
                    json={
                        "model": "deepseek-v4-pro",
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": "do not log"}],
                    },
                )

    assert "request_preview" in caplog.text
    assert "response_preview" in caplog.text
    assert "do not log" not in caplog.text
    assert not (tmp_path / "logs").exists()


@pytest.mark.asyncio
async def test_streaming_forwards_bytes_and_stores_complete_content(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    sse = (
        sse_frame(
            "message_start",
            {"type": "message_start", "message": {"id": "msg_1", "content": []}},
        )
        + sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        )
        + sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Need tool"},
            },
        )
        + sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig"},
            },
        )
        + sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0})
        + sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {},
                },
            },
        )
        + sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
        )
        + sse_frame("content_block_stop", {"type": "content_block_stop", "index": 1})
        + sse_frame("message_stop", {"type": "message_stop"})
    )

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                content=sse,
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                headers={"x-conversation-id": "conv-stream"},
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [],
                },
            )

    assert response.status_code == 200
    assert response.content == sse.encode("utf-8")
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    stored = store.find_by_tool_use_ids("conv-stream", ["toolu_1"])
    assert stored[0].message_id == "msg_1"
    assert stored[0].content == [
        {"type": "thinking", "thinking": "Need tool", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
    ]


@pytest.mark.asyncio
async def test_streaming_full_log_contains_sse_chunks_and_full_response(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
        log_full_request=True,
        log_dir=tmp_path / "logs",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    sse = (
        sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": "deepseek-v4-pro",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        )
        + sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        + sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
        )
        + sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0})
        + sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        )
        + sse_frame("message_stop", {"type": "message_stop"})
    )

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                content=sse,
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                headers={"x-conversation-id": "conv-stream-log"},
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "stream log"}],
                },
            )

    assert response.content == sse.encode("utf-8")
    payload = json.loads(
        (tmp_path / "logs" / "conv-stream-log" / "1.json").read_text(encoding="utf-8")
    )
    assert payload["client"]["stream"] is True
    assert payload["response"]["sse"]["chunk_1"]["type"] == "message_start"
    assert payload["response"]["full"] == {
        "id": "msg_stream",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-v4-pro",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.mark.asyncio
async def test_streaming_error_passthrough_does_not_store(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    with respx.mock(assert_all_called=True) as router:
        router.post("https://upstream.test/v1/messages").mock(
            return_value=httpx.Response(401, json={"error": {"message": "no"}})
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                headers={"x-conversation-id": "conv-stream"},
                json={
                    "model": "deepseek-v4-pro",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [],
                },
            )

    assert response.status_code == 401
    assert response.json() == {"error": {"message": "no"}}
