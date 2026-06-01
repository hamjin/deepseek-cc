import json
import warnings

import httpx
import pytest
import respx
from httpx import ASGITransport

from app.config import Settings
from app.main import create_app


def test_openapi_operation_ids_are_unique_without_warnings(tmp_path):
    settings = Settings(db_path=tmp_path / "proxy.sqlite3")
    app = create_app(settings)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        schema = app.openapi()

    operation_ids = [
        operation["operationId"]
        for path_item in schema["paths"].values()
        for method, operation in path_item.items()
        if method in {"delete", "get", "patch", "post", "put"}
    ]
    assert len(operation_ids) == len(set(operation_ids))


@pytest.mark.asyncio
async def test_models_endpoint_passthrough_preserves_query_and_headers(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test/anthropic",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    upstream_body = {"data": [{"id": "deepseek-v4-pro", "type": "model"}]}

    with respx.mock(assert_all_called=False) as router:
        route = router.get("https://upstream.test/anthropic/v1/models").mock(
            return_value=httpx.Response(200, json=upstream_body)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/v1/models",
                params={"limit": "1000"},
                headers={
                    "x-api-key": "client-secret",
                    "anthropic-version": "2023-06-01",
                    "accept-encoding": "gzip",
                },
            )

    assert response.status_code == 200
    assert response.json() == upstream_body
    assert str(route.calls[0].request.url) == "https://upstream.test/anthropic/v1/models?limit=1000"
    assert route.calls[0].request.headers["x-api-key"] == "client-secret"
    assert route.calls[0].request.headers["anthropic-version"] == "2023-06-01"
    assert route.calls[0].request.headers["accept-encoding"] == "identity"
    assert route.calls[0].request.headers["host"] == "upstream.test"


@pytest.mark.asyncio
async def test_count_tokens_endpoint_passthrough_preserves_query_body_and_headers(tmp_path):
    settings = Settings(
        upstream_base_url="https://upstream.test/anthropic",
        db_path=tmp_path / "proxy.sqlite3",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)

    request_body = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hello"}],
    }
    upstream_body = {"input_tokens": 7}

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://upstream.test/anthropic/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json=upstream_body)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages/count_tokens",
                params={"beta": "true"},
                headers={
                    "x-api-key": "client-secret",
                    "anthropic-version": "2023-06-01",
                    "accept-encoding": "br",
                    "content-type": "application/json",
                },
                json=request_body,
            )

    assert response.status_code == 200
    assert response.json() == upstream_body
    assert (
        str(route.calls[0].request.url)
        == "https://upstream.test/anthropic/v1/messages/count_tokens?beta=true"
    )
    assert json.loads(route.calls[0].request.content) == request_body
    assert route.calls[0].request.headers["x-api-key"] == "client-secret"
    assert route.calls[0].request.headers["anthropic-version"] == "2023-06-01"
    assert route.calls[0].request.headers["accept-encoding"] == "identity"
    assert route.calls[0].request.headers["host"] == "upstream.test"
