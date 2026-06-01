from app.config import Settings
from app.services.proxy import build_upstream_headers


def test_build_upstream_headers_forwards_incoming_headers_except_hop_by_hop() -> None:
    headers = build_upstream_headers(
        {
            "authorization": "Bearer client-secret",
            "anthropic-version": "2023-06-01",
            "x-client-trace-id": "trace-1",
            "accept-language": "zh-CN",
            "accept-encoding": "gzip",
            "host": "127.0.0.1:8000",
            "content-length": "999",
            "connection": "keep-alive",
        },
        Settings(upstream_base_url="https://upstream.test/anthropic"),
    )

    assert headers == {
        "authorization": "Bearer client-secret",
        "anthropic-version": "2023-06-01",
        "x-client-trace-id": "trace-1",
        "accept-language": "zh-CN",
        "accept-encoding": "identity",
        "host": "upstream.test",
    }


def test_build_upstream_headers_forces_no_compression() -> None:
    headers = build_upstream_headers(
        {},
        Settings(upstream_base_url="https://api.deepseek.com/anthropic"),
    )

    assert headers == {
        "accept-encoding": "identity",
        "host": "api.deepseek.com",
    }
