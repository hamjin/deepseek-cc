import json

from app.services.transcript import SSEAccumulator


def sse_frame(event: str, payload: dict, newline: str = "\n") -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"event: {event}{newline}data: {data}{newline}{newline}"


def event_payload(delta_type: str, value_key: str, value: str) -> dict:
    return {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": delta_type, value_key: value},
    }


def test_sse_accumulator_reconstructs_thinking_text_and_tool_use():
    accumulator = SSEAccumulator()
    events = [
        sse_frame(
            "message_start",
            {"type": "message_start", "message": {"id": "msg_1", "content": []}},
        ),
        sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        sse_frame(
            "content_block_delta",
            event_payload("thinking_delta", "thinking", "Need"),
        ),
        sse_frame(
            "content_block_delta",
            event_payload("thinking_delta", "thinking", " tool"),
        ),
        sse_frame(
            "content_block_delta",
            event_payload("signature_delta", "signature", "sig"),
        ),
        sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse_frame(
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
        ),
        sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
            },
        ),
        sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '"abc"}'},
            },
        ),
        sse_frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
        sse_frame("message_stop", {"type": "message_stop"}),
    ]

    forwarded = b"".join(accumulator.feed(chunk.encode("utf-8")) for chunk in events)

    assert forwarded == "".join(events).encode("utf-8")
    assert accumulator.complete is True
    assert accumulator.message_id == "msg_1"
    assert accumulator.content == [
        {"type": "thinking", "thinking": "Need tool", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"query": "abc"}},
    ]


def test_sse_accumulator_incomplete_stream_not_complete():
    accumulator = SSEAccumulator()

    accumulator.feed(
        b'event: content_block_start\n'
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"tool_use","id":"toolu_1",'
        b'"name":"lookup","input":{}}}\n\n'
    )

    assert accumulator.complete is False
    assert accumulator.content == []


def test_sse_accumulator_handles_fragmented_crlf_and_multiline_data():
    accumulator = SSEAccumulator()
    frame = (
        b'event: message_start\r\n'
        b'data: {"type":"message_start",\r\n'
        b'data: "message":{"id":"msg_split","role":"assistant","content":[]}}\r\n\r\n'
        b'event: message_stop\r\n'
        b'data: {"type":"message_stop"}\r\n\r\n'
    )

    forwarded = b"".join(
        accumulator.feed(frame[index : index + 1]) for index in range(len(frame))
    )

    assert forwarded == frame
    assert accumulator.message_id == "msg_split"
    assert accumulator.complete is True
