import json

from app.services.request_logging import (
    RequestLogRecorder,
    build_log_payload,
    normalize_json_like_strings,
    truncate_for_console,
)


def test_truncate_for_console_limits_to_twenty_characters():
    assert truncate_for_console("abcdefghijklmnopqrstuvwxyz") == "abcdefghijklmnopqrst..."
    assert truncate_for_console("short") == "short"


def test_build_log_payload_matches_example_shape_for_non_stream_response():
    client = {"messages": [{"role": "user", "content": "hello"}]}
    response = {"id": "msg_1", "content": [{"type": "text", "text": "ok"}]}

    payload = build_log_payload(client, response, stream_events=[])

    assert payload == {
        "client": client,
        "response": {
            "sse": {},
            "full": response,
        },
    }


def test_request_log_recorder_writes_turn_files(tmp_path):
    recorder = RequestLogRecorder(tmp_path / "log")
    payload = {
        "client": {"messages": [{"role": "user", "content": "hello"}]},
        "response": {"sse": {}, "full": {"content": []}},
    }

    path = recorder.write("metadata.user_id", "session/id with bad chars", payload)

    assert path == tmp_path / "log" / "session_id_with_bad_chars" / "1.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload

    second = recorder.write("metadata.user_id", "session/id with bad chars", payload)
    assert second == tmp_path / "log" / "session_id_with_bad_chars" / "2.json"


def test_normalize_json_like_strings_converts_objects_and_arrays_only():
    value = {
        "metadata": {
            "user_id": '{"device_id":"abc","account_uuid":"","session_id":"s1"}',
            "plain": "not json",
            "number_string": "123",
            "json_string": '"abc"',
            "array": '[{"id":1}]',
        },
        "messages": [{"content": "normal text"}],
    }

    normalized = normalize_json_like_strings(value)

    assert normalized["metadata"]["user_id"] == {
        "device_id": "abc",
        "account_uuid": "",
        "session_id": "s1",
    }
    assert normalized["metadata"]["array"] == [{"id": 1}]
    assert normalized["metadata"]["plain"] == "not json"
    assert normalized["metadata"]["number_string"] == "123"
    assert normalized["metadata"]["json_string"] == '"abc"'
    assert normalized["messages"][0]["content"] == "normal text"
