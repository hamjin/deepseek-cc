from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from app.services.transcript import SSEAccumulator, find_sse_delimiter
from app.storage.sqlite import dumps_json

JsonDict = dict[str, Any]
DEFAULT_PREVIEW_CHARS = 20
BINARY_THRESHOLD = 0.3


def truncate_for_console(value: str, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def json_preview(value: Any, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    return truncate_for_console(dumps_json(value), limit)


def response_body_to_json(content: bytes) -> Any:
    text = content.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _is_likely_binary(text: str) -> bool:
    """U+FFFD replacement chars > 30% of text → binary data."""
    if not text:
        return False
    return text.count("�") / len(text) > BINARY_THRESHOLD


def response_preview(content: bytes, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    text = content.decode("utf-8", errors="replace")
    if _is_likely_binary(text):
        return f"<binary {len(content)} bytes, hex: {content[:24].hex()}>"
    return truncate_for_console(text, limit)


def normalize_json_like_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_json_like_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json_like_strings(item) for item in value]
    if isinstance(value, str):
        return parse_json_container_string(value)
    return value


def parse_json_container_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, dict | list):
        return normalize_json_like_strings(parsed)
    return value


def build_log_payload(
    client: JsonDict,
    full_response: Any,
    stream_events: list[JsonDict],
) -> JsonDict:
    return {
        "client": client,
        "response": {
            "sse": {f"chunk_{index}": event for index, event in enumerate(stream_events, start=1)},
            "full": full_response,
        },
    }


def build_stream_full_response(
    accumulator: SSEAccumulator,
    events: list[JsonDict] | None = None,
) -> JsonDict:
    message = first_stream_message(events or []) or {
        "id": accumulator.message_id,
        "type": "message",
        "role": "assistant",
    }
    message["content"] = accumulator.content
    apply_message_deltas(message, events or [])
    return message


def first_stream_message(events: list[JsonDict]) -> JsonDict | None:
    for event in events:
        if event.get("type") == "message_start" and isinstance(event.get("message"), dict):
            return copy.deepcopy(event["message"])
    return None


def apply_message_deltas(message: JsonDict, events: list[JsonDict]) -> None:
    for event in events:
        if event.get("type") != "message_delta":
            continue
        delta = event.get("delta")
        if isinstance(delta, dict):
            message.update(delta)
        usage = event.get("usage")
        if isinstance(usage, dict):
            existing = message.get("usage")
            if isinstance(existing, dict):
                existing.update(usage)
            else:
                message["usage"] = copy.deepcopy(usage)


def safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "unknown"


class RequestLogRecorder:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)

    def write(self, _source: str, conversation_id: str, payload: JsonDict) -> Path:
        directory = self.log_dir / safe_path_part(conversation_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._next_turn(directory)}.json"
        normalized_payload = normalize_json_like_strings(payload)
        path.write_text(
            json.dumps(normalized_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _next_turn(directory: Path) -> int:
        turns: list[int] = []
        for path in directory.glob("*.json"):
            if path.stem.isdigit():
                turns.append(int(path.stem))
        return max(turns, default=0) + 1


class StreamLogCollector:
    def __init__(self) -> None:
        self._buffer = b""
        self.accumulator = SSEAccumulator()
        self.events: list[JsonDict] = []
        self.preview = ""

    def feed(self, chunk: bytes) -> bytes:
        self.accumulator.feed(chunk)
        self._append_preview(chunk)
        self._buffer += chunk
        while True:
            delimiter = find_sse_delimiter(self._buffer)
            if delimiter is None:
                break
            start, end = delimiter
            raw_event, self._buffer = self._buffer[:start], self._buffer[end:]
            event = parse_sse_json_event(raw_event)
            if event is not None:
                self.events.append(event)
        return chunk

    def full_response(self) -> JsonDict:
        if self.accumulator.complete:
            return build_stream_full_response(self.accumulator, self.events)
        return {"incomplete": True, "content": self.accumulator.content}

    def _append_preview(self, chunk: bytes) -> None:
        if len(self.preview) > DEFAULT_PREVIEW_CHARS:
            return
        self.preview += chunk.decode("utf-8", errors="replace")

    def preview_text(self, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
        return truncate_for_console(self.preview, limit)


def parse_sse_json_event(raw_event: bytes) -> JsonDict | None:
    data_lines: list[str] = []
    for raw_line in raw_event.splitlines():
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
