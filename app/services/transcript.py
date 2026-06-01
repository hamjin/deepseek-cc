from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.storage.sqlite import SQLiteStore, StoredAssistant, dumps_json

JsonDict = dict[str, Any]


class UnknownToolUseError(ValueError):
    pass


@dataclass(frozen=True)
class ConversationKey:
    value: str
    source: str


class TranscriptRepairer:
    def __init__(self, store: SQLiteStore, strict: bool):
        self.store = store
        self.strict = strict

    def repair_body(self, conversation_key: str, body: JsonDict) -> JsonDict:
        repaired = copy.deepcopy(body)
        messages = repaired.get("messages")
        if isinstance(messages, list):
            repaired["messages"] = self.repair_messages(conversation_key, messages)
        return repaired

    def repair_messages(self, conversation_key: str, messages: list[JsonDict]) -> list[JsonDict]:
        out: list[JsonDict] = []
        injected_response_ids: set[int] = set()

        for index, message in enumerate(messages):
            tool_result_ids = extract_tool_result_ids(message)
            if tool_result_ids:
                stored = self.store.find_by_tool_use_ids(conversation_key, tool_result_ids)
                known_tool_ids = tool_ids_for_stored(stored)
                missing = [tool_id for tool_id in tool_result_ids if tool_id not in known_tool_ids]
                if missing and self.strict:
                    raise UnknownToolUseError(f"unknown_tool_use_id: {', '.join(missing)}")

                for assistant in stored:
                    if assistant.id in injected_response_ids:
                        continue
                    if previous_assistant_matches(out, assistant):
                        if previous_assistant_needs_replace(out[-1], assistant):
                            replacement = prune_unanswered_tool_uses(
                                {
                                    "role": "assistant",
                                    "content": copy.deepcopy(assistant.content),
                                },
                                set(tool_result_ids),
                            )
                            if replacement is not None:
                                out[-1] = replacement
                            else:
                                out.pop()
                        else:
                            replacement = prune_unanswered_tool_uses(out[-1], set(tool_result_ids))
                            if replacement is not None:
                                out[-1] = replacement
                            else:
                                out.pop()
                        injected_response_ids.add(assistant.id)
                    elif not any_assistant_matches(out, assistant):
                        injected = prune_unanswered_tool_uses(
                            {
                                "role": "assistant",
                                "content": copy.deepcopy(assistant.content),
                            },
                            set(tool_result_ids),
                        )
                        if injected is not None:
                            out.append(injected)
                        injected_response_ids.add(assistant.id)

            next_tool_result_ids = next_message_tool_result_ids(messages, index)
            repaired_message = prune_unanswered_tool_uses(message, next_tool_result_ids)
            if repaired_message is not None:
                out.append(repaired_message)
        return out


def resolve_conversation_key(headers: dict[str, str], body: JsonDict) -> ConversationKey:
    lowered = {key.lower(): value for key, value in headers.items()}
    header_value = lowered.get("x-conversation-id")
    if header_value:
        return ConversationKey(header_value, "header")

    metadata = body.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("user_id"), str):
        return ConversationKey(metadata["user_id"], "metadata.user_id")

    stable = {
        "model": body.get("model"),
        "system": body.get("system"),
        "first_user": first_non_tool_user_content(body.get("messages")),
        "prefix": message_prefix_hashes(body.get("messages")),
    }
    digest = hashlib.sha256(dumps_json(stable).encode("utf-8")).hexdigest()
    return ConversationKey(digest, "fallback_hash")


def resolve_storage_conversation_key(
    store: SQLiteStore,
    fallback: ConversationKey,
    body: JsonDict,
    assistant_content: list[JsonDict] | None = None,
) -> ConversationKey:
    ids = extract_tool_result_ids_from_messages(body.get("messages"))
    if assistant_content is not None:
        ids.extend(tool_use_ids_in_content(assistant_content) - set(ids))
    stored = store.find_by_tool_use_ids(fallback.value, ids)
    if stored:
        return ConversationKey(stored[0].conversation_key, "tool_use_id")
    return fallback


def request_hash(body: JsonDict) -> str:
    return hashlib.sha256(dumps_json(body).encode("utf-8")).hexdigest()


def first_non_tool_user_content(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not contains_tool_result_content(content):
            return content
    return None


def message_prefix_hashes(messages: Any) -> list[JsonDict]:
    if not isinstance(messages, list):
        return []
    hashes = []
    for message in messages[:2]:
        if not isinstance(message, dict):
            continue
        hashes.append(
            {
                "role": message.get("role"),
                "content_hash": hashlib.sha256(
                    dumps_json(message.get("content")).encode("utf-8")
                ).hexdigest(),
            }
        )
    return hashes


def contains_tool_result_content(content: Any) -> bool:
    return bool(extract_tool_result_ids({"role": "user", "content": content}))


def extract_tool_result_ids(message: JsonDict) -> list[str]:
    if message.get("role") != "user":
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    ids: list[str] = []
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_result"
            and isinstance(block.get("tool_use_id"), str)
        ):
            ids.append(block["tool_use_id"])
    return ids


def extract_tool_result_ids_from_messages(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        for tool_id in extract_tool_result_ids(message):
            if tool_id not in seen:
                ids.append(tool_id)
                seen.add(tool_id)
    return ids


def tool_ids_for_stored(stored: list[StoredAssistant]) -> set[str]:
    ids: set[str] = set()
    for assistant in stored:
        ids.update(tool_use_ids_in_content(assistant.content))
    return ids


def previous_assistant_matches(out: list[JsonDict], assistant: StoredAssistant) -> bool:
    return (
        bool(out)
        and out[-1].get("role") == "assistant"
        and assistant_matches(out[-1], assistant)
    )


def any_assistant_matches(out: list[JsonDict], assistant: StoredAssistant) -> bool:
    return any(
        message.get("role") == "assistant" and assistant_matches(message, assistant)
        for message in out
    )


def assistant_matches(message: JsonDict, assistant: StoredAssistant) -> bool:
    message_ids = tool_use_ids_in_content(message.get("content"))
    stored_ids = tool_use_ids_in_content(assistant.content)
    return bool(message_ids & stored_ids)


def previous_assistant_needs_replace(message: JsonDict, assistant: StoredAssistant) -> bool:
    return message.get("content") != assistant.content


def next_message_tool_result_ids(messages: list[JsonDict], index: int) -> set[str]:
    if index + 1 >= len(messages):
        return set()
    next_message = messages[index + 1]
    if not isinstance(next_message, dict):
        return set()
    return set(extract_tool_result_ids(next_message))


def prune_unanswered_tool_uses(message: JsonDict, tool_result_ids: set[str]) -> JsonDict | None:
    if message.get("role") != "assistant":
        return copy.deepcopy(message)

    content = message.get("content")
    if not isinstance(content, list) or not tool_use_ids_in_content(content):
        return copy.deepcopy(message)

    pruned_content: list[JsonDict] = []
    for block in content:
        if not is_tool_use_block(block) or block["id"] in tool_result_ids:
            pruned_content.append(copy.deepcopy(block))

    if not pruned_content:
        return None

    repaired = copy.deepcopy(message)
    repaired["content"] = pruned_content
    return repaired


def is_tool_use_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") in {"tool_use", "server_tool_use"}
        and isinstance(block.get("id"), str)
    )


def tool_use_ids_in_content(content: Any) -> set[str]:
    if not isinstance(content, list):
        return set()
    ids: set[str] = set()
    for block in content:
        if is_tool_use_block(block):
            ids.add(block["id"])
    return ids


def should_store_content(content: list[JsonDict], store_all: bool) -> bool:
    return store_all or bool(tool_use_ids_in_content(content))


class SSEAccumulator:
    def __init__(self) -> None:
        self._buffer = b""
        self._blocks: dict[int, JsonDict] = {}
        self._stopped: set[int] = set()
        self._partial_json: dict[int, str] = {}
        self.message_id: str | None = None
        self.complete = False
        self.content: list[JsonDict] = []

    def feed(self, chunk: bytes) -> bytes:
        self._buffer += chunk
        while True:
            delimiter = find_sse_delimiter(self._buffer)
            if delimiter is None:
                break
            start, end = delimiter
            raw_event, self._buffer = self._buffer[:start], self._buffer[end:]
            self._handle_event(raw_event)
        return chunk

    def _handle_event(self, raw_event: bytes) -> None:
        data_lines: list[str] = []
        for raw_line in raw_event.splitlines():
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                return
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            return
        data = "\n".join(data_lines)
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return

        event_type = payload.get("type")
        if event_type == "message_start":
            message = payload.get("message")
            if isinstance(message, dict) and isinstance(message.get("id"), str):
                self.message_id = message["id"]
        elif event_type == "content_block_start":
            self._start_block(payload)
        elif event_type == "content_block_delta":
            self._apply_delta(payload)
        elif event_type == "content_block_stop":
            self._stop_block(payload)
        elif event_type == "message_stop":
            self._finalize()

    def _start_block(self, payload: JsonDict) -> None:
        index = payload.get("index")
        block = payload.get("content_block")
        if isinstance(index, int) and isinstance(block, dict):
            self._blocks[index] = copy.deepcopy(block)
            if block.get("type") in {"tool_use", "server_tool_use"}:
                self._partial_json[index] = ""

    def _apply_delta(self, payload: JsonDict) -> None:
        index = payload.get("index")
        delta = payload.get("delta")
        if not isinstance(index, int) or not isinstance(delta, dict):
            return
        block = self._blocks.setdefault(index, {})
        delta_type = delta.get("type")
        if delta_type == "text_delta" and isinstance(delta.get("text"), str):
            block["text"] = block.get("text", "") + delta["text"]
        elif delta_type == "thinking_delta" and isinstance(delta.get("thinking"), str):
            block["thinking"] = block.get("thinking", "") + delta["thinking"]
        elif delta_type == "signature_delta" and isinstance(delta.get("signature"), str):
            block["signature"] = block.get("signature", "") + delta["signature"]
        elif delta_type == "input_json_delta" and isinstance(delta.get("partial_json"), str):
            self._partial_json[index] = self._partial_json.get(index, "") + delta["partial_json"]

    def _stop_block(self, payload: JsonDict) -> None:
        index = payload.get("index")
        if not isinstance(index, int) or index not in self._blocks:
            return
        if index in self._partial_json:
            partial = self._partial_json.pop(index)
            if partial:
                try:
                    self._blocks[index]["input"] = json.loads(partial)
                except json.JSONDecodeError:
                    self._blocks[index]["_partial_json"] = partial
        self._stopped.add(index)

    def _finalize(self) -> None:
        if set(self._blocks) != self._stopped:
            return
        self.content = [self._blocks[index] for index in sorted(self._blocks)]
        self.complete = True


def find_sse_delimiter(buffer: bytes) -> tuple[int, int] | None:
    candidates = [
        (buffer.find(b"\r\n\r\n"), 4),
        (buffer.find(b"\n\n"), 2),
        (buffer.find(b"\r\r"), 2),
    ]
    found = [(index, length) for index, length in candidates if index != -1]
    if not found:
        return None
    index, length = min(found, key=lambda item: item[0])
    return index, index + length
