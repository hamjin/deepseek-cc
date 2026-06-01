from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonBlock = dict[str, Any]


@dataclass(frozen=True)
class StoredAssistant:
    id: int
    conversation_key: str
    message_id: str | None
    content: list[JsonBlock]
    content_hash: str


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._connection is None:
            if self.path.parent != Path(""):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=5000")
        return self._connection

    def init_db(self) -> None:
        conn = self.connect()
        conn.executescript(
            """
            create table if not exists assistant_responses (
              id integer primary key autoincrement,
              conversation_key text not null,
              message_id text,
              created_at text not null default current_timestamp,
              request_hash text,
              content_json text not null,
              content_hash text not null
            );

            create index if not exists idx_assistant_responses_conversation
            on assistant_responses(conversation_key, id);

            create table if not exists tool_uses (
              conversation_key text not null,
              tool_use_id text not null,
              assistant_response_id integer not null,
              created_at text not null default current_timestamp,
              primary key (conversation_key, tool_use_id),
              foreign key (assistant_response_id) references assistant_responses(id)
            );
            """
        )
        conn.commit()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def save_assistant_response(
        self,
        conversation_key: str,
        message_id: str | None,
        content_blocks: list[JsonBlock],
        request_hash: str | None,
    ) -> int:
        content_json = dumps_json(content_blocks)
        content_hash = sha256_text(content_json)
        conn = self.connect()
        with conn:
            cursor = conn.execute(
                """
                insert into assistant_responses
                  (conversation_key, message_id, request_hash, content_json, content_hash)
                values (?, ?, ?, ?, ?)
                """,
                (conversation_key, message_id, request_hash, content_json, content_hash),
            )
            row_id = int(cursor.lastrowid)
            for tool_use_id in extract_tool_use_ids(content_blocks):
                conn.execute(
                    """
                    insert or replace into tool_uses
                      (conversation_key, tool_use_id, assistant_response_id)
                    values (?, ?, ?)
                    """,
                    (conversation_key, tool_use_id, row_id),
                )
        return row_id

    def find_by_tool_use_ids(
        self, conversation_key: str, tool_use_ids: list[str]
    ) -> list[StoredAssistant]:
        if not tool_use_ids:
            return []
        placeholders = ",".join("?" for _ in tool_use_ids)
        rows = self.connect().execute(
            f"""
            select distinct ar.id, ar.conversation_key, ar.message_id, ar.content_json,
                            ar.content_hash
            from tool_uses tu
            join assistant_responses ar on ar.id = tu.assistant_response_id
            where tu.tool_use_id in ({placeholders})
            order by
              case when tu.conversation_key = ? then 0 else 1 end,
              ar.id
            """,
            [*tool_use_ids, conversation_key],
        )
        return [stored_from_row(row) for row in rows.fetchall()]

    def latest_for_conversation(self, conversation_key: str) -> StoredAssistant | None:
        row = self.connect().execute(
            """
            select id, conversation_key, message_id, content_json, content_hash
            from assistant_responses
            where conversation_key = ?
            order by id desc
            limit 1
            """,
            (conversation_key,),
        ).fetchone()
        return stored_from_row(row) if row else None


def stored_from_row(row: sqlite3.Row) -> StoredAssistant:
    return StoredAssistant(
        id=int(row["id"]),
        conversation_key=str(row["conversation_key"]),
        message_id=row["message_id"],
        content=json.loads(row["content_json"]),
        content_hash=str(row["content_hash"]),
    )


def extract_tool_use_ids(content_blocks: list[JsonBlock]) -> list[str]:
    ids: list[str] = []
    for block in content_blocks:
        if block.get("type") in {"tool_use", "server_tool_use"} and isinstance(
            block.get("id"), str
        ):
            ids.append(block["id"])
    return ids


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
