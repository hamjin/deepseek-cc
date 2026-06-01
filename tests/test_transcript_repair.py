from app.services.transcript import TranscriptRepairer, UnknownToolUseError
from app.storage.sqlite import SQLiteStore

ASSISTANT_CONTENT = [
    {
        "type": "thinking",
        "thinking": "Need a tool.",
        "signature": "sig-1",
        "extra": {"keep": True},
    },
    {"type": "text", "text": "I will call a tool."},
    {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "lookup",
        "input": {"query": "x"},
    },
]


def test_injects_stored_assistant_before_missing_tool_result(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    store.save_assistant_response("conv-1", "msg-1", ASSISTANT_CONTENT, "req")
    repairer = TranscriptRepairer(store, strict=False)

    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
    ]

    repaired = repairer.repair_messages("conv-1", messages)

    assert repaired == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": ASSISTANT_CONTENT},
        messages[1],
    ]


def test_injects_by_tool_use_id_when_conversation_key_changed(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    store.save_assistant_response("old-fallback-key", "msg-1", ASSISTANT_CONTENT, "req")
    repairer = TranscriptRepairer(store, strict=False)

    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        }
    ]

    repaired = repairer.repair_messages("new-fallback-key", messages)

    assert repaired == [{"role": "assistant", "content": ASSISTANT_CONTENT}, messages[0]]


def test_replaces_previous_assistant_missing_thinking(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    store.save_assistant_response("conv-1", "msg-1", ASSISTANT_CONTENT, "req")
    repairer = TranscriptRepairer(store, strict=False)

    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
    ]

    repaired = repairer.repair_messages("conv-1", messages)

    assert repaired[1] == {"role": "assistant", "content": ASSISTANT_CONTENT}
    assert len(repaired) == 3


def test_repairs_server_tool_use_content(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    server_content = [
        {"type": "thinking", "thinking": "Need search.", "signature": "sig-1"},
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "x"},
        },
    ]
    store.save_assistant_response("conv-1", "msg-1", server_content, "req")
    repairer = TranscriptRepairer(store, strict=False)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "srvtoolu_1", "content": "ok"}
            ],
        }
    ]

    repaired = repairer.repair_messages("changed-conv", messages)

    assert repaired == [{"role": "assistant", "content": server_content}, messages[0]]


def test_does_not_duplicate_complete_assistant(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    store.save_assistant_response("conv-1", "msg-1", ASSISTANT_CONTENT, "req")
    repairer = TranscriptRepairer(store, strict=False)

    messages = [
        {"role": "assistant", "content": ASSISTANT_CONTENT},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
        },
    ]

    assert repairer.repair_messages("conv-1", messages) == messages


def test_multiple_tool_results_from_same_assistant_inject_once(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    content = ASSISTANT_CONTENT + [
        {"type": "tool_use", "id": "toolu_2", "name": "lookup", "input": {"query": "y"}}
    ]
    store.save_assistant_response("conv-1", "msg-1", content, "req")
    repairer = TranscriptRepairer(store, strict=False)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "b"},
            ],
        }
    ]

    repaired = repairer.repair_messages("conv-1", messages)

    assert repaired == [{"role": "assistant", "content": content}, messages[0]]


def test_removes_unanswered_parallel_agent_tool_uses_from_logged_failure(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    repairer = TranscriptRepairer(store, strict=False)

    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Need several lookups.", "signature": "sig-1"},
                {
                    "type": "tool_use",
                    "id": "call_00_JeUOW9yOd9D2JpMHspzb0662",
                    "name": "Agent",
                    "input": {},
                },
                {
                    "type": "tool_use",
                    "id": "call_01_0VOnPiGTl0MH1K2MwpoL1466",
                    "name": "Agent",
                    "input": {},
                },
                {
                    "type": "tool_use",
                    "id": "call_02_l75LRu7RCm6dBWUa4QKp1786",
                    "name": "Agent",
                    "input": {},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_02_l75LRu7RCm6dBWUa4QKp1786",
                    "content": "ok",
                }
            ],
        },
    ]

    repaired = repairer.repair_messages("conv-1", messages)

    assert repaired == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Need several lookups.", "signature": "sig-1"},
                {
                    "type": "tool_use",
                    "id": "call_02_l75LRu7RCm6dBWUa4QKp1786",
                    "name": "Agent",
                    "input": {},
                },
            ],
        },
        messages[1],
    ]


def test_unknown_tool_use_forwards_or_raises_by_mode(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "missing", "content": "ok"}],
        }
    ]

    assert TranscriptRepairer(store, strict=False).repair_messages("conv", messages) == messages

    try:
        TranscriptRepairer(store, strict=True).repair_messages("conv", messages)
    except UnknownToolUseError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("strict mode should reject unknown tool ids")
