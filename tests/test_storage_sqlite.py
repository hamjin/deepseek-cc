from app.storage.sqlite import SQLiteStore


def test_stores_content_and_indexes_tool_uses(tmp_path):
    db_path = tmp_path / "proxy.sqlite3"
    store = SQLiteStore(db_path)
    store.init_db()
    content = [
        {"type": "thinking", "thinking": "keep", "signature": "sig", "unknown": 1},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"x": 1}},
    ]

    row_id = store.save_assistant_response("conv-1", "msg-1", content, "req-hash")

    stored = store.find_by_tool_use_ids("conv-1", ["toolu_1"])
    assert len(stored) == 1
    assert stored[0].id == row_id
    assert stored[0].content == content
    assert store.find_by_tool_use_ids("conv-2", ["toolu_1"])[0].id == row_id


def test_stores_content_and_indexes_server_tool_uses(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    content = [
        {"type": "thinking", "thinking": "keep", "signature": "sig"},
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "x"},
        },
    ]

    row_id = store.save_assistant_response("conv-1", "msg-1", content, "req-hash")

    stored = store.find_by_tool_use_ids("changed-conv", ["srvtoolu_1"])
    assert len(stored) == 1
    assert stored[0].id == row_id
    assert stored[0].content == content


def test_tool_use_id_lookup_is_primary_even_without_matching_conversation(tmp_path):
    store = SQLiteStore(tmp_path / "proxy.sqlite3")
    store.init_db()
    content = [{"type": "tool_use", "id": "toolu_global", "name": "lookup", "input": {}}]
    store.save_assistant_response("fallback-before", "msg-1", content, "req-hash")

    stored = store.find_by_tool_use_ids("fallback-after", ["toolu_global"])

    assert len(stored) == 1
    assert stored[0].conversation_key == "fallback-before"
    assert stored[0].content == content


def test_persists_across_store_instances(tmp_path):
    db_path = tmp_path / "proxy.sqlite3"
    store = SQLiteStore(db_path)
    store.init_db()
    content = [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}]
    store.save_assistant_response("conv-1", "msg-1", content, "req-hash")
    store.close()

    reopened = SQLiteStore(db_path)
    reopened.init_db()

    assert reopened.find_by_tool_use_ids("conv-1", ["toolu_1"])[0].content == content
