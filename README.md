# DeepSeek Anthropic Thinking Repair Proxy

FastAPI proxy for DeepSeek's Anthropic-compatible Messages API.

It fixes multi-turn tool-call failures like:

```json
{
  "error": {
    "message": "The `content[].thinking` in the thinking mode must be passed back to the API.",
    "type": "invalid_request_error"
  }
}
```

DeepSeek thinking mode requires the assistant thinking/reasoning block from tool-call turns to be included in later requests. Some clients keep only visible text or `tool_use` blocks, then send `tool_result` without the original `thinking` block. This proxy stores upstream assistant `content[]` blocks and repairs later `tool_result` turns before forwarding them.

## What It Does

The proxy accepts Anthropic-compatible `POST /v1/messages` requests, repairs the
message history when a later `tool_result` is missing the original assistant
`thinking` block, and forwards the repaired request to DeepSeek.

Repair is keyed primarily by `tool_use.id` / `tool_result.tool_use_id`, so Claude
Code and Anthropic SDK clients do not need to send a custom conversation header.

Both regular JSON responses and `stream: true` SSE responses are supported.
Streaming bytes are forwarded unchanged while the proxy reconstructs assistant
content blocks for later repair.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Configure

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Set:

```text
UPSTREAM_BASE_URL=https://api.deepseek.com/anthropic
```

### Configuration Reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `UPSTREAM_BASE_URL` | `https://api.deepseek.com/anthropic` | Anthropic-compatible DeepSeek base URL. The proxy appends `/v1/messages`. |
| `DB_PATH` | `./deepseek_anthropic_proxy.sqlite3` | SQLite file used to store assistant content blocks and tool call indexes. |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Upstream HTTP timeout for regular and streaming requests. |
| `ENABLE_THINKING_REPAIR` | `true` | Enables automatic injection/replacement of missing assistant thinking blocks before `tool_result` turns. |
| `REPAIR_STRICT` | `false` | If true, unknown `tool_result.tool_use_id` returns local `400`; if false, request is forwarded unchanged. |
| `LOG_FULL_REQUEST` | `false` | If true, writes full request/response JSON files under `LOG_DIR`. Console logs still use short previews. |
| `LOG_DIR` | `./log` | Directory for full exchange logs when `LOG_FULL_REQUEST=true`. Files are written as `log/<conversation-id>/<turn>.json`. |
| `STORE_ALL_ASSISTANT_CONTENT` | `false` | If true, stores every assistant `content[]`; if false, stores only responses containing `tool_use`. |

Console logging always prints only short request/response previews. By default
each preview is truncated to 20 characters. `LOG_FULL_REQUEST=true` stores the
full exchange on disk instead of printing it to console. Keep it disabled unless
debugging; log files may contain user prompts, tool outputs, and thinking
blocks.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or use the startup script:

```powershell
.\scripts\start-proxy.ps1
```

Linux/macOS:

```bash
chmod +x ./scripts/start-proxy.sh
./scripts/start-proxy.sh
```

First run with dependency install:

```powershell
.\scripts\start-proxy.ps1 -Install
```

Linux/macOS first run:

```bash
./scripts/start-proxy.sh --install
```

Script options:

| Platform | Option | Example |
| --- | --- | --- |
| Windows | `-Port` | `.\scripts\start-proxy.ps1 -Port 8010` |
| Windows | `-BindHost` | `.\scripts\start-proxy.ps1 -BindHost 0.0.0.0` |
| Windows | `-Reload` | `.\scripts\start-proxy.ps1 -Reload` |
| Linux/macOS | `--port` | `./scripts/start-proxy.sh --port 8010` |
| Linux/macOS | `--host` | `./scripts/start-proxy.sh --host 0.0.0.0` |
| Linux/macOS | `--reload` | `./scripts/start-proxy.sh --reload` |

## Client Setup

Configure Anthropic-compatible clients to use:

```text
http://127.0.0.1:8000
```

Claude Code environment example:

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8000"
$env:ANTHROPIC_API_KEY = "sk-..."
claude
```

Anthropic Python SDK example:

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:8000",
    api_key="sk-...",
)

message = client.messages.create(
    model="deepseek-v4-pro",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

No client changes are required for the normal repair path. The proxy uses
`tool_use.id` / `tool_result.tool_use_id` as the primary lookup key, so Claude
Code and Anthropic SDK clients can keep their default headers.

Optionally, send a stable header on every turn to improve collision isolation:

```text
x-conversation-id: your-session-id
```

If omitted, the proxy still repairs by tool call ID. It stores a best-effort
conversation key from `metadata.user_id`, then a deterministic hash of stable
request fields, but lookup does not require that key to match.

## Endpoint

```text
POST /v1/messages
```

The proxy forwards Anthropic Messages requests to:

```text
{UPSTREAM_BASE_URL}/v1/messages
```

Both regular JSON and `stream: true` SSE responses are supported. Streaming bytes are forwarded unchanged while the proxy reconstructs final assistant content for storage after `message_stop`.

For `POST /v1/messages`, if a request includes both
`thinking: {"type": "disabled"}` and Anthropic-format
`output_config.effort`, the proxy removes `output_config.effort` before
forwarding. DeepSeek rejects disabled thinking with a reasoning effort, and this
preserves the client's explicit non-thinking mode.

Other `/v1/*` endpoints are forwarded unchanged to the same upstream base URL.
For example, `GET /v1/models?limit=1000` is forwarded to
`{UPSTREAM_BASE_URL}/v1/models?limit=1000`, and
`POST /v1/messages/count_tokens?beta=true` is forwarded to
`{UPSTREAM_BASE_URL}/v1/messages/count_tokens?beta=true`. Thinking repair and
assistant response storage only run on `POST /v1/messages`.

Incoming request headers are forwarded to the upstream API as-is, except
hop-by-hop transport headers such as `connection`, `content-length`, and
`transfer-encoding`. `host` is rewritten to match `UPSTREAM_BASE_URL`, and
`accept-encoding` is forced to `identity` so upstream responses are not
compressed. API keys must be supplied by the client request, for example through
`x-api-key` or `authorization`; the proxy does not read or inject a local API
key.

## Repair Behavior

The proxy stores successful assistant responses that contain `tool_use` blocks.
For a later request containing `tool_result` blocks, it looks up the referenced
`tool_use.id` and repairs the outgoing `messages`:

- If the previous assistant message has the matching `tool_use` but lacks the
  stored `thinking` block, its `content` is replaced with the stored full
  assistant content.
- If no matching assistant message is present, the stored assistant message is
  inserted immediately before the `tool_result` user message.
- If the tool ID is unknown and `REPAIR_STRICT=false`, the request is forwarded
  unchanged.
- If the tool ID is unknown and `REPAIR_STRICT=true`, the proxy returns a local
  `400 unknown_tool_use_id`.

The proxy never synthesizes or modifies `thinking` text or signatures. It only
replays blocks previously returned by the upstream API.

## Full Exchange Logs

When `LOG_FULL_REQUEST=true`, each turn is written to:

```text
log/<conversation-id>/<turn>.json
```

The JSON shape follows [example.json](example.json):

```json
{
  "client": {
    "...": "full repaired upstream request JSON"
  },
  "response": {
    "sse": {
      "chunk_1": {
        "...": "stream event JSON"
      }
    },
    "full": {
      "...": "non-stream response JSON, or stream reconstructed as a message"
    }
  }
}
```

For non-streaming responses, `response.sse` is `{}` and `response.full` is the
upstream JSON response. For streaming responses, `response.sse` contains each SSE
JSON event as `chunk_N`, and `response.full` is reconstructed into a normal
assistant message object from the stream, including message metadata such as
`model`, `stop_reason`, `stop_sequence`, and `usage` when those fields appear in
SSE events.

If a logged string is itself a complete JSON object or JSON array, it is expanded
before writing the log file. This makes fields such as `metadata.user_id` easier
to inspect:

```json
{
  "metadata": {
    "user_id": {
      "device_id": "...",
      "session_id": "..."
    }
  }
}
```

Plain text remains plain text; only strings that parse as full JSON objects or
arrays are expanded.

## Storage

Uses stdlib `sqlite3`; no ORM. Stored assistant blocks preserve `thinking.signature`, unknown fields, block order, and `tool_use` fields. The proxy does not synthesize, summarize, rename, or reorder thinking blocks.

Tool call IDs are treated as the main repair key. The conversation key is still
stored and used to prefer same-conversation matches if a provider ever reuses a
tool ID.

## Security Notes

- Keep `.env` private; it can contain deployment-specific paths and URLs.
- Keep `LOG_FULL_REQUEST=false` unless actively debugging. Full exchange logs can
  contain prompts, tool outputs, and thinking blocks.
- The SQLite database may contain assistant thinking blocks and tool outputs.
  Store it on a trusted disk and delete it when no longer needed.
- For production multi-worker deployment, use a single writer or replace SQLite
  with a server database.

## License

This project is licensed under the GNU Affero General Public License v3.0. See
[LICENSE](LICENSE) for details.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```
