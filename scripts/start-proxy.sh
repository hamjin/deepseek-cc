#!/usr/bin/env bash
set -euo pipefail

BIND_HOST="127.0.0.1"
PORT="8000"
INSTALL="0"
RELOAD="0"

usage() {
  cat <<'EOF'
Usage: ./scripts/start-proxy.sh [options]

Options:
  --host HOST      Bind host. Default: 127.0.0.1
  --port PORT      Bind port. Default: 8000
  --install        Install project dependencies before starting
  --reload         Enable uvicorn reload
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      BIND_HOST="${2:?--host requires a value}"
      shift 2
      ;;
    --port)
      PORT="${2:?--port requires a value}"
      shift 2
      ;;
    --install)
      INSTALL="1"
      shift
      ;;
    --reload)
      RELOAD="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating virtual environment at .venv"
  python3 -m venv "$PROJECT_ROOT/.venv"
fi

if [[ "$INSTALL" == "1" ]]; then
  echo "Installing project dependencies"
  "$PYTHON" -m pip install -e "$PROJECT_ROOT[dev]"
fi

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  echo "Warning: .env not found. Defaults will be used unless you copy .env.example to .env." >&2
fi

UVICORN_ARGS=(
  -m uvicorn
  app.main:app
  --host "$BIND_HOST"
  --port "$PORT"
)

if [[ "$RELOAD" == "1" ]]; then
  UVICORN_ARGS+=(--reload)
fi

echo "Starting DeepSeek Anthropic Thinking Repair Proxy"
echo "URL: http://$BIND_HOST:$PORT"
echo "Stop: Ctrl+C"

cd "$PROJECT_ROOT"
exec "$PYTHON" "${UVICORN_ARGS[@]}"
