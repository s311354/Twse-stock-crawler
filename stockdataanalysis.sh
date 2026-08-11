#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ -f "$SCRIPT_DIR/.env.example" ]; then
  ENV_FILE="$SCRIPT_DIR/.env.example"
else
  ENV_FILE=""
fi

if [ -n "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
  case "${TWSE_RUNTIME_MODE:-release}" in
    debug|DEBUG) echo "Loaded environment variables from $ENV_FILE" ;;
  esac
fi

if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="${PYTHON:-python3}"
elif python3 -c "import requests, pandas, tabulate, xlsxwriter, matplotlib" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif [ -x "$VENV_PYTHON" ]; then
  PYTHON_BIN="$VENV_PYTHON"
else
  PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" -c "import requests, pandas, tabulate, xlsxwriter, matplotlib" >/dev/null 2>&1; then
  echo "Missing Python dependencies for stockanalysis.sh." >&2
  if [ -x "$VENV_PYTHON" ]; then
    echo "Install them with: $VENV_PYTHON -m pip install -r \"$SCRIPT_DIR/requirements.txt\"" >&2
  else
    echo "Create a local virtualenv and install requirements:" >&2
    echo "  python3 -m venv \"$SCRIPT_DIR/.venv\"" >&2
    echo "  \"$SCRIPT_DIR/.venv/bin/python\" -m pip install -r \"$SCRIPT_DIR/requirements.txt\"" >&2
  fi
  exit 1
fi

STOCKLIST="$SCRIPT_DIR/stocklist_elec"
if [ ! -r "$STOCKLIST" ]; then
  echo "Stock list is not readable: $STOCKLIST" >&2
  exit 1
fi

# Build argv without fragile multiline continuations; the stock list is data,
# never a shell command.
set -- "$SCRIPT_DIR/main.py"
set -- "$@" -o shirong
set -- "$@" -e 500
set -- "$@" -b 0
set -- "$@" -t ELEC
set -- "$@" --mops-history-years "${MOPS_HISTORY_YEARS:-5}"
set -- "$@" --account-equity "${TWSE_ACCOUNT_EQUITY:-1000000}"
set -- "$@" -m
set -- "$@" "$STOCKLIST"

exec "$PYTHON_BIN" "$@"
