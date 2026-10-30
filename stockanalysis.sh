#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY315_PYTHON="$SCRIPT_DIR/.venv-py315/bin/python"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ -f "$SCRIPT_DIR/.env" ]; then
  ENV_FILE="$SCRIPT_DIR/.env"
elif [ -f "$SCRIPT_DIR/.env.example" ]; then
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
elif [ -x "$PY315_PYTHON" ] && "$PY315_PYTHON" -c "import sys, requests, pandas, tabulate, xlsxwriter, matplotlib; sys.exit(0 if sys.version_info[:2] == (3, 15) else 1)" >/dev/null 2>&1; then
  PYTHON_BIN="$PY315_PYTHON"
elif [ -x "$VENV_PYTHON" ] && "$VENV_PYTHON" -c "import requests, pandas, tabulate, xlsxwriter, matplotlib" >/dev/null 2>&1; then
  PYTHON_BIN="$VENV_PYTHON"
else
  PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" -c "import requests, pandas, tabulate, xlsxwriter, matplotlib" >/dev/null 2>&1; then
  echo "Missing Python dependencies for stockanalysis.sh." >&2
  echo "Selected Python interpreter failed its dependency check: $PYTHON_BIN" >&2
  if [ -x "$VENV_PYTHON" ]; then
    echo "Install them with: $VENV_PYTHON -m pip install -r \"$SCRIPT_DIR/requirements.txt\"" >&2
  else
    echo "Create a local virtualenv and install requirements:" >&2
    echo "  python3 -m venv \"$SCRIPT_DIR/.venv\"" >&2
    echo "  \"$SCRIPT_DIR/.venv/bin/python\" -m pip install -r \"$SCRIPT_DIR/requirements.txt\"" >&2
  fi
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/main.py" -t ELEC -o shirong \
  -e 10 \
  -b 0 \
  -p 7 \
  -l \
  -m \
  -s I \
  "$SCRIPT_DIR/stocklist_elec_list"
