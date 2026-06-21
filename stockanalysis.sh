#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
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
  echo "Loaded environment variables from $ENV_FILE"
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

"$PYTHON_BIN" "$SCRIPT_DIR/main.py" -t ELEC -o shirong \
														-e 10 \
														-b 0 \
														-p 7 \
														-l \
														-m \
														-s I \
                                             			"$SCRIPT_DIR/stocklist_elec_list" "$SCRIPT_DIR/holidays_2026"
