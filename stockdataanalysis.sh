#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: ./stockdataanalysis.sh [--profile PROFILE] [--check]

Profiles: elec, semi, chemi, bio, comp, air, veh

The profile selects stocklist_<profile>, the matching TWSE market type, and
portfolio_holdings_<profile>.txt when that holdings file exists. The default
profile is elec; TWSE_ANALYSIS_PROFILE provides the same selection by env var.
--check validates the complete runtime configuration without network or email.
EOF
}

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

ANALYSIS_PROFILE="${TWSE_ANALYSIS_PROFILE:-elec}"
CHECK_CONFIG=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --check)
      CHECK_CONFIG=1
      shift
      ;;
    --profile)
      if [ "$#" -lt 2 ]; then
        echo "--profile requires a value" >&2
        usage >&2
        exit 2
      fi
      ANALYSIS_PROFILE=$2
      shift 2
      ;;
    --profile=*)
      ANALYSIS_PROFILE=${1#--profile=}
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ANALYSIS_PROFILE=$(printf '%s' "$ANALYSIS_PROFILE" | tr '[:upper:]' '[:lower:]')
case "$ANALYSIS_PROFILE" in
  elec) STOCKTYPE=ELEC ;;
  semi) STOCKTYPE=SEMI ;;
  chemi) STOCKTYPE=CHEMI ;;
  bio) STOCKTYPE=BIO ;;
  comp) STOCKTYPE=COMP ;;
  air) STOCKTYPE=AIR ;;
  veh) STOCKTYPE=VEH ;;
  *)
    echo "Unsupported analysis profile: $ANALYSIS_PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

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

STOCKLIST="$SCRIPT_DIR/stocklist_$ANALYSIS_PROFILE"
if [ ! -r "$STOCKLIST" ]; then
  echo "Stock list is not readable: $STOCKLIST" >&2
  exit 1
fi

DEFAULT_PAE_HOLDINGS_FILE="$SCRIPT_DIR/portfolio_holdings_$ANALYSIS_PROFILE.txt"
PAE_HOLDINGS_FILE="${TWSE_PAE_HOLDINGS_FILE:-$DEFAULT_PAE_HOLDINGS_FILE}"
PAE_HOLDINGS_ENABLED=0
if [ -n "${TWSE_PAE_HOLDINGS_FILE:-}" ] && [ ! -r "$PAE_HOLDINGS_FILE" ]; then
  echo "PAE holdings file is not readable: $PAE_HOLDINGS_FILE" >&2
  exit 1
fi
if [ -r "$PAE_HOLDINGS_FILE" ] && awk '
  {
    sub(/#.*/, "")
    gsub(/[[:space:],;()]/, "")
    if (length($0) > 0) {
      found = 1
      exit
    }
  }
  END { exit(found ? 0 : 1) }
' "$PAE_HOLDINGS_FILE"; then
  PAE_HOLDINGS_ENABLED=1
elif [ -r "$PAE_HOLDINGS_FILE" ]; then
  echo "PAE holdings file is empty; using COLD_START_ALL_CASH: $PAE_HOLDINGS_FILE" >&2
fi

echo "TWSE analysis profile: $STOCKTYPE ($STOCKLIST)"

# Build argv without fragile multiline continuations; the stock list is data,
# never a shell command.
set -- "$SCRIPT_DIR/main.py"
set -- "$@" -o shirong
set -- "$@" -e 500
set -- "$@" -b 0
set -- "$@" -t "$STOCKTYPE"
set -- "$@" --mops-history-years "${MOPS_HISTORY_YEARS:-5}"
set -- "$@" --account-equity "${TWSE_ACCOUNT_EQUITY:-1000000}"
if [ "$PAE_HOLDINGS_ENABLED" -eq 1 ]; then
  set -- "$@" --pae-holdings-file "$PAE_HOLDINGS_FILE"
fi
set -- "$@" -m
if [ "$CHECK_CONFIG" -eq 1 ]; then
  set -- "$@" --check-config
fi
set -- "$@" "$STOCKLIST"

exec "$PYTHON_BIN" "$@"
