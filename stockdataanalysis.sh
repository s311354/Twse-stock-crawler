#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: ./stockdataanalysis.sh [--profile PROFILE] [-b DAYS] [-e DAYS] [--check]
       ./stockdataanalysis.sh --profile PROFILE -b START_DATE -e END_DATE
         --pae-holdings-file HISTORICAL_FILE --pae-current-cash CASH
         --account-equity EQUITY --replay-id ID --replay-evidence-file JSON [--check]

Profiles: elec, semi, chemi, bio, comp, air, veh, eleccomp

The semi profile selects TPEx OTC semiconductor daily quotes (all stock numbers).
Other profiles select stocklist_<profile>, the matching TWSE market type, and
portfolio_holdings_<profile>.txt when that holdings file exists. The default
profile is elec; TWSE_ANALYSIS_PROFILE provides the same selection by env var.
Holding token format:
  symbol[:shares[:entry_date[:entry_les[:entry_decision]]]]
where entry_decision is BUY, EARLY_BUY, or BUY_SMALL. Existing shorter tokens
remain valid.
--check validates the complete runtime configuration without network or email.
Integer -b/-e retain calendar-day offsets relative to the execution date.
Date pairs (-b YYYY-MM-DD -e YYYY-MM-DD) select an inclusive history range and
automatically replay at END_DATE 13:30 Taipei. Timestamps and premarket/intraday
replay are not supported. Do not mix dates and days.
Historical date replay reads pinned evidence, writes isolated resumable
research outputs, and never emails. Prepare restrictions in the evidence package.
The evidence package, not today's stocklist, supplies the replay universe.
--check still sends nothing. An explicit historical portfolio is
required; an empty holdings file declares no holdings. Normal runs still email.
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
BACKTRACK_END=700
BACKTRACK_BEGIN=0
EXPLICIT_HOLDINGS=""
EXPLICIT_CASH=""
EXPLICIT_EQUITY=""
REPLAY_RESTRICTIONS=""
REPLAY_MAIL=0
REPLAY_ID=""
REPLAY_EVIDENCE=""
REPLAY_STATE=""
REPLAY_OUTPUT=""
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
    --replay-mail)
      REPLAY_MAIL=1
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
    -b|-e|--pae-holdings-file|--pae-current-cash|--account-equity|--replay-restrictions-file|--replay-id|--replay-evidence-file|--replay-state-path|--replay-output-dir)
      if [ "$#" -lt 2 ]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      case "$1" in
        -b) BACKTRACK_BEGIN=$2 ;;
        -e) BACKTRACK_END=$2 ;;
        --pae-holdings-file) EXPLICIT_HOLDINGS=$2 ;;
        --pae-current-cash) EXPLICIT_CASH=$2 ;;
        --account-equity) EXPLICIT_EQUITY=$2 ;;
        --replay-restrictions-file) REPLAY_RESTRICTIONS=$2 ;;
        --replay-id) REPLAY_ID=$2 ;;
        --replay-evidence-file) REPLAY_EVIDENCE=$2 ;;
        --replay-state-path) REPLAY_STATE=$2 ;;
        --replay-output-dir) REPLAY_OUTPUT=$2 ;;
      esac
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
# Detect date syntax only; Python validates real calendar dates and ordering.
# Replay must be selected BEFORE consulting current holdings or adding -m.
DATE_RANGE=0
case "$BACKTRACK_BEGIN" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) DATE_RANGE=1 ;;
esac
case "$BACKTRACK_END" in
  [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9])
    if [ "$DATE_RANGE" -ne 1 ]; then
      echo "-b/-e must both be integers or both YYYY-MM-DD dates" >&2
      exit 2
    fi
    ;;
  *)
    if [ "$DATE_RANGE" -eq 1 ]; then
      echo "-b/-e must both be integers or both YYYY-MM-DD dates" >&2
      exit 2
    fi
    ;;
esac
REPLAY_MODE=$DATE_RANGE
if [ "$REPLAY_MAIL" -eq 1 ] && [ "$REPLAY_MODE" -ne 1 ]; then
  echo "--replay-mail requires a date -b/-e pair" >&2
  exit 2
fi
if [ "$REPLAY_MODE" -eq 1 ]; then
  BACKTRACK_BEGIN=${BACKTRACK_BEGIN:-0}
  if [ -z "$EXPLICIT_HOLDINGS" ] || [ -z "$EXPLICIT_CASH" ] || [ -z "$EXPLICIT_EQUITY" ]; then
    echo "Replay requires explicit historical --pae-holdings-file, --pae-current-cash and --account-equity (not current environment defaults)." >&2
    exit 2
  fi
else
  BACKTRACK_BEGIN=${BACKTRACK_BEGIN:-9}
fi

ANALYSIS_PROFILE=$(printf '%s' "$ANALYSIS_PROFILE" | tr '[:upper:]' '[:lower:]')
case "$ANALYSIS_PROFILE" in
  elec) STOCKTYPE=ELEC ;;
  semi) STOCKTYPE=SEMI ;;
  chemi) STOCKTYPE=CHEMI ;;
  bio) STOCKTYPE=BIO ;;
  comp) STOCKTYPE=COMP ;;
  air) STOCKTYPE=AIR ;;
  veh) STOCKTYPE=VEH ;;
  eleccomp) STOCKTYPE=ELECCOMP ;;
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
if [ "$REPLAY_MODE" -ne 1 ] && [ "$ANALYSIS_PROFILE" != "semi" ] && [ ! -r "$STOCKLIST" ]; then
  echo "Stock list is not readable: $STOCKLIST" >&2
  exit 1
fi

DEFAULT_PAE_HOLDINGS_FILE="$SCRIPT_DIR/portfolio_holdings_$ANALYSIS_PROFILE.txt"
PAE_HOLDINGS_FILE="${TWSE_PAE_HOLDINGS_FILE:-$DEFAULT_PAE_HOLDINGS_FILE}"
if [ -n "$EXPLICIT_HOLDINGS" ]; then
  PAE_HOLDINGS_FILE=$EXPLICIT_HOLDINGS
  if [ ! -r "$PAE_HOLDINGS_FILE" ]; then
    echo "PAE holdings file is not readable: $PAE_HOLDINGS_FILE" >&2
    exit 1
  fi
fi
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
if [ -n "$EXPLICIT_HOLDINGS" ]; then
  PAE_HOLDINGS_ENABLED=1
fi

if [ "$REPLAY_MODE" -eq 1 ]; then
  STOCKLIST=ALL
  echo "Historical replay: market and universe from pinned evidence (no email)"
elif [ "$ANALYSIS_PROFILE" = "semi" ]; then
  STOCKLIST=ALL
  echo "TPEx analysis profile: SEMI (historical sector universe by stock number)"
else
  echo "TWSE analysis profile: $STOCKTYPE ($STOCKLIST)"
fi

# Build argv without fragile multiline continuations; the stock list is data,
# never a shell command.
set -- "$SCRIPT_DIR/main.py"
set -- "$@" -o shirong
set -- "$@" -e "$BACKTRACK_END"
set -- "$@" -b "$BACKTRACK_BEGIN"
set -- "$@" -t "$STOCKTYPE"
set -- "$@" --mops-history-years "${MOPS_HISTORY_YEARS:-5}"
set -- "$@" --account-equity "${EXPLICIT_EQUITY:-${TWSE_ACCOUNT_EQUITY:-1000000}}"
if [ "$PAE_HOLDINGS_ENABLED" -eq 1 ]; then
  set -- "$@" --pae-holdings-file "$PAE_HOLDINGS_FILE"
fi
if [ -n "$EXPLICIT_CASH" ]; then
  set -- "$@" --pae-current-cash "$EXPLICIT_CASH"
fi
if [ "$REPLAY_MODE" -eq 1 ]; then
  if [ "$REPLAY_MAIL" -eq 1 ]; then
    set -- "$@" --replay-mail
  fi
else
  set -- "$@" -m
fi
if [ -n "$REPLAY_RESTRICTIONS" ]; then
  set -- "$@" --replay-restrictions-file "$REPLAY_RESTRICTIONS"
fi
if [ -n "$REPLAY_ID" ]; then
  set -- "$@" --replay-id "$REPLAY_ID"
fi
if [ -n "$REPLAY_EVIDENCE" ]; then
  set -- "$@" --replay-evidence-file "$REPLAY_EVIDENCE"
  if [ "$REPLAY_MODE" -eq 1 ]; then STOCKLIST=ALL; fi
fi
if [ -n "$REPLAY_STATE" ]; then
  set -- "$@" --replay-state-path "$REPLAY_STATE"
fi
if [ -n "$REPLAY_OUTPUT" ]; then
  set -- "$@" --replay-output-dir "$REPLAY_OUTPUT"
fi
if [ "$CHECK_CONFIG" -eq 1 ]; then
  set -- "$@" --check-config
fi
set -- "$@" "$STOCKLIST"

exec "$PYTHON_BIN" "$@"
