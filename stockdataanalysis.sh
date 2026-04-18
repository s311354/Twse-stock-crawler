#!/bin/sh
#
: $ python3 stockanalysis.py -h
:
: positional arguments:
:   outputformats         If a single file format is passed in, then we assume
:                         it contains a semicolon-separated list of files that
:                         we expect this script to output. If multiple file
:                         formats are passed in, then we assume output file
:                         formats are listed directly as arguments. Support
:                         text, xlsx formats
:   stocklist             If a single stock is passed in, then we assume it
:                         contains a semicolon-separated list of files that we
:                         expect this script to output. If multiple stocks are
:                         passed in, then we assume stocks are listed directly
:                         as arguments.
:   holidays_2022         Public holidays in Taiwan, comma separated.
:
: optional arguments:
:  -h, --help            show this help message and exit
:  --stocktype {VEH,ELEC,SEMI,AIR,BIO,COMM}
:                        The stock market you want to choose.
:  --output_file_names {SHIRONG,shirong}
:                        The owner you want to choose output file name.
:  --backtrackend BACKTRACKEND
:                        The owner you want to choose the end of backtrack
:                        days.
:  --backtrackstart BACKTRACKSTART
:                        The owner you want to choose the start of backtrack
:                        days. 
:  --emailsubject EMAILSUBJECT
:                        The owner you want to set the email Subject.
:  --ccreceiver CCRECEIVER
:                        The owner you want to cc the email to someone apart from the recipient.
#

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ -x "$VENV_PYTHON" ]; then
  PYTHON_BIN="$VENV_PYTHON"
else
  PYTHON_BIN="${PYTHON:-python3}"
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

"$PYTHON_BIN" "$SCRIPT_DIR/stockanalysis.py" -o shirong  \
                                             -e 7 \
                                             -b 0 \
                                             -t ELEC \
                                             -m \
                                             "$SCRIPT_DIR/stocklist_elec" "$SCRIPT_DIR/holidays_2026"
