## Taiwan Stock Exchange (TWSE) Crawler ##

### Introduction ###

Taiwan Stock Exchange (TWSE) Crawler is a Python tool for collecting daily TWSE market data and turning it into reproducible entry, exit, and risk signals. It retrieves price, volume, transaction statistics, and valuation metrics such as P/E ratio.

Use it for:

- Historical OHLCV analysis
- Technical indicator calculation
- Trading signal generation
- Strategy backtesting
- Performance evaluation
- Quantitative investment research

Historical data is stored as CSV files in the [data](./data) directory and can be further processed for technical analysis, strategy backtesting, and quantitative research.

### Python Architecture ###

The crawler keeps the legacy scripts compatible while introducing a cleaner, lower-coupling architecture:

```
.
|-- main.py                         # project entrypoint
|-- interface/
|   `-- cli.py                      # CLI argument adapter
|-- application/
|   |-- analysis_dataset.py         # one-pass analysis and decision assembly
|   |-- fundamental_service.py      # MOPS discovery/cache/derivation orchestration
|   |-- strategy_contracts.py       # schema validation and CSV projection
|   `-- stock_service.py            # workflow orchestration
|-- domain/
|   |-- fundamental_derivation.py   # point-in-time TTM/ROIC/FCF formulas
|   |-- model_contracts.py           # immutable versioned decision evidence
|   |-- models.py                   # domain aliases/models
|   |-- fundamentals.py             # point-in-time MOPS quality gate
|   |-- quality_v5.py               # LES v5 Base/Extended quality and confidence
|   |-- pipeline.py                 # pandas pipeline stages
|   |-- repositories.py             # stock repository interface
|   |-- services.py                 # pure profit/indicator services
|   |-- strategy.py                 # strategy abstraction/plugin registry
|   `-- strategies/
|       |-- expected_return_exit.py # Expected Return Exit Model v1 strategy
|       |-- expected_return_exit_v2.py # Expected Return Exit Model v2 strategy
|       |-- expected_return_exit_v3.py # Expected Return Exit Model v3 strategy
|       |-- low_entry_score.py      # built-in Low Entry Score v1 strategy
|       |-- low_entry_score_v2.py   # refined Low Entry Score v2 strategy
|       |-- low_entry_score_v3.py   # advanced Low Entry Score v3 strategy
|       |-- low_entry_score_v4.py   # backward-compatible LES v4 strategy
|       |-- low_entry_score_v4_1.py # quality-first stabilized LES v4.1
|       `-- low_entry_score_v5.py   # quality-first panic-reversal LES v5
|-- infrastructure/
|   |-- crawler/twse_client.py      # TWSE HTTP boundary
|   |-- fundamentals/
|   |   |-- mops_client.py          # filing/announcement/correction discovery
|   |   |-- ixbrl_parser.py         # normalized Inline XBRL facts
|   |   |-- mops_repository.py      # SQLite metadata + raw revision cache
|   |   `-- twse_financial_batch.py # daily all-company financial snapshots
|   |-- report/html_renderer.py     # strategy HTML renderer factory
|   |-- storage/csv_repository.py   # CSV output boundary
|   |-- storage/stock_repository.py # CSV/SQLite/DuckDB stock repositories
|   |-- storage/chart_repository.py # chart output boundary
|   `-- notification/mail.py        # SMTP boundary
|-- config/
|   `-- settings.py                 # paths and runtime defaults
`-- data/                           # generated CSV outputs
```

Use [`main.py`](main.py) for the clean architecture entrypoint. [`stockanalysis.py`](stockanalysis.py) remains as a legacy-compatible module while the new CLI delegates orchestration to [`application/stock_service.py`](application/stock_service.py).

### Strategy Plugin System ###

Strategies implement the [`Strategy`](domain/strategy.py) abstraction:

```python
class Strategy(ABC):
    name: str

    def run(self, data: pandas.DataFrame) -> pandas.DataFrame:
        ...
```

Built-in strategies are lazily registered through [`domain/strategy.py`](domain/strategy.py). External strategies can register themselves with `register_strategy(MyStrategy())`.

### Low-Entry Model v5 Decision Mapping ###

| Condition | Decision |
|---|---|
| Confirmed hard fundamental failure | `NO_BUY` |
| Base data is not point-in-time or lacks minimum coverage | `DATA_INSUFFICIENT` |
| Base-only, ordinary quality-watch, or unsupported sector | `WATCH_ONLY` |
| Active temporary falling-knife block | `PANIC_WATCH` |
| Panic reversal detected, awaiting next-day confirmation | `REVERSAL_WATCH` |
| Completed broad-market panic reversal or depressed recovery plus quality, valuation and risk checks; includes the tightly guarded MOPS fallback starter | conditional `BUY_SMALL` (1/8) |
| Deep capitulation absorption with two panic conditions, three reversal confirmations, QS `>= 80`, Valuation PASS, both risk gates and no Veto | conditional `BUY_SMALL` (1/8) |
| At least two prior broad-rally sessions followed by an orderly stock retest, guarded breadth-starter QS `>= 75`, Valuation PASS, repriced risk PASS and no Veto | conditional `BUY_SMALL` (1/8) |
| Deep recent-rally give-back with directly verified QS `>= 80`, demand seed, Valuation PASS, repriced risk PASS and no Veto; next session must reclaim the full signal bar | conditional `BUY_SMALL` (1/8) |
| Strong completed breadth reversal with QS `>= 80`, score `>= 35`, raw score `>= 45`, four reversal confirmations, strong CLV/CMF, both risk gates and no Veto | conditional `BUY` (maximum 1/4) |
| Technical score `< 60` | `AVOID` |
| Technical score `60-69` | `WATCH` |
| Technical score `70` through the applicable entry threshold | `ARMED` |
| Required score plus Trigger, Valuation, Market Data and Risk gates | `BUY` |
| Reversal day followed by valid next-day confirmation and all risk gates | `EARLY_BUY` |

The v5 Valuation Gate is produced by the point-in-time service in
`domain/valuation_v5.py`; callers no longer need to inject four precomputed
snapshot keys. It uses only annual filings whose `available_at` is no later
than the signal date, plus a point-in-time TTM snapshot and the recent market
P/E median. Required evidence is normalized EPS, revenue growth, operating
margin, expected dividend per share, share dilution, current NTD price, and a
positive recent P/E. If any input is absent, the report shows
`Valuation Model Status=UNKNOWN`, lists the missing evidence in
`Valuation Reasons`, and cannot emit `BUY` or `EARLY_BUY`.

### Expected Return Exit Model v3 ###

Expected Return Exit Model v3 (EREM v3) consumes OHLCV, TAIEX, plus optional entry LES and turnover history. Its deterministic component rules are:

- Trend, 0-20: EMA20 > EMA50 > EMA200 adds 12; Close > EMA20 adds 8.
- Momentum, 0-15: RSI14 in 45-70, MACD > Signal, and a rising MACD histogram each add 5; a two-day histogram decline or bearish divergence removes 5, with the component clipped to 0-15.
- Volatility, 0-10: ATR14 <= ATR20MA and Close below the upper Bollinger band each add 5; excessive ATR or an upper-band touch with momentum exhaustion removes 5, clipped to 0-10.
- Distribution, 0-15: rising OBV, price rising on volume above VolumeMA20, and no distribution warning each add 5. Warnings include price weakness on high volume, five falling OBV days, and price/OBV divergence.
- Remaining return, 0-15: Reward/Risk > 2 adds 10, 1-2 adds 5, and Close below `Mean120 + 2*StdDev120` adds 5. A zero or negative risk denominator produces `N/A`, never infinity.
- Relative strength, 0-10: positive `Stock Return60 - TAIEX Return60` adds 10, where `Return60 = Price[t] / Price[t-60] - 1`. At least 61 aligned trading-day observations are required.
- Trend structure, 0-5: both a Higher High and Higher Low across consecutive 20-day windows add 5.
- Entry quality, 0-5: Entry LES >= 75 adds 5; 60-74 adds 3. A value recorded on an entry row is forward-filled so the original entry quality is preserved throughout later holding rows.
- Turnover, 0-5: turnover strength >= 1.2 with non-declining price adds 5. If no turnover column is supplied, `Volume / VolumeMA20` is used as an explicit proxy.
- VWAP safety, 0-5: Close <= rolling VWAP60 adds 5.

EREM decision mapping:

- `STRONG HOLD`: Holding Confidence >= 85
- `HOLD`: 70 <= Holding Confidence < 85
- `HOLD`: 55 <= Holding Confidence < 70
- `REDUCE 25%`: 40 <= Holding Confidence < 55
- `EXIT`: Holding Confidence < 40

The 55-69 band deliberately remains `HOLD`, and the 40-54 band limits reduction to 25%, reducing unnecessary turnover and overtrading.

### Email Configuration ###

Email is optional when running `main.py`, but `stockdataanalysis.sh` enables it with `-m`. Create an untracked `.env` with your own values; use [`.env.example`](./.env.example) only as a list of supported variables. Configure:

- `TWSE_SMTP_EMAIL`: sender account
- `TWSE_SMTP_PASSWORD`: SMTP password or app password
- `TWSE_SMTP_RECEIVERS`: comma-separated recipients
- `TWSE_SMTP_CC`: optional CC recipients
- `TWSE_SMTP_HOST` and `TWSE_SMTP_PORT`: optional server override

Never commit real credentials. Keep `.env.example` limited to placeholders and rotate any credential accidentally committed there.

### Usage ###

List all available trading strategies:

```sh
python main.py --list-strategies
```

This displays all registered strategies with their versions and descriptions:

```text
Available Trading Strategies:

Name                           Version    Description
----------------------------------------------------------------------------------------------------
expected_return_exit           N/A
expected_return_exit_v2        N/A
expected_return_exit_v3        3.0.0      Holding confidence scoring with trend, momentum, ...
low_entry_score                N/A
low_entry_score_v2             N/A
low_entry_score_v3             N/A
low_entry_score_v4             N/A
low_entry_score_v4_1           4.1.0      Balanced entry profile with Quality Gate, ...
low_entry_score_v5             5.0.0      Quality-first panic reversal entry strategy ...
```

Full command-line reference:

```text
$ python main.py -h
usage: main.py [-h] [--list-strategies] [-t {VEH,ELEC,BUI,AIR,CHEMI,BIO,SEMI,COMP,COMM}] [-o {SHIRONG,shirong}] [-e ENDBACKTRACK] [-b BEGINBACKTRACK] [-s SUBJECT] [-cc CCRECEIVER]
               [-l] [-p PERIOD] [-m] [--skip-mops]
               [--mops-history-years MOPS_HISTORY_YEARS] [--account-equity ACCOUNT_EQUITY] [--mode {release,debug}]
               [stocklist ...]

positional arguments:
  stocklist             Stock numbers directly, or legacy row indexes in a semicolon-separated stocklist file.

options:
  -h, --help            show this help message and exit
  --list-strategies     List all available trading strategies with descriptions and exit.
  -t, --type {VEH,ELEC,BUI,AIR,CHEMI,BIO,SEMI,COMP,COMM}
                        The stock market you want to choose.
  -o, --output_file_names {SHIRONG,shirong}
                        Output file name prefix.
  -e, --endbacktrack ENDBACKTRACK
                        Older backtrack boundary in calendar days (inclusive).
  -b, --beginbacktrack BEGINBACKTRACK
                        Newer backtrack boundary in calendar days (exclusive).
  -s, --subject SUBJECT
                        Email subject.
  -cc, --ccreceiver CCRECEIVER
                        Email CC receiver.
  -l, --linechart       Show a stock profit-ratio line chart.
  -p, --period PERIOD   Line-chart period.
  -m, --mail            Send email to recipients.
  --skip-mops           Skip MOPS/iXBRL refresh; LES v5 Extended Quality is unavailable and formal BUY is disabled.
  --mops-history-years MOPS_HISTORY_YEARS
                        Annual filing history used by LES v5 Extended Quality (minimum 3, default 5).
  --account-equity ACCOUNT_EQUITY
                        NTD account equity used for board-lot risk sizing.
  --mode {release,debug}
                        Runtime output mode. release suppresses diagnostic details; debug enables per-date and per-symbol diagnostics.
```

### Local Run Instructions ###

Create or refresh the local environment:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run once without email:

```sh
.venv/bin/python main.py -o shirong -e 500 -b 5 -t ELEC ./stocklist_elec
```

Run the automated workflow after configuring `.env`:

```sh
./stockdataanalysis.sh
```

Runtime output defaults to `release`: it keeps market status, aggregate progress,
warnings/errors, the final structured `run_summary`, and `Sending Email at ...`.
Per-date, per-symbol, cache/catalog, and intermediate indicator diagnostics are
available only in `debug` mode:

```sh
TWSE_RUNTIME_MODE=debug ./stockdataanalysis.sh
.venv/bin/python main.py --mode debug -o shirong -e 10 -b 0 -t ELEC ./stocklist_elec
```

Use `TWSE_RUNTIME_MODE=release` or `--mode release` for routine scheduled runs.
`TWSE_LOG_LEVEL` remains supported for release severity filtering, but setting it
to `DEBUG` does not bypass release mode.

`stockdataanalysis.sh` enables MOPS/iXBRL by default and accepts optional runtime sizing controls:

```sh
MOPS_HISTORY_YEARS=5 TWSE_ACCOUNT_EQUITY=1000000 ./stockdataanalysis.sh
```

The first run can take longer because it creates each stock's five-year iXBRL cache. The first run on a new Taipei calendar day verifies the two recent periods once; an unchanged same-day rerun should log `remote_refresh_skipped=true` per cached stock with no MOPS catalog/download traffic. TWSE batch coverage is optional enrichment and does not disable this fast path. To verify the cache-first core without contacting TWSE/MOPS:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/fundamentals/ -q --assert=plain
```

The script loads the untracked `.env` when present; `.env.example` is documentation only and is never sourced. It uses `$PYTHON` when provided, otherwise it tries a dependency-ready `python3`, then falls back to `.venv/bin/python`.

To force a specific interpreter:

```
PYTHON=python3 ./stockdataanalysis.sh
```

To send mail from the clean entrypoint directly:

```
.venv/bin/python main.py -o shirong -e 500 -b 5 -t ELEC -m ./stocklist_elec
```

The shell script automatically loads SMTP and runtime variables from `.env` when present, or `.env.example` as a fallback. Real email sending still requires real `TWSE_SMTP_*` values.

Outputs are written under [`data`](data/), including daily TWSE CSV files, `shirong_analysis_dataset.csv`, and the ignored `data/mops/` raw/SQLite cache. The analysis dataset and email include LES v5 Base/Extended Quality status, score and Data Confidence, Critical Field Coverage, Valuation Gate, Panic/Reversal evidence, Falling-Knife state, all six technical components, required score, Trigger, risk gate, `RewardRisk`, stop/target, `Position Fraction`, and suggested executable shares, plus the existing EREM v3 columns.

Run tests:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q --assert=plain
```

### Build the Docker Image ###

```
docker build . --tag=user/twsestockcrawler:0.01
```

### Run the Docker Image ###

Use a bind mount to make a host directory available at `/mnt`:

```sh
docker run --rm -it -v "$(pwd)/${MOUNT}:/mnt" user/twsestockcrawler:0.01
```

### Potential Defect ###

The crawler now locks each tracked slot to a TWSE stock number before appending OHLC history. If a tracked stock is missing on a date, the row is skipped. Max-profit calculations still depend on the selected backtrack window and available valid trading days.

### Contact ###

Found a bug? [Send an email](mailto:s041978@hotmail.com).

Happy Investing! :)
