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
|   |-- portfolio_allocation.py     # aggregate PAE runtime pipeline and port
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
|       |-- low_entry_score_v5.py   # quality-first panic-reversal LES v5
|       `-- portfolio_allocation_v1.py # portfolio allocation engine v1
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
|   |-- storage/portfolio_persistence.py # PAE 2-of-3 session evidence
|   |-- storage/chart_repository.py # chart output boundary
|   `-- notification/mail.py        # SMTP boundary
|-- config/
|   `-- settings.py                 # paths and runtime defaults
`-- data/                           # generated CSV outputs
```

### Low-Entry Model v5 Decision Mapping ###

| Condition | Decision |
|---|---|
| Confirmed hard fundamental failure | `NO_BUY` |
| Base data is not point-in-time or lacks minimum coverage | `DATA_INSUFFICIENT` |
| Base-only, ordinary quality-watch, or unsupported sector | `WATCH_ONLY` |
| Active temporary falling-knife block | `PANIC_WATCH` |
| Panic reversal detected, awaiting next-day confirmation | `REVERSAL_WATCH` |
| Outside Broad Rally: confirmed QS `>= 80`, score `>= 75` (or the higher risk-off requirement), Valuation `SUPPORTIVE`, Trigger, both risk gates, Market Gate and Falling-Knife `CLEAR` | conditional `BUY_SMALL` (1/8) |
| Completed broad-market panic reversal or depressed recovery plus quality, valuation and risk checks; includes the tightly guarded MOPS fallback starter | conditional `BUY_SMALL` (1/8) |
| Deep capitulation absorption with two panic conditions, three reversal confirmations, QS `>= 80`, Valuation PASS, both risk gates and no Veto | conditional `BUY_SMALL` (1/8) |
| At least two prior broad-rally sessions followed by an orderly stock retest, guarded breadth-starter QS `>= 75`, Valuation PASS, repriced risk PASS and no Veto | conditional `BUY_SMALL` (1/8) |
| Deep recent-rally give-back with directly verified QS `>= 80`, demand seed, Valuation PASS, repriced risk PASS and no Veto; next session must reclaim the full signal bar | conditional `BUY_SMALL` (1/8) |
| Strong completed breadth reversal with QS `>= 80`, score `>= 35`, raw score `>= 45`, four reversal confirmations, strong CLV/CMF, both risk gates and no Veto | conditional `BUY` (maximum 1/4) |
| Technical score `< 60` | `AVOID` |
| Technical score `60-69` | `WATCH` |
| Technical score `70` through the applicable entry threshold | `ARMED` |
| Required score plus Trigger, Valuation, Market Data and Risk gates | `BUY` |
| Reversal day followed by valid next-day confirmation and all risk gates | conditional `BUY_SMALL` |

### Expected Return Exit Model v3 ###

Expected Return Exit Model v3 (EREM v3) consumes OHLCV, TAIEX, plus optional entry LES and turnover history. Its deterministic component rules are:

- Trend, 0-20: EMA20 > EMA50 > EMA200 adds 12; Close > EMA20 adds 8.
- Momentum, 0-15: RSI14 in 45-70, MACD > Signal, and a rising MACD histogram each add 5; a two-day histogram decline or bearish divergence removes 5, with the component clipped to 0-15.
- Volatility, 0-10: ATR14 <= ATR20MA and Close below the upper Bollinger band each add 5; excessive ATR or an upper-band touch with momentum exhaustion removes 5, clipped to 0-10.
- Distribution, 0-15: rising OBV, price rising on volume above VolumeMA20, and no distribution warning each add 5. Warnings include price weakness on high volume, five falling OBV days, and price/OBV divergence.
- Remaining return, 0-15: Reward/Risk > 2 adds 10, 1-2 adds 5, and Close below `Mean120 + 2*StdDev120` adds 5. A zero or negative risk denominator produces `N/A`, never infinity.
- Relative strength, 0-10: positive `Stock Return60 - TAIEX Return60` adds 10, where `Return60 = Price[t] / Price[t-60] - 1`. At least 61 aligned trading-day observations are required.
- Trend structure, 0-5: both a Higher High and Higher Low across consecutive 20-day windows add 5.
- Entry quality, 0-5: Entry LES >= 75 adds 5; 60-74 adds 3. A value recorded on an entry row is forward-filled from the declared entry date.
- Turnover, 0-5: turnover strength >= 1.2 with non-declining price adds 5. If no turnover column is supplied, `Volume / VolumeMA20` is used as an explicit proxy.
- VWAP state, 0-5: Close >= rolling VWAP60 adds 5; Close below VWAP60 but rising toward it adds 2; continued weakness adds 0.

Entry LES influence decays with the holding period:

```text
EntryWeight = max(10%, 40% * exp(-HoldingDays / 20))
HoldingConfidence = (1 - EntryWeight) * CurrentEREM
                    + EntryWeight * EntryLES
```

When Entry LES exists but Holding Days is unavailable, the model uses only the 10% floor and reports the missing field; unknown recency cannot claim 40% new-entry protection.

EREM decision mapping:

- `STRONG HOLD`: Holding Confidence >= 85
- `HOLD`: 70 <= Holding Confidence < 85
- `HOLD`: 55 <= Holding Confidence < 70
- `REDUCE 25%`: 40 <= Holding Confidence < 55
- `EXIT`: Holding Confidence < 40

### Portfolio Allocation Engine v1 ###

Portfolio Allocation Engine v1 (`portfolio_allocation_v1`) consumes the final LES v5, EREM v3, Quality v5, and Valuation v5 outputs. It does not recompute their indicators or combine them into a master score. Its decision order is:

```text
PortfolioMode from effective holding count
  -> Hard Gates
  -> Conservative Expected Return (CER)
  -> Absolute Required Return
  -> COLD_START / BUILDING: skip pairwise dominance
  -> STEADY_STATE: Quality Advantage + CER Switch Hurdle
  -> STEADY_STATE: Dual Dominance + Persistence / Obvious Opportunity
  -> Concentration, Correlation and Liquidity Gates
  -> Target Weights and Cash
```

`PortfolioMode` is recalculated from the current effective holdings on every run: `0 = COLD_START`, `1 = BUILDING`, and `>= 2 = STEADY_STATE`. It does not depend on prior runs, so the same portfolio input always produces the same mode.

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

Full command-line reference:

```text
$ python main.py -h
usage: main.py [-h] [--list-strategies] [--check-config] [-t {VEH,ELEC,BUI,AIR,CHEMI,BIO,SEMI,COMP,COMM}] [-o {SHIRONG,shirong}] [-e ENDBACKTRACK] [-b BEGINBACKTRACK] [-s SUBJECT] [-cc CCRECEIVER]
               [-l] [-p PERIOD] [-m] [--skip-mops]
               [--mops-history-years MOPS_HISTORY_YEARS] [--account-equity ACCOUNT_EQUITY]
               [--pae-holdings-file PAE_HOLDINGS_FILE] [--pae-current-cash PAE_CURRENT_CASH]
               [--mode {release,debug}]
               [stocklist ...]

positional arguments:
  stocklist             Stock numbers directly, or legacy row indexes in a semicolon-separated stocklist file.

options:
  -h, --help            show this help message and exit
  --list-strategies     List all available trading strategies with descriptions and exit.
  --check-config        Validate settings and inputs, print a safe JSON summary, and exit without network or email.
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
  --pae-holdings-file PAE_HOLDINGS_FILE
                        PAE holdings text file using symbols or symbol:shares.
  --pae-current-cash PAE_CURRENT_CASH
                        Optional current portfolio cash in NTD.
  --mode {release,debug}
                        Runtime output mode. release suppresses diagnostic details; debug enables per-date and per-symbol diagnostics.
```

### Local Run Instructions ###

Create or refresh the local environment:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the automated workflow after configuring `.env`:

```sh
./stockdataanalysis.sh --profile semi
```

### Contact ###

Found a bug? [Send an email](mailto:s041978@hotmail.com).

Happy Investing! :)
