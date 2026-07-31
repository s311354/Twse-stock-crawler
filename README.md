## Taiwan Stock Exchange (TWSE) Crawler ##

Turn Taiwan market data into testable trading decisions.

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
|   |-- fundamental_service.py      # MOPS discovery/cache/derivation orchestration
|   `-- stock_service.py            # workflow orchestration
|-- domain/
|   |-- fundamental_derivation.py   # point-in-time TTM/ROIC/FCF formulas
|   |-- models.py                   # domain aliases/models
|   |-- fundamentals.py             # point-in-time MOPS quality gate
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
|       `-- low_entry_score_v4_1.py # quality-first stabilized LES v4.1
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

The daily report combines `low_entry_score_v4_1` from [`domain/strategies/low_entry_score_v4_1.py`](domain/strategies/low_entry_score_v4_1.py) with `expected_return_exit_v3` from [`domain/strategies/expected_return_exit_v3.py`](domain/strategies/expected_return_exit_v3.py). LES v1-v4 and EREM v1-v2 remain registered for compatibility.

Low Entry Model v4.1 defaults to `BALANCED_PROFILE` and evaluates entry eligibility in this fixed order:

1. `Quality Gate`: point-in-time TWSE/MOPS facts return `PASS`, `FAIL`, or `UNKNOWN`. Missing history, missing fields, invalid values, unsupported non-electronic industry codes, unverified publication provenance, or facts published after the signal time can never become `PASS`.
2. `Market Data Gate`: requires at least 60 ordered, unique, completed daily bars with valid OHLCV and date-aligned positive TAIEX closes. The current calendar date is probed, but it is retained only after the official TWSE industry table and TAIEX close both exist; a missing intraday closing table is skipped.
3. `No Veto`: seven non-offsettable checks cover a break below the prior low, high-volume weak close, materially accelerating decline, repeated breakdown with negative three-day close location, a true gap from above prior support that remains unrecovered, simultaneous TAIEX/RS20 deterioration, and a Quality status that is neither `PASS` nor an explicitly eligible balanced `PROVISIONAL`.
4. `Score`: market regime 15, price position 15, stabilization 30, demand confirmation 20, momentum 10, and execution/risk 10.
5. `Balanced Trigger`: requires one price confirmation (`Close > EMA5 and VWAP20`, a prior-high breakout, or a confirmed higher low with rising EMA5), `CLV > 0.2`, and one demand confirmation (improving `CMF20`, positive five-day OBV slope, or up-day volume ratio at least 0.6).
6. `Risk Gate`: requires a confirmed swing-low stop, stop distance no more than `1.8 ATR14`, reward/risk at least 1.7, executable TWSE odd-lot sizing, and sufficient liquidity. The strict profile retains the `1.5 ATR14`/RR 2 gate.

### Low-Entry Model v4.1 Decision Mapping ###

For a stock whose `Quality Gate` is `PASS`, `BALANCED_PROFILE` maps the
Low-Entry score and execution evidence as follows:

| Condition | Decision | Meaning |
|---|---|---|
| Score `< 55` | `AVOID` | Entry quality is below the minimum watch threshold. |
| Score `55-64` | `WATCH` | Monitor for stabilization and improving demand. |
| Score `>= 65`, but Trigger or Risk Gate is incomplete | `ARMED` | Candidate is ready for confirmation, but no order is authorized. |
| Score `>= 75` and Trigger, Risk Gate, and Market Data Gate all pass | `BUY` | Standard entry is eligible when no Falling-Knife Veto exists. |

A score never overrides `Quality Gate`, `Market Data Gate`, or a
Falling-Knife Veto. Industry-specific and provisional `BUY_SMALL` paths retain
their own stricter eligibility, trigger, liquidity, and risk requirements.

### MOPS / iXBRL Data Path ###

The automatic pipeline uses the official MOPS single-company XBRL catalog, historical material announcements, and financial-report correction query. It then:

1. Uses the board-approved MOPS announcement only to discover an eligible filing period; it is not mislabeled as the full iXBRL filing timestamp.
2. Downloads only the latest filing, its prior-year comparable, and the configured annual history.
3. Stores each raw document under `data/mops/raw/<symbol>/<year>Q<quarter>/<sha256>.html`.
4. Stores filing metadata and normalized facts in `data/mops/mops.sqlite3` tables `filings` and `facts`.
5. Preserves changed content as a new `revision_number`; identical content hashes remain idempotent.
6. Records the first successful raw-document observation as the conservative `available_at` boundary. Corrections preserve their announcement evidence, but revised facts cannot become effective before the crawler actually observed that revision.
7. Restores iXBRL scale, context period, unit, and dimensions before deriving annual FCF/ROIC and TTM values.

The MOPS path is cache-first per stock:

- The first successful live run seeds five annual reports plus the latest periods required for TTM calculation. If the catalog does not expose an exact filing timestamp, the raw iXBRL is still downloaded and versioned, but it becomes point-in-time eligible only from `FIRST_OBSERVED_DOWNLOAD`; historical runs do not seed or consume undated candidates. `MOPS_HISTORY_YEARS` controls the annual window and defaults to five in `stockdataanalysis.sh`.
- Each daily run fetches the all-company general-industry income statement (`t187ap06_L_ci`) and balance sheet (`t187ap07_L_ci`) exactly once through the shared TWSE HTTP client. Rows are joined only when company and reporting period match.
- A complete five-year iXBRL cache bypasses all per-company MOPS requests when the TWSE batch period and content hash are unchanged and the monthly correction check is current. A new quarter or changed same-quarter batch hash re-enables MOPS discovery and verifies the two most recent required filings.
- Once per Taipei calendar month, the planner rechecks correction announcements for the complete retained catalog. A corrected period is downloaded as a new immutable revision.
- An unchanged filing-set hash reuses the stored annual/TTM fundamental bundle used by the Quality Gate instead of deriving it again.
- All periods for one company share one SQLite connection. Each filing and its normalized facts use one atomic transaction.
- Point-in-time reads use one `filings LEFT JOIN facts` query and the indexed integer `available_at_epoch`; the query does not apply `julianday()` to stored rows.

The default MOPS cache is `data/mops`. SQLite retries a transient open failure three times and uses a 30-second busy timeout. If the project is on an external disk that is intermittently unavailable, place only the cache on an internal disk:

```sh
MOPS_CACHE_DIR="$HOME/Library/Caches/twse-stock-crawler/mops" ./stockdataanalysis.sh
```

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

```text
$ python main.py -h
usage: main.py [-h] [-t {VEH,ELEC,BUI,AIR,CHEMI,BIO,SEMI,COMP,COMM}] [-o {SHIRONG,shirong}] [-e ENDBACKTRACK] [-b BEGINBACKTRACK] [-s SUBJECT] [-cc CCRECEIVER]
               [-l] [-p PERIOD] [-m] [--skip-mops]
               [--mops-history-years MOPS_HISTORY_YEARS] [--account-equity ACCOUNT_EQUITY]
               stocklist [stocklist ...]

positional arguments:
  stocklist             Stock numbers directly, or legacy row indexes in a semicolon-separated stocklist file.

options:
  -h, --help            show this help message and exit
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
  --skip-mops           Skip MOPS/iXBRL refresh; LES v4 remains WATCH_ONLY without cached context.
  --mops-history-years MOPS_HISTORY_YEARS
                        Annual filing history used by LES v4 (minimum 3, default 5).
  --account-equity ACCOUNT_EQUITY
                        NTD account equity used for board-lot risk sizing.
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

`stockdataanalysis.sh` enables MOPS/iXBRL by default and accepts optional runtime sizing controls:

```sh
MOPS_HISTORY_YEARS=5 TWSE_ACCOUNT_EQUITY=1000000 ./stockdataanalysis.sh
```

The first run can take longer because it creates each stock's five-year iXBRL cache. A later unchanged daily run should log one `TWSE financial batch loaded ... requests=2` message and `remote_refresh_skipped=true` per cached stock, with no MOPS catalog/download traffic. New quarters, changed same-quarter values, and the monthly correction pass may log `verify_revision` for the two recent periods. To verify the cache-first core without contacting TWSE/MOPS:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/fundamentals/ -q --assert=plain
```

The script loads `.env` when present. If it is absent, it falls back to `.env.example`; keep that file limited to non-secret placeholders. It uses `$PYTHON` when provided, otherwise it tries a dependency-ready `python3`, then falls back to `.venv/bin/python`.

To force a specific interpreter:

```
PYTHON=python3 ./stockdataanalysis.sh
```

To send mail from the clean entrypoint directly:

```
.venv/bin/python main.py -o shirong -e 500 -b 5 -t ELEC -m ./stocklist_elec
```

The shell script automatically loads SMTP and runtime variables from `.env` when present, or `.env.example` as a fallback. Real email sending still requires real `TWSE_SMTP_*` values.

Outputs are written under [`data`](data/), including daily TWSE CSV files, `shirong_analysis_dataset.csv`, and the ignored `data/mops/` raw/SQLite cache. The analysis dataset and email include LES v4.1 fields such as `LES Profile`, `Quality Gate`, all six component scores, RSI/MACD reversal evidence, `Falling Knife Veto`, balanced/strict triggers, standard/strict risk gates, `RewardRisk`, stop/target, `Position Fraction`, and suggested executable shares, plus the existing EREM v3 columns.

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
