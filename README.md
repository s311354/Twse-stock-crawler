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

The Python application uses ports and adapters: `interface` translates requests,
`application` coordinates workflows, `domain` owns model rules and contracts,
and `infrastructure` implements external services, storage and delivery.
`infrastructure/bootstrap.py` is the composition root; legacy facades remain
compatibility boundaries, not a second implementation of the models.

```text
.
|-- main.py                         # delegates to interface.cli.main()
|-- stockdataanalysis.sh            # profile/runtime validation and CLI wrapper
|-- pyproject.toml                  # versioned Python package and CLI metadata
|-- aistock_twse_integration/        # owner-research and v6 execution JSON adapters
|-- interface/
|   |-- cli.py                      # daily analysis and date-range replay adapter
|   |-- les_v6.py                   # offline unified/Hybrid/legacy comparison entry
|   |-- replay.py                   # strict-replay request/result entry
|   `-- research.py                 # evidence audit, jobs, archives and reproduction
|-- application/
|   |-- stock_service.py            # market -> fundamentals -> strategies -> report -> notification
|   |-- use_cases.py                # port-based workflow steps
|   |-- analysis_dataset.py         # daily entry/EREM composition and typed decision assembly
|   |-- strategy_execution.py       # prepare shared features and invoke plugins
|   |-- daily_entry.py              # compose v6 entry with separately owned valuation fields
|   |-- daily_valuation.py          # v5 fundamental/valuation sidecar, not v5 entry scoring
|   |-- low_entry_v6_daily.py        # explicit deployed Hybrid config and report projection
|   |-- evidence_context.py         # work-scoped read-only financial evidence packets
|   |-- daily_work.py               # bounded, version-guarded preprocessing caches
|   |-- fundamental_service.py      # MOPS discovery/cache/derivation orchestration
|   |-- filing_catalog_refresh.py   # cache-first filing refresh planner
|   |-- portfolio_allocation_v2.py  # aggregate PAE v2 allocation pipeline
|   |-- strategy_contracts.py       # schema validation and CSV projection
|   |-- replay_contracts.py         # cutoff, account, identity and version contracts
|   |-- replay_inputs.py            # select cutoff-valid inputs from pinned evidence
|   |-- replay_service.py           # shared StrictReplayEngine, no fills or notifications
|   `-- research_audit.py           # coverage and business-result comparison
|-- domain/
|   |-- fundamental_derivation.py   # point-in-time financial derivation
|   |-- model_contracts.py           # immutable versioned decision evidence
|   |-- features.py                 # shared immutable technical features and price-scale pairs
|   |-- numeric_history.py          # metadata-free numeric history boundary
|   |-- strategy.py                 # strategy abstraction/plugin registry
|   `-- strategies/
|       |-- low_entry_score_v6.py   # v6 plugins and next-session execution validation
|       |-- low_entry_v6_hybrid.py  # context selection and one six-factor 100-point ledger
|       |-- low_entry_v6_decision.py # explicit penalties, safety caps and final-score decision
|       |-- low_entry_v6_quality.py # unified QualityResult and five-dimension scoring
|       |-- low_entry_v6_quality_evidence.py # field/source/version verification
|       |-- low_entry_v6_report.py  # shared native report/gate validation
|       |-- low_entry_score_v5.py   # retained v5 plugin and valuation compatibility
|       |-- expected_return_exit_v3.py # EREM holding/exit model
|       `-- portfolio_allocation_v2.py # PAE capital-allocation model
|-- infrastructure/
|   |-- bootstrap.py                # dependency wiring for the daily application
|   |-- legacy_crawler_facade.py    # retained crawler API compatibility
|   |-- crawler/                    # TWSE/TPEx market data and trading restrictions
|   |-- calendar/                   # annual schedules, observed versions and emergency evidence
|   |-- fundamentals/
|   |   |-- mops_client.py          # filing/announcement/correction discovery
|   |   |-- ixbrl_parser.py         # normalized Inline XBRL facts
|   |   |-- mops_repository.py      # SQLite metadata + raw revision cache
|   |   `-- twse_financial_batch.py # official batch financial data boundary
|   |-- runtime/
|   |   |-- market_data_adapter.py  # market-data port adapter
|   |   |-- fundamental_adapter.py  # thin fundamental port adapter
|   |   |-- fundamental_refresh.py  # cache-first refresh coordinator
|   |   |-- strategy_adapter.py     # LES/EREM/PAE execution adapter
|   |   `-- output_adapters.py      # report, SMTP, and chart adapters
|   |-- daily_parallel.py           # optional spawn-based per-stock computation
|   |-- daily_performance.py        # staged timings/counters and durable performance receipt
|   |-- replay_runtime.py           # shared engine/version wiring
|   |-- replay_evidence.py          # immutable evidence loading and integrity verification
|   |-- research_job.py             # pinned work context, daily retention and recovery
|   |-- research_objects.py         # content-addressed evidence/source/environment objects
|   |-- research_archive.py         # immutable archive and reproduction support
|   |-- report/                     # HTML renderer factory and v6 presentation
|   |-- storage/                    # market/fundamental caches, report evidence, PAE/replay state
|   `-- notification/mail.py        # SMTP boundary, separate from research execution
|-- config/
|   `-- settings.py                 # paths and runtime defaults
|-- tests/                          # model, evidence, replay, compatibility and workflow tests
`-- data/                           # market caches, report/evidence artifacts and runtime state
```

The tree highlights active boundaries, not every file. Earlier LES, EREM and
PAE plugins remain available through the registry for explicit compatibility use.

**Daily workflow**

```text
stockdataanalysis.sh -> main.py -> interface.cli -> infrastructure.bootstrap
  -> StockAnalysisService
     -> market data + official calendar/restriction evidence
     -> cache-first financial refresh + point-in-time source selection
     -> EvidenceContext + shared unit-safe technical features
     -> DailyEntryService: LES v6 Hybrid + separate valuation sidecar
     -> EREM -> typed decisions -> PAE portfolio allocation
     -> evidence-backed CSV/JSON + HTML -> optional notification
```

The deployed v6 Hybrid uses decision policy **4**: context is selected before
scoring, six factor budgets are **30/10/20/25/5/10**, and unrounded Final maps
to **BUY >=72** or **BUY_SMALL >=62**. Quality and non-compensable safety checks
remain separate from technical points. Policy 3 (**75/65**) is explicitly
retained as `low_entry_score_v6_hybrid_legacy`; the unified comparison plugin
`low_entry_score_v6` is not silently upgraded.

**Historical research and retention**

`interface.replay` and `interface.research` reuse `StrictReplayEngine` and the
native daily model builder. A `ResearchJob` pins and verifies one immutable
dataset and model/code identity, selects cutoff-valid evidence for each day,
and shares that selection with coverage auditing. Explicit account inputs and
isolated PAE state are required; bounded preprocessing caches do not share
account-dependent EREM/PAE results across accounts. Content-addressed objects,
incremental research state and completion receipts support retention/recovery.
These entries do not send mail or simulate fills. BankEmulator owns fills,
fees and account-ledger evolution; the crawler owns evidence and research.

### Low-Entry Model v6 (daily entry pipeline) ###

Daily analysis explicitly uses the `low_entry_score_v6_hybrid` Strategy plugin for
quality-first reversal, trend-pullback and leadership-continuation interpretation
with `BUY` / `BUY_SMALL` signals. Daily analysis,
typed entry fields, HTML and email subjects now use v6. The v5 valuation/CER
sidecar and EREM/PAE policies remain separately identified, and old plugins are retained.

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
- `REDUCE 25%-50%`: 40 <= Holding Confidence < 55. The recommended reduction
  increases in 5% steps as Holding Confidence falls: HC 54 maps to 25%, while
  HC 40 maps to 50%.
- `EXIT`: Holding Confidence < 40

### Portfolio Allocation Engine v2 ###

Portfolio Allocation Engine v2 (`portfolio_allocation_v2`, schema `pae.portfolio_decision.v2`) is the daily production allocator. It consumes published LES v6 and EREM v3 outputs, with the existing valuation/CER sidecar, and adds opportunity-cost allocation, complete-portfolio improvement, explicit cash preference, and atomic four-seat replacement rules.

The daily premarket entry universe accepts only hard-gate-complete LES v6 `BUY` and `BUY_SMALL` decisions. Its setup-specific thresholds, position cap, and execution plan come from the frozen v6 report, not a v5 threshold. PAE remains an independent allocation gate and may decline an LES BUY. A high LES score never overrides missing core quality, failed Risk, Falling-Knife, liquidity, or point-in-time evidence. Legacy `EARLY_BUY` remains readable for old holdings but is not produced by v6.

| Effective holdings | Mode | Authority |
|---:|---|---|
| 0-2 | `DIRECT_SELECTION` | `NON_BINDING`; LES controls entries and EREM controls holdings; CER diagnostics cannot veto a hard-gate-complete LES entry |
| 3 | `PAE_ACTIVE_OPEN_SLOT` | `BINDING`; the fourth slot must pass absolute gates and Portfolio Improvement |
| 4 | `PAE_ACTIVE_FULL` | `BINDING`; a new symbol requires Dual Dominance, Switch Hurdle, Persistence/Fast Path, and atomic seat release |
| >4 | `POLICY_BREACH_REDUCE_ONLY` | No new name or net equity increase; hard exits retain priority |

PAE v2 fails closed for new capital when Quality, Valuation, CER, LES, Risk, Falling-Knife, liquidity, position-cap, or point-in-time evidence is missing. Missing EREM evidence alone produces `HOLD_DATA_INSUFFICIENT`, never an automatic exit. Cash remains a valid output and four holdings are a hard cap, not a target.

At four holdings, a replacement is one atomic `EXIT_THEN_ENTER` plan: the incumbent must exit completely before the challenger can enter. A failed exit cancels entry; a failed entry leaves the account in cash. The crawler publishes this bounded plan but does not upgrade frozen LES decisions or substitute a Top-3 fallback.

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
usage: main.py [-h] [--list-strategies] [--check-config] [-t {VEH,ELEC,BUI,AIR,CHEMI,BIO,SEMI,COMP,COMM,ELECCOMP}] [-o {SHIRONG,shirong}] [-e ENDBACKTRACK] [-b BEGINBACKTRACK] [-s SUBJECT] [-cc CCRECEIVER]
               [-l] [-p PERIOD] [-m] [--skip-mops]
               [--mops-history-years MOPS_HISTORY_YEARS] [--account-equity ACCOUNT_EQUITY]
               [--pae-holdings-file PAE_HOLDINGS_FILE] [--include-symbols-file INCLUDE_SYMBOLS_FILE]
               [--pae-current-cash PAE_CURRENT_CASH]
               [--mode {release,debug}]
               [stocklist ...]

positional arguments:
  stocklist             Stock numbers directly, or legacy row indexes in a semicolon-separated stocklist file.

options:
  -h, --help            show this help message and exit
  --list-strategies     List all available trading strategies with descriptions and exit.
  --check-config        Validate settings and inputs, print a safe JSON summary, and exit without network or email.
  -t, --type {VEH,ELEC,BUI,AIR,CHEMI,BIO,SEMI,COMP,COMM,ELECCOMP}
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
  --include-symbols-file INCLUDE_SYMBOLS_FILE
                        Add symbol-only rows to a shared market analysis universe without marking them as PAE holdings.
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

Run the automated workflow after configuring `.env.example`:

```sh
./stockdataanalysis.sh --profile semi
```

### Contact ###

Found a bug? [Send an email](mailto:s041978@hotmail.com).

Happy Investing! :)
