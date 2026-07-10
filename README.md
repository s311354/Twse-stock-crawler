## Taiwan Stock Exchange (TWSE) Crawler ##

### Introduction ###

Taiwan Stock Exchange (TWSE) Crawler is a Python-based stock data collection and analysis tool.

The project automatically retrieves daily stock trading data from the Taiwan Stock Exchange (TWSE), including price, volume, transaction statistics, and valuation metrics such as PE ratio.

The collected data can be used for:
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
|   `-- stock_service.py            # workflow orchestration
|-- domain/
|   |-- models.py                   # domain aliases/models
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
|       `-- low_entry_score_v3.py   # advanced Low Entry Score v3 strategy
|-- infrastructure/
|   |-- crawler/twse_client.py      # TWSE HTTP boundary
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

The report combines `low_entry_score_v3` from [`domain/strategies/low_entry_score_v3.py`](domain/strategies/low_entry_score_v3.py) with `expected_return_exit_v3` from [`domain/strategies/expected_return_exit_v3.py`](domain/strategies/expected_return_exit_v3.py). The v1 and v2 EREM plugins remain registered for compatibility.

Low Entry Score v3 consumes TWSE daily OHLCV data and calculates:
- Trend score: `EMA20 > EMA50 > EMA200`, plus price above `EMA20`
- Momentum score: RSI14 neutral/rebound zone, plus MACD line above signal
- Volatility score: price near lower Bollinger band, plus ATR14 contraction below ATR20MA
- Volume score: volume above volume MA20, plus OBV rising
- Structure score: recent support holding, plus higher-low formation
- ATR14 risk control: `策略停損 = Close - 2*ATR14`, `策略停利 = Close + 3*ATR14`

Decision mapping:

- `BUY`: score >= 75
- `WATCH`: 60 <= score < 75
- `WAIT`: score < 60

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

### To Authenticate Email  ###

Authentication is supported, using the regular SMTP mechanism. When executing twse stock crawler, SMTP server generally requires authentication. The arguments are the email account and the password to authenticate with. Please set up **SMTP_USER** and **SMTP_PASSWORD** in [`.env.example`](./.env.example). This method will return normally of the authentication was successful. In addition, the receiver field combines the values (if any) of the To, Cc, and Bcc fields from msg. Please also specify the receivers in [`.env.example`](./.env.example)

### Usage ###

```
$  python main.py -h
usage: main.py [-h] [-t {VEH,ELEC,SEMI,AIR,BIO,COMM}] [-o {SHIRONG,shirong}] [-e ENDBACKTRACK] [-b BEGINBACKTRACK] [-s SUBJECT] [-cc CCRECEIVER]
               [-l] [-p PERIOD] [-m]
               stocklist [stocklist ...] holidays [holidays ...]

positional arguments:
  stocklist             Stock numbers directly, or legacy row indexes in a semicolon-separated stocklist file.
  holidays              Public holidays in Taiwan, comma separated or a holidays file.

options:
  -h, --help            show this help message and exit
  -t, --type {VEH,ELEC,SEMI,AIR,BIO,COMM}
                        The stock market you want to choose.
  -o, --output_file_names {SHIRONG,shirong}
                        Output file name prefix.
  -e, --endbacktrack ENDBACKTRACK
                        End of backtrack days.
  -b, --beginbacktrack BEGINBACKTRACK
                        Begin of backtrack days.
  -s, --subject SUBJECT
                        Email subject.
  -cc, --ccreceiver CCRECEIVER
                        Email CC receiver.
  -l, --linechart       Show a stock profit-ratio line chart.
  -p, --period PERIOD   Line-chart period.
  -m, --mail            Send email to recipients.
```

### Local Run Instructions ###

Create or refresh the local environment:

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the stock data analysis workflow:

```
./stockdataanalysis.sh
```

The script uses `$PYTHON` when provided, otherwise it tries a dependency-ready `python3`, then falls back to `.venv/bin/python`.

If you want the shell to load the sample SMTP/runtime variables first:

```
set -a
. ./.env.example
set +a
./stockdataanalysis.sh
```

To force a specific interpreter:

```
PYTHON=python3 ./stockdataanalysis.sh
```

Or call the clean entrypoint directly:

```
.venv/bin/python main.py -o shirong -e 35 -b 0 -t ELEC -m ./stocklist_elec ./holidays_2026
```

The shell script automatically loads SMTP and runtime variables from `.env` when present, or `.env.example` as a fallback. Real email sending still requires real `TWSE_SMTP_*` values.

Outputs are written under [`data`](data/), including daily TWSE CSV files and `shirong_analysis_dataset.csv`. The analysis dataset includes Low Entry v3 columns plus EREM v3 columns such as `EREM 分數`, `EREM Holding Confidence`, `EREM 決策訊號`, `EREM RewardRisk`, `EREM Relative Strength60`, `EREM Turnover Strength`, and `EREM VWAP60`.

Run tests:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q --assert=plain
```

### To Build the image with python package dependencies ###

```
docker build . --tag=user/twsestockcrawler:0.01 
```

### To Run and interact with above built image ###

We can use Docker CLI commands to directly manage bind mounts:

```
docker run -it  -v $(pwd)/${MOUNT}:/mnt user/twsestockcrawler:0.01  bash
```

### Potential Defect ###

The crawler now locks each tracked slot to a TWSE stock number before appending OHLC history. If a tracked stock is missing on a date, the row is skipped. Max-profit calculations still depend on the selected backtrack window and available valid trading days.

### Contacts Me ###

If you catch bugs, please <a href="mailto:s041978@hotmail.com">email</a> to me.

Happy Investing! :)
