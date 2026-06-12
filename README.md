## Taiwan Stock Exchange (TWSE) Crawler ##

### Introduction ###

Taiwan Stock Exchange Crawler is a crawler which directly crawl the data from [TWSE](https://www.twse.com.tw/exchangeReport/MI_INDE). The historical trading information of individual securities will be download in .csv format in [`twse`](twse/). The data can be used to analyze the trend of performance indicators and calculate the maximum profit we can achieve by choosing a single day to buy one stock and choosing a different day in the future to sell that stock, choosing on the same day immediately to sell that stock, completing at most three transactions, or completing as many transactions as we like, but we need to pay the transaction fee for each transaction. We could increase our probability of earning at taiwan stock market, leading us to earn passive profits.

### Data Description and The Trend of Performance Indicators ###

The crawled csv file inculde daily information: 'Stock Number', 'Stock Name', 'Stock Opening Price', 'Stock Price High', 'Stock Price Low' of individual securities in past few days.

![image](https://github.com/s311354/Twse-stock-crawler/blob/main/demo/data_description.jpg)

We can review the trend of performance indicators involves understanding how well the company's stock has performed over the past few days by looking at the trend of maximum profit we can achieve by completing as many transactions as we like, but we need to pay the transaction fee for each transaction.

![image](https://github.com/s311354/Twse-stock-crawler/blob/main/demo/trend_performance.png)

We also can review the table of maximum profit that describes the past performance and financial health of the stock.

![image](https://github.com/s311354/Twse-stock-crawler/blob/main/demo/max_profit_analysis_chart.png)

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
|   |-- services.py                 # pure profit/indicator services
|   `-- strategy.py                 # entry strategy decisions
|-- infrastructure/
|   |-- crawler/twse_client.py      # TWSE HTTP boundary
|   |-- storage/csv_repository.py   # CSV output boundary
|   |-- storage/chart_repository.py # chart output boundary
|   `-- notification/mail.py        # SMTP boundary
|-- config/
|   `-- settings.py                 # paths and runtime defaults
`-- data/                           # generated CSV outputs
```

Use [`main.py`](main.py) for the clean architecture entrypoint. [`stockanalysis.py`](stockanalysis.py) remains as a legacy-compatible module while the new CLI delegates orchestration to [`application/stock_service.py`](application/stock_service.py).

### To Authenticate Email  ###

Authentication is supported, using the regular SMTP mechanism. When executing twse stock crawler, SMTP server generally requires authentication. The arguments are the email account and the password to authenticate with. Please set up **SMTP_USER** and **SMTP_PASSWORD** in [`.env.example`](./.env.example). This method will return normally of the authentication was successful. In addition, the receiver field combines the values (if any) of the To, Cc, and Bcc fields from msg. Please also specify the receivers in [`.env.example`](./.env.example)

### Usage ###

```
$  python main.py -h
usage: main.py [-h] [-t {VEH,ELEC,SEMI,AIR,BIO,COMM}] [-o {SHIRONG,shirong}] [-e ENDBACKTRACK] [-b BEGINBACKTRACK] [-s SUBJECT] [-cc CCRECEIVER]
               [-l] [-p PERIOD] [-m]
               stocklist [stocklist ...] holidays [holidays ...]

positional arguments:
  stocklist             Stock indexes directly, or a semicolon-separated stocklist file.
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

### Quick Start  ###

```
$ ./stockanalysis.sh
```

![image](https://github.com/s311354/Twse-stock-crawler/blob/main/demo/output.gif)

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

Or call the clean entrypoint directly:

```
.venv/bin/python main.py -o shirong -e 35 -b 0 -t ELEC -m ./stocklist_elec ./holidays_2026
```

The shell script automatically loads SMTP and runtime variables from `.env` when present, or `.env.example` as a fallback.

Outputs are written under [`data`](data/), including daily TWSE CSV files and `shirong_analysis_dataset.csv`.

Run tests:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q
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

There's a potential risk of max profit happening in the order of TWSE data between different dates. This probably causes imprecise max profit and profit ratio.

### Contacts Me ###

If you catch bugs, please <a href="mailto:s041978@hotmail.com">email</a> to me.

Happy Investing! :)

## Reference ##

+ [TWSE](https://www.twse.com.tw/zh/index.html)
