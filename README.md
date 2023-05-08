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

### Files ###

```
.
|-- demo
|   |-- data_description.jpg
|   `-- trend_performance.png
|-- Dockerfile
|-- generatestocklist.sh
|-- holidays_2023
|-- __init__.py
|-- max_profit_analysis_chart.png
|-- README.md
|-- stockanalysis.py
|-- stockanalysis.sh
|-- stocklist_air
|-- stocklist_comm
|-- stocklist_elec
|-- textmewhenitsdone.py
|-- twse
|   |-- shirong_20221226.csv
|   |-- shirong_20221227.csv
|   |-- shirong_20230316.csv
...
|   `-- shirong_20230317.csv
`-- twstockcrawler.py
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

### To Authenticate Email  ###

Authentication is supported, using the regular SMTP mechanism. When executing twse stock crawler, SMTP server generally requires authentication. The arguments are the email account and the password to authenticate with. Please set up **your-email-account@gmail.com** and **your-password** in [`twstockcrawler.py`](./twstockcrawler.py). This method will return normally of the authentication was successful. In addition, the receiver field combines the values (if any) of the To, Cc, and Bcc fields from msg. Please also specify the receivers in [`twstockcrawler.py`](./twstockcrawler.py)

### Usage ###

```
$  python stockanalysis.py -h
usage: stockanalysis.py [-h] [-t {VEH,ELEC,SEMI,AIR,BIO,COMM}] [-o {SHIRONG,shirong}] [-e ENDBACKTRACK] [-b BEGINBACKTRACK] [-s SUBJECT] [-cc CCRECEIVER] [-l] [-p PERIOD] [-m]
                        stocklist [stocklist ...] holidays [holidays ...]

positional arguments:
  stocklist             If a single file format is passed in, then we assume it contains asemicolon-separated list of stock that we expect this script to stock list. If multiple stocks formats are
                        passed in, then we assume stocks are listed directly as arguments.
  holidays              Public holidays in Taiwan, comma separated.

optional arguments:
  -h, --help            show this help message and exit
  -t {VEH,ELEC,SEMI,AIR,BIO,COMM}, --type {VEH,ELEC,SEMI,AIR,BIO,COMM}
                        The stock market you want to choose.
  -o {SHIRONG,shirong}, --output_file_names {SHIRONG,shirong}
                        The owner you want to choose output file name.
  -e ENDBACKTRACK, --endbacktrack ENDBACKTRACK
                        The owner you want to choose the end of backtrack days.
  -b BEGINBACKTRACK, --beginbacktrack BEGINBACKTRACK
                        The owner you want to choose the begin of backtrack days.
  -s SUBJECT, --subject SUBJECT
                        The owner you want to set the email Subject.
  -cc CCRECEIVER, --ccreceiver CCRECEIVER
                        The owner you want to cc the email to someone apart from the recipient.
  -l, --linechart       The owner you want to show a trend of stock profit ratio over time.
  -p PERIOD, --period PERIOD
                        the owner you want to show the period of time on the line chart of stocks profit ratio.
  -m, --mail            the owner you want to send the email to recipients from your mail address.

```

### Quick Start  ###

```
$ ./stockanalysis.sh
```

![image](https://github.com/s311354/Twse-stock-crawler/blob/main/demo/output.gif)

### Potential Defect ###

There's a potential risk of max profit happening in the order of TWSE data between different dates. This probably causes imprecise max profit and profit ratio.

### Contacts Me ###

If you catch bugs, please <a href="mailto:s041978@hotmail.com">email</a> to me.

Happy Investing! :)

## Reference ##

+ [TWSE](https://www.twse.com.tw/zh/index.html)
