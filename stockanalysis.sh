#!/bin/sh
#
: $  python3 stockanalysis.py -h
: usage: stockanalysis.py [-h] [-t {VEH,ELEC,SEMI,AIR,BIO,COMM}] [-o {SHIRONG,shirong}] [-e ENDBACKTRACK] [-b BEGINBACKTRACK] [-s SUBJECT] [-cc CCRECEIVER] [-l] [-p PERIOD] [-m]
:                         stocklist [stocklist ...] holidays [holidays ...]
: 
: positional arguments:
:   stocklist             If a single file format is passed in, then we assume it contains asemicolon-separated list of stock that we expect this script to stock list. If multiple stocks formats are
:                         passed in, then we assume stocks are listed directly as arguments.
:   holidays              Public holidays in Taiwan, comma separated.
: 
: optional arguments:
:   -h, --help            show this help message and exit
:   -t {VEH,ELEC,SEMI,AIR,BIO,COMM}, --type {VEH,ELEC,SEMI,AIR,BIO,COMM}
:                         The stock market you want to choose.
:   -o {SHIRONG,shirong}, --output_file_names {SHIRONG,shirong}
:                         The owner you want to choose output file name.
:   -e ENDBACKTRACK, --endbacktrack ENDBACKTRACK
:                         The owner you want to choose the end of backtrack days.
:   -b BEGINBACKTRACK, --beginbacktrack BEGINBACKTRACK
:                         The owner you want to choose the begin of backtrack days.
:   -s SUBJECT, --subject SUBJECT
:                         The owner you want to set the email Subject.
:   -cc CCRECEIVER, --ccreceiver CCRECEIVER
:                         The owner you want to cc the email to someone apart from the recipient.
:   -l, --linechart       The owner you want to show a trend of stock profit ratio over time.
:   -p PERIOD, --period PERIOD
:                         the owner you want to show the period of time on the line chart of stocks profit ratio.
:   -m, --mail            the owner you want to send the email to recipients from your mail address.
#

python stockanalysis.py -t ELEC -o shirong -e 40 -b 0 -p 7 -l -m \
			-s I \
			./stocklist_elec_candidates ./holidays_2023


