#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import smtplib
from typing import List
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


GMAIL = "gmail.com"
HMAIL = "hotmail.com"

StockhtmlMessage = """\
<html>
    <head>
        <style>
        table, th, td {{ border: 1px solid black; border-collapse: collapse; }}
        th, td {{ padding: 5px; }}
        </style>
    </head>

    <body>
        <p>
        Hey, Thank you for your waiting! The estimated max profit within {days} days ({begin} ~ {end}) is done. We are continuing with our effort in processing it and will update the latest information to you soon. :) <br>
        <br>

        ## Here is the estimated max profit: ($NTD)<br>
        {stocktable} <br>
        <br>
        </p>

        <br>
        **Note** <br>
        - I am not a professional investor and also not responsible for your losses. This is just the edcational content. <br>
        - There's a potential risk of max profit happening in the order of TWSE data between different dates. This probably causes imprecise max profit and profit ratio. <br>

        <br>
        Regards, <br>
        Shi-rong (Louis) Liu <br>
        https://louissrliu.github.io/ <br>
    </body>
</html>
"""

StockhtmlImgMessage = """\
        Hey, Thank you for your waiting! The estimated trend performance indicators back track until {date} is done. We are continuing with our effort in processing it and will update the latest information to you soon. :) 

        Please find the trend of performance indicators attached.

        **Note**
        - I am not a professional investor and also not responsible for your losses. This is just the edcational content.
        - There's a potential risk of max profit happening in the order of TWSE data between different dates. This probably causes imprecise max profit and profit ratio.

        Regards,
        Shi-rong (Louis) Liu
"""


class TextMeWhenItsDone(object):
    """
    A :class:~practice_common_algorithm.tool.TextMeWhenItsDone object is the sending an email module

    The smtplib module defines an SMTP client session object that can be used to send mail to any internet machine with an SMTP or ESMTP listener daemon
    """

    def __init__(self, email: str):
        # Connect to the SMTP server
        if email[email.index("@")+1:] == GMAIL:
            self.server = smtplib.SMTP("smtp.gmail.com", 587)
        elif email[email.index("@")+1:] == HMAIL:
            self.server = smtplib.SMTP("smtp.live.com", 587)


    def __del__(self):
        # Disconnect from the SMTP server
        self.server.quit()


    def login(self, email: str, password: str) -> None:
        # Log in to the email account
        self.server.starttls()
        self.server.login(email, password)


    def imgstockprofittableme(self, subject: str, email: str, password: str, backtrack: str, stocktype: int, receiver= "YO", ccreceiver = "") -> None:
        # Create the email message
        msg = MIMEMultipart()
        msg['Subject'] = "[TWSE Stock {}] [{}]: The Trend of Performance Indicators (back track until {})".format(subject, stocktype, backtrack)

        # RFC 5322
        msg['From']  = email
        msg['Bcc']  = ccreceiver

        # Record the MIME type of both parts - text/plain and text/html
        part2 = MIMEText(StockhtmlImgMessage.format(date = backtrack), 'plain')
        msg.attach(part2)

        # Attach the image file
        with open('./max_profit_analysis_chart.png', 'rb') as file:
            img = MIMEImage(file.read())
            img.add_header('Content-Disposition', 'attachment', filename = 'max_profit_analysis_chart.png')
            msg.attach(img)

        # Send the email
        self.server.sendmail(from_addr = email, to_addrs = receiver, msg = msg.as_string())


    def textstockprofittableme(self, subject: str, email: str, password: str, iso_scheduled_times: List[str], transactiondays: int, stocktype: int, stockprofittable, receiver="YO", ccreceiver = "") -> None:
        # Create the email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "[TWSE Stock {}] [{}]: Estimated Max Profit ({} ~ {})".format(subject, stocktype, iso_scheduled_times[0], iso_scheduled_times[-1])

        # RFC 5322
        msg['From']  = email
        msg['Bcc']    = ccreceiver

        # Record the MIME type of both parts - text/plain and text/html
        part2 = MIMEText(StockhtmlMessage.format(begin = iso_scheduled_times[0],
                                                 end   = iso_scheduled_times[-1],
                                                 days  = transactiondays,
                                                 stocktable = stockprofittable), 'html')

        # The HTML message, is best and preferred
        msg.attach(part2)
        self.server.sendmail(from_addr = email, to_addrs = receiver, msg = msg.as_string())
