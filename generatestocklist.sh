#!/bin/bash
#
:  --stocktype {VEH,ELEC,SEMI,AIR,BIO,COMM}
#

# Interface: User Command
if  [ $# != 3 ]; then
    echo "Usage: $0 <LIST_NUMBER_START> <LIST_NUMBER_END> <STOCK_TYPE>" 
    exit 1
fi

stocklist=stocklist_$(echo "$3" | tr '[:upper:]' '[:lower:]')

echo $stocklist

# Clean file
if [ -f "$stocklist" ]; then
    rm -rf "$stocklist"
else
    echo "No Such File"
fi

# Process to create list of stock ticker
for((i=$1;i<=$2;i=i+1))
do
    if [ $i == $2 ]; then
        stocknumbers+=$i
    else
        stocknumbers+=$i";"
    fi
done

echo ${stocknumbers[@]} >> $stocklist
