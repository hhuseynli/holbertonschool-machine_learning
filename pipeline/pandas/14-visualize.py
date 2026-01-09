#!/usr/bin/env python3
""" Visualize data """
import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')\

df = df.drop(columns="Weighted_Price")

df = df.rename({"Timestamp": "Date"})

df["Date"] = pd.to_datetime(df["Date"], unit=s]
df = df.set_index("Date")

df["Close"] = df["Close"].fillna(method="ffill")

cols = dict.fromkeys(["High", "Low", "Open"], df["Close"])
df = df.fillna(value=cols)

cols = dict.fromkeys(["Volume_(BTC)", "Volume_(Currency)"], 0)
df = df.fillna(value=cols)

df = df[df.index >= '2017-01-01']

daily_df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Print transformed DataFrame before plotting
print(daily_df)

daily_df[['Open', 'High', 'Low', 'Close']].plot(figsize=(12, 6))
plt.title('BTC Price (Daily, 2017+)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()
