#!/usr/bin/env python3
""" Fill empty cells """


def fill(df):
    """ Input / Output """
    df = df.loc[:, "Timestamp": "Volume_(Currency)"]
    df["Close"] = df["Close"].fillna(method="ffill")
    df = df.fillna(value=dict.fromkeys(["High", "Low", "Open"], df["Close"]))
    columns = dict.fromkeys(["Volume_(BTC)", "Volume_(Currency)"], 0)
    df = df.fillna(value=columns)
    return df
