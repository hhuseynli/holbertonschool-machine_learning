#!/usr/bin/env python3
""" Slice a pandas DataFrame """


def slice(df):
    """ Slice the required rows """

    df = df[["High", "Low", "Close", "Volume_BTC"]].iloc[::60]
    return df
