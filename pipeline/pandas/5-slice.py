#!/usr/bin/env python3
""" Slice a pandas DataFrame """


def slice(df):
    """ Slice the required rows """

    df = df.loc[::60, ["High", "Low", "Close", "Value_BTC"]]
    return df
