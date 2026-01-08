#!/usr/bin/env python3
""" Convert into a numpy array """


def array(df):
    """
    Input: Dataframe

    Output: ndarray
    """
    # Select last 10 rows
    df = df[["High", "Close"]].tail(10)

    # Convert to ndarray
    arr = df.to_numpy()

    return arr
