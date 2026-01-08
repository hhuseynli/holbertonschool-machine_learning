#!/usr/bin/env python3
""" Convert into a numpy array """

import pandas as pd


def array(df):
    # Select last 10 rows
    df = df.tail(10)

    # Convert to ndarray
    arr = to_numpy(df)

    return arr
