#!/usr/bin/env python3
""" This script modifies a DataFrame """
import pandas as pd


def rename(df):
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit=s)
    return df[["Datetime", "Close"]]
