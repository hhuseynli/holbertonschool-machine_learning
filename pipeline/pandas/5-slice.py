#!/usr/bin/env python3
"""60th rows"""


def slice(df):
    """60th rows"""
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
