import pandas as pd


def clean_interactions(df: pd.DataFrame):
    df.date = pd.to_datetime(df.date)
    return df
