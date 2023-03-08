import pandas as pd


def clean_interactions(df: pd.DataFrame):
    df.date = pd.to_datetime(df.date)
    df.drop(
        inplace=True,
        columns=["u", "i"],
    )
    df = df.fillna({x: False for x in ["is_train", "is_val", "is_test"]})
    return df
