import pandas as pd

from relevanceai.dataset import Dataset


def test_to_pandas_dataframe(test_df: Dataset):
    df = test_df.to_pandas_dataframe()
    assert type(df) == pd.DataFrame
