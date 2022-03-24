import pandas as pd

from relevanceai.dataset import Dataset


def test_to_pandas_dataframe(test_dataset: Dataset):
    df = test_dataset.to_pandas_dataframe()
    assert type(df) == pd.DataFrame
