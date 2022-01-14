import pandas as pd
from relevanceai.http_client import Dataset


def test_datasets_api(test_dataset_df: Dataset):
    """Testing the datasets API
    Simple smoke tests for now until we are happy with functionality :)
    """
    test_dataset_df.info()
    test_dataset_df.describe()
    test_dataset_df.head()
    assert True


def test_apply(test_dataset_df: Dataset):
    RANDOM_STRING = "you are the kingj"
    test_dataset_df["sample_1_label"].apply(lambda x: x + RANDOM_STRING)
    assert test_dataset_df["sample_1_label"][0].endswith(RANDOM_STRING)


def test_info(test_dataset_df: Dataset):
    info = test_dataset_df.info()
    assert isinstance(info, pd.DataFrame)
