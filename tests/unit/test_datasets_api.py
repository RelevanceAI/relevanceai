import pandas as pd
from relevanceai.http_client import Dataset
from relevanceai import Client


def test_read_df_check(test_read_df, sample_vector_documents):
    assert test_read_df["inserted"] == len(
        sample_vector_documents
    ), "Did not insert properly"


def test_datasets_api(test_dataset_df: Dataset):
    """Testing the datasets API
    Simple smoke tests for now until we are happy with functionality :)
    """
    test_dataset_df.info()
    test_dataset_df.describe()
    test_dataset_df.head()
    test_dataset_df.schema()
    assert True


def test_apply(test_dataset_df: Dataset):
    RANDOM_STRING = "you are the kingj"
    test_dataset_df["sample_1_label"].apply(
        lambda x: x + RANDOM_STRING, output_field="sample_1_label_2"
    )
    assert test_dataset_df["sample_1_label_2"][0].endswith(RANDOM_STRING)


def test_info(test_dataset_df: Dataset):
    info = test_dataset_df.info()
    assert isinstance(info, pd.DataFrame)
