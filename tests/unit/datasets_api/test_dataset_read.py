"""
    Testing dataset read operations
"""

import pandas as pd
from relevanceai.http_client import Dataset, Client


def test_Dataset_init(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    assert True


def test_info(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    info = df.info()
    assert True


def test_shape(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    shape = df.shape
    assert True


def test_head(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    head = df.head()
    assert True


def test_describe(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    description = df.describe()
    assert True


def test_sample(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    sample_n = df.sample(n=10)
    assert len(sample_n) == 10


def test_value_counts(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    value_counts = df["sample_1_label"].value_counts()
    value_counts = df["sample_1_label"].value_counts(normalize=True)
    assert True


def test_filter(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    items = df.filter(items=["sample_1_label"])
    like = df.filter(like="sample_1_label")
    regex = df.filter(regex="s$")
    assert True


def test_read_df_check(test_read_df, sample_vector_documents):
    assert test_read_df["inserted"] == len(
        sample_vector_documents
    ), "Did not insert properly"


def test_datasets_schema(test_dataset_df: Dataset):
    """Testing the datasets API
    Simple smoke tests for now until we are happy with functionality :)
    """
    test_dataset_df.schema
    assert True


def test_info(test_dataset_df: Dataset):
    info = test_dataset_df.info()
    assert isinstance(info, pd.DataFrame)


def test_df_get_smoke(test_dataset_df: Dataset):
    """Test the df"""
    # This is to cover the 255 error before
    assert test_dataset_df.get(["321", "3421"])
