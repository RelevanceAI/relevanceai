"""
    Testing dataset read operations
"""

import pandas as pd

from relevanceai import Client

from relevanceai.dataset_api import Dataset


def test_Dataset_init(test_df: Dataset):
    assert True


def test_Dataset_json_encoder(test_client: Client, obj_dataset_id: str):
    test_df = test_client.Dataset(obj_dataset_id)
    assert "value1" in test_df.schema


def test_info(test_test_df: Dataset):
    test_test_df.info()
    assert True


def test_shape(test_df: Dataset):
    test_df.shape
    assert True


def test_head(test_df: Dataset):
    test_df.head()
    assert True


def test_describe(test_df: Dataset):
    test_df.describe()
    assert True


def test_sample(test_df: Dataset):
    sample_n = test_df.sample(n=10)
    assert len(sample_n) == 10


def test_value_counts(test_df: Dataset):
    test_df["sample_1_label"].value_counts()
    assert True


def test_value_counts_normalize(test_df: Dataset):
    test_df["sample_1_label"].value_counts(normalize=True)
    assert True


def test_filter(test_df: Dataset):
    test_df.filter(items=["sample_1_label"])
    test_df.filter(like="sample_1_label")
    test_df.filter(regex="s$")
    assert True


def test_read_df_check(test_read_df, vector_documents):
    assert test_read_df["inserted"] == len(vector_documents), "Did not insert properly"


def test_datasets_schema(test_df: Dataset):
    test_df.schema
    assert True


def test_info(test_df: Dataset):
    info = test_df.info()
    assert isinstance(info, pd.DataFrame)


def test_df_get_smoke(test_df: Dataset):
    assert test_df.get(["321", "3421"])
