"""
    Testing dataset read operations
"""

import pandas as pd
import time

from relevanceai import Client

from relevanceai.dataset import Dataset


def test_Dataset_init(test_dataset: Dataset):
    assert True


def test_Dataset_json_encoder(test_client: Client, obj_dataset_id: str):
    test_dataset = test_client.Dataset(obj_dataset_id)
    assert "value1" in test_dataset.schema


def test_info(test_dataset: Dataset):
    test_dataset.info()
    assert True


def test_shape(test_dataset: Dataset):
    test_dataset.shape
    assert True


def test_head(test_dataset: Dataset):
    test_dataset.head()
    assert True


def test_describe(test_dataset: Dataset):
    test_dataset.describe()
    assert True


def test_sample(test_dataset: Dataset):
    sample_n = test_dataset.sample(n=10)
    assert len(sample_n) == 10


def test_value_counts(test_dataset: Dataset):
    test_dataset["sample_1_label"].value_counts()
    assert True


def test_value_counts_normalize(test_dataset: Dataset):
    test_dataset["sample_1_label"].value_counts(normalize=True)
    assert True


def test_filter(test_dataset: Dataset):
    test_dataset.filter(items=["sample_1_label"])
    test_dataset.filter(like="sample_1_label")
    test_dataset.filter(regex="s$")
    assert True


def test_read_df_check(test_read_df, vector_documents):
    assert test_read_df is None, "Did not insert properly"


def test_datasets_schema(test_dataset: Dataset):
    test_dataset.schema
    assert True


def test_info(test_dataset: Dataset):
    info = test_dataset.info()
    assert isinstance(info, pd.DataFrame)


def test_df_get_smoke(test_dataset: Dataset):
    assert test_dataset.get(["321", "3421"])


def test_df_metadata(test_dataset: Dataset):
    metadata = {"value": "hey"}
    time.sleep(1)
    test_dataset.insert_metadata(metadata)
    time.sleep(1)
    new_metadata = test_dataset.metadata
    assert new_metadata["value"] == "hey"

    new_metadata = {"value": "cool", "old_value": "hey"}
    response = test_dataset.upsert_metadata(new_metadata)
    new_metadata = test_dataset.metadata
    assert new_metadata["value"] == "cool"
    assert new_metadata["old_value"] == "hey"
