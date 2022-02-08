import pytest

from typing import Dict, List

from relevanceai import Client


@pytest.fixture(scope="session")
def test_df(test_client: Client, vector_dataset: List[Dict]):
    df = test_client.Dataset(vector_dataset)
    return df


@pytest.fixture(scope="session")
def test_nested_assorted_df(
    test_client: Client, sample_nested_assorted_dataset: List[Dict]
):
    df = test_client.Dataset(sample_nested_assorted_dataset)
    return df


@pytest.fixture(scope="session")
def test_sample_obj_dataset_df(test_client: Client, sample_obj_dataset):
    df = test_client.Dataset(sample_obj_dataset)
    return df
