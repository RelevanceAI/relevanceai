import pytest

from typing import Dict, List, NamedTuple

from relevanceai import Client
from relevanceai.dataset_api import Dataset


@pytest.fixture(scope="session")
def test_df(test_client: Client, vector_dataset_id: str) -> Dataset:
    df = test_client.Dataset(vector_dataset_id)
    return df


@pytest.fixture(scope="session")
def test_nested_assorted_df(
    test_client: Client, assorted_nested_dataset: List[Dict]
) -> Dataset:
    df = test_client.Dataset(assorted_nested_dataset)
    return df


@pytest.fixture(scope="session")
def test_sample_obj_dataset_id_df(
    test_client: Client, obj_dataset_id: List[NamedTuple]
) -> Dataset:
    df = test_client.Dataset(obj_dataset_id)
    return df
