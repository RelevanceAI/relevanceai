import pytest

from typing import Dict, List, NamedTuple

from relevanceai import Client
from relevanceai.interfaces import Dataset


@pytest.fixture(scope="function")
def test_df(test_client: Client, vector_dataset_id: str) -> Dataset:
    df = test_client.Dataset(vector_dataset_id)
    yield df
    # df.delete()


@pytest.fixture(scope="function")
def test_nested_assorted_df(
    test_client: Client, assorted_nested_dataset: str
) -> Dataset:
    df = test_client.Dataset(assorted_nested_dataset)
    return df


@pytest.fixture(scope="function")
def test_sample_obj_dataset_id_df(test_client: Client, obj_dataset_id: str) -> Dataset:
    df = test_client.Dataset(obj_dataset_id)
    return df


@pytest.fixture(scope="function")
def test_clustered_df(test_client: Client, clustered_dataset_id: str) -> Dataset:
    df = test_client.Dataset(clustered_dataset_id)
    return df
