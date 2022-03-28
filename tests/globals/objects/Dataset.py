import pytest

from relevanceai.client import Client
from relevanceai.dataset import Dataset


@pytest.fixture(scope="function")
def test_dataset(
    test_client: Client,
    vector_dataset_id: str,
) -> Dataset:
    dataset = test_client.Dataset(vector_dataset_id)
    return dataset


@pytest.fixture(scope="function")
def test_nested_assorted_df(
    test_client: Client,
    assorted_nested_dataset: str,
) -> Dataset:
    dataset = test_client.Dataset(assorted_nested_dataset)
    return dataset


@pytest.fixture(scope="function")
def test_sample_obj_dataset_id_df(
    test_client: Client,
    obj_dataset_id: str,
) -> Dataset:
    dataset = test_client.Dataset(obj_dataset_id)
    return dataset


@pytest.fixture(scope="function")
def test_clustered_df(
    test_client: Client,
    clustered_dataset_id: str,
) -> Dataset:
    dataset = test_client.Dataset(clustered_dataset_id)
    return dataset
