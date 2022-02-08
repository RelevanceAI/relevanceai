import pytest

from relevanceai import Client


@pytest.fixture(scope="session")
def test_dataset_df(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    return df


@pytest.fixture(scope="session")
def test_sample_nested_assorted_dataset_df(
    test_client: Client, sample_nested_assorted_dataset
):
    df = test_client.Dataset(sample_nested_assorted_dataset)
    return df


@pytest.fixture(scope="session")
def test_sample_nested_assorted_dataset_df(
    test_client: Client, sample_nested_assorted_dataset
):
    df = test_client.Dataset(sample_nested_assorted_dataset)
    return df


@pytest.fixture(scope="session")
def test_sample_obj_dataset_df(test_client: Client, sample_obj_dataset):
    df = test_client.Dataset(sample_obj_dataset)
    return df
