import pytest


@pytest.mark.skip(reason="Run this to test")
def test_new_config(test_client):
    from relevanceai.utils.datasets import get_ecommerce_dataset_clean
    from relevanceai import Client

    datasets = test_client.list_datasets()
    print(datasets["datasets"][0:2])

    dataset = get_ecommerce_dataset_clean()

    print(len(dataset))
    dataset[0].keys()
    datasets = test_client.list_datasets()
    assert True
