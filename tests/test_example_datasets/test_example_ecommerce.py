import pytest


def test_get_ecommerce_dataset_subset():
    from relevanceai.datasets import get_ecommerce_3_dataset

    assert len(get_ecommerce_3_dataset(number_of_documents=1000)) == 1000
