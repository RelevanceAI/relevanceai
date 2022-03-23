import pytest


def test_get_sample_ecommerce_dataset():
    from relevanceai.utils.datasets import get_ecommerce_1_dataset

    assert len(get_ecommerce_1_dataset(number_of_documents=100)) == 100
