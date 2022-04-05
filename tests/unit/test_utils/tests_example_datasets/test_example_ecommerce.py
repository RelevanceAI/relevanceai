import pytest


# @pytest.mark.skip(reason="bad API_KEY on old aus east")
def test_get_ecommerce_dataset_subset():
    from relevanceai.utils.datasets import get_ecommerce_3_dataset

    assert len(get_ecommerce_3_dataset(number_of_documents=1000)) == 1000
