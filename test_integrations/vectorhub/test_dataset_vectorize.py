from unittest import result
from relevanceai.datasets import mock_documents
from relevanceai.http_client import Client


def test_dataset_vectorize(test_dataset):
    dataset_id = "vectorhub-test"
    results = test_dataset.vectorize(text_fields=["sample_3_description"])
    # This means teh results are good yay!
    assert results is None
    assert "sample_3_description_use_vector_" in test_dataset.schema
