"""Test DR on cluster
"""
from relevanceai import Client


def test_dr_on_cluster(test_client: Client):
    # Run dim reduction d
    from relevanceai.datasets import mock_documents

    docs = mock_documents()
    # Reduce dimensions

    ds = test_client.Dataset("sample")
    ds.insert_documents(docs)
    ds.auto_reduce_dimensions("pca-3", vector_fields=["sample_1_vector_"])
    assert True
