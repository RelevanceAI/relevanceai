"""Test DR on cluster
"""
from relevanceai import Client

# def test_dr_on_cluster():
#     # Run dim reduction d
#     from relevanceai.package_utils.datasets import mock_documents

#     docs = mock_documents()
#     # Reduce dimensions

#     ds = test_client.Dataset("sample456")
#     ds.insert_documents(docs)
#     ds.auto_reduce_dimensions("pca-3", vector_fields=["sample_1_vector_"])
#     ds.auto_cluster("kmeans-2", vector_fields=["_dr_.pca-3.sample_1_vector_"])
#     ds.delete()
#     assert True
