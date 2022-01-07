"""Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""
from relevanceai.vector_tools.cluster import KMeans
import pytest


def test_cluster_integration(test_client, test_sample_vector_dataset):
    """Test for the entire clustering workflow."""
    # Retrieve a previous dataset
    VECTOR_FIELD = "sample_1_vector_"
    ALIAS = "kmeans_10"
    docs = test_client.datasets.documents.list(test_sample_vector_dataset)

    # check if docs are inserted
    if len(docs["documents"]) == 0:
        raise ValueError("Missing docs")
    cluster = KMeans(k=10)
    # Now when we want to fit the documents
    docs["documents"] = cluster.fit_documents([VECTOR_FIELD], docs["documents"])

    # Centroids
    cluster_centers = cluster.get_centroid_docs()
    test_client.services.cluster.centroids.insert(
        test_sample_vector_dataset,
        vector_fields=[VECTOR_FIELD],
        alias=ALIAS,
        cluster_centers=cluster_centers,
    )
    cluster_metadata = cluster.to_metadata()
    # Insert the centroids
    # test_client.services.cluster.centroids.metadata(
    #     test_sample_vector_dataset,
    #     vector_field=VECTOR_FIELD,
    #     cluster_centers=cluster_centers,
    #     alias=ALIAS)
    assert True

def test_cluster_integration_multiple_vector_fields(test_client, test_sample_vector_dataset):
    """Test for the entire clustering workflow."""
    # Retrieve a previous dataset
    VECTOR_FIELDS = "sample_1_vector_"
    ALIAS = "kmeans_10"
    docs = test_client.datasets.documents.list(test_sample_vector_dataset)
    
    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=test_sample_vector_dataset,
        vector_fields=VECTOR_FIELDS
    )
    assert True
