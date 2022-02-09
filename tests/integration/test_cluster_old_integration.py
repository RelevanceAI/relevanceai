"""
Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""

import pytest

from relevanceai import Client
from relevanceai.dataset_api import Dataset

from relevanceai.clusterer import ClusterOps
from relevanceai.clusterer import CentroidClusterBase


def test_old_cluster_integration(test_client: Client, vector_dataset_id):
    """Test for the entire clustering workflow."""
    # Retrieve a previous dataset
    VECTOR_FIELD = "sample_1_vector_"
    ALIAS = "kmeans_10"
    documents = test_client.datasets.documents.get_where(vector_dataset_id)

    from relevanceai.vector_tools.cluster import KMeans

    # check if documents are inserted
    if len(documents["documents"]) == 0:
        raise ValueError("Missing documents")
    cluster = KMeans(k=10)
    # Now when we want to fit the documents
    documents["documents"] = cluster.fit_documents(
        [VECTOR_FIELD], documents["documents"]
    )

    # Centroids
    cluster_centers = cluster.get_centroid_documents()
    test_client.services.cluster.centroids.insert(
        vector_dataset_id,
        vector_fields=[VECTOR_FIELD],
        alias=ALIAS,
        cluster_centers=cluster_centers,
    )
    cluster_metadata = cluster.to_metadata()
    # Insert the centroids
    # test_client.services.cluster.centroids.metadata(
    #     vector_dataset_id,
    #     vector_field=VECTOR_FIELD,
    #     cluster_centers=cluster_centers,
    #     alias=ALIAS)
    assert True
