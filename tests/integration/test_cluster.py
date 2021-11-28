"""Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""
from relevanceai.visualise.cluster import KMeans

def test_cluster_integration(
        test_client, test_sample_vector_dataset
    ):
    """Test for the entire clustering workflow.
    """
    # Retrieve a previous dataset 
    docs = test_client.datasets.documents.list(test_sample_vector_dataset)
    cluster = KMeans(k=10)
    # Now when we want to fit the documents
    docs['documents'] = cluster.fit_documents(
        ["sample_1_vector_"], 
        docs['documents']
    )
    test_client.update_documents(
        test_sample_vector_dataset,
        docs['documents']
    )

    # Centroids
    cluster_centers = cluster.get_centroid_docs()
    test_client.services.cluster.centroids.insert(
        test_sample_vector_dataset,
        alias="kmeans_10",
        cluster_centers=cluster_centers
    )
    cluster_metadata = cluster.get_metadata()
    assert True
