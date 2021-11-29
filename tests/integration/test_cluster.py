"""Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""
from relevanceai.vector_tools.cluster import KMeans

@pytest.mark.skip(reason="centroids/insert buggy")
def test_cluster_integration(
        test_client, test_sample_vector_dataset
    ):
    """Test for the entire clustering workflow.
    """
    # Retrieve a previous dataset 
    VECTOR_FIELD = "sample_1_vector_"
    docs = test_client.datasets.documents.list(test_sample_vector_dataset)

    # check if docs are inserted
    if len(docs['documents']) == 0:
        raise ValueError("Missing docs")
    cluster = KMeans(k=10)
    # Now when we want to fit the documents
    docs['documents'] = cluster.fit_documents(
        [VECTOR_FIELD],
        docs['documents']
    )
    test_client.update_documents(
        test_sample_vector_dataset,
        docs['documents']
    )

    # Centroids
    cluster_centers = cluster.get_centers()
    test_client.services.cluster.centroids.insert(
        test_sample_vector_dataset,
        vector_field=[VECTOR_FIELD],
        alias="kmeans_10",
        cluster_centers=cluster_centers
    )
    cluster_metadata = cluster.to_metadata()
    # Insert the centroids
    # test_client.services.cluster.centroids.insert(
    #     test_sample_vector_dataset, 
    #     vector_field="dostring_use_vector_",
    #     cluster_centers=cluster_centers,
    #     alias="kmeans_10")    
    assert True
