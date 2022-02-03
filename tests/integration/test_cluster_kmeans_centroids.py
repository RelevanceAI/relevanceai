from relevanceai.http_client import Client, Dataset, ClusterOps


def test_dataset_api_kmeans_centroids_properties(
    test_client: Client, test_dataset_df: Dataset
):
    from sklearn.cluster import KMeans

    vector_field = "sample_1_vector_"
    alias = "test_alias"
    from relevanceai.clusterer import KMeansModel

    model = KMeansModel()

    clusterer: ClusterOps = test_client.ClusterOps(model=model, alias=alias)
    clusterer.fit_predict_update(dataset=test_dataset_df, vector_fields=[vector_field])

    assert f"_cluster_.{vector_field}.{alias}" in test_dataset_df.schema

    centroids = clusterer.centroids
    assert len(centroids) > 0
