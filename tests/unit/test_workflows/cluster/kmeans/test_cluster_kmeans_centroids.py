from relevanceai.client import Client
from relevanceai.dataset import Dataset
from relevanceai.operations.cluster import ClusterOps


def test_dataset_api_kmeans_centroids_properties(
    test_client: Client, test_dataset: Dataset
):

    alias: str = "test_alias"
    vector_field: str = "sample_1_vector_"

    from relevanceai.operations.cluster.models.kmeans import KMeansModel

    model = KMeansModel()

    clusterer: ClusterOps = test_client.ClusterOps(model=model, alias=alias)
    clusterer.fit(dataset=test_df, vector_fields=[vector_field])

    assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema

    # TODO: see why centroids fail
    centroids = clusterer.list_closest_to_center()
    assert len(centroids) > 0
