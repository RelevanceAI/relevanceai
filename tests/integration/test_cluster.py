"""
Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""

import pytest

from relevanceai import Client
from relevanceai.dataset_api import Dataset

from relevanceai.clusterer import Clusterer
from relevanceai.clusterer import CentroidClusterBase


def test_dataset_api_kmeans_integration(test_client: Client, test_dataset_df: Dataset):
    from sklearn.cluster import KMeans

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    class KMeansModel(CentroidClusterBase):
        def __init__(self, model):
            self.model: KMeans = model

        def fit_transform(self, vectors):
            return self.model.fit_predict(vectors)

        def get_centers(self):
            return self.model.cluster_centers_

    model = KMeansModel(model=KMeans())

    clusterer = Clusterer(model=model, alias=alias)

    clusterer.fit(dataset=test_dataset_df, vector_fields=[vector_field])

    assert f"_cluster_.{vector_field}.{alias}" in test_dataset_df.schema


def test_cluster_integration(test_client, test_sample_vector_dataset):
    """Test for the entire clustering workflow."""
    # Retrieve a previous dataset
    VECTOR_FIELD = "sample_1_vector_"
    ALIAS = "kmeans_10"
    documents = test_client.datasets.documents.list(test_sample_vector_dataset)

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


@pytest.mark.parametrize(
    "vector_fields", [["sample_1_vector_"], ["sample_2_vector_", "sample_1_vector_"]]
)
def test_cluster_integration_one_liner(
    test_client: Client, test_sample_vector_dataset, vector_fields
):
    """Smoke Test for the entire clustering workflow."""
    # Retrieve a previous dataset
    VECTOR_FIELDS = vector_fields
    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=test_sample_vector_dataset,
        vector_fields=VECTOR_FIELDS,
        overwrite=True,
        alias="sample_cluster",
    )
    assert True
