import pytest

from sklearn.cluster import MiniBatchKMeans

from relevanceai.client import Client
from relevanceai.dataset import Dataset

from tests.globals.constants import NOT_IMPLEMENTED


class TestClusterOps:
    def test_dataset_cluster(self, test_dataset: Dataset):
        vector_field = "sample_1_vector_"

        alias = "cluster_test_1"
        test_dataset.cluster(
            model="kmeans",
            n_clusters=10,
            alias=alias,
            vector_fields=[vector_field],
        )
        assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema

        alias = "cluster_test_2"
        test_dataset.cluster(
            model="kmeans",
            cluster_config=dict(n_clusters=10),
            alias=alias,
        )
        assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema

        alias = "cluster_test_3"
        test_dataset.cluster(
            model="optics",
            alias=alias,
        )
        assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema

        alias = "cluster_test_4"
        test_dataset.cluster(
            model=MiniBatchKMeans(n_clusters=10),
            alias=alias,
        )
        assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema

    def test_ClusterOps(self, test_client: Client, test_dataset: Dataset):
        vector_field = "sample_1_vector_"
        alias = "kmeans-10"

        operator = test_client.ClusterOps(model="kmeans", n_clusters=10)
        operator(test_dataset, vector_fields=[vector_field])

        schema = test_dataset.schema
        assert f"_cluster_" in schema
        assert f"_cluster_{vector_field}" in schema
        assert f"_cluster_{vector_field}.{alias}" in schema

    @pytest.mark.skip(NOT_IMPLEMENTED)
    def test_list_closest(self, test_client: Client, test_dataset: Dataset):
        assert False

    @pytest.mark.skip(NOT_IMPLEMENTED)
    def test_list_furthest(self, test_client: Client, test_dataset: Dataset):
        assert False
