import pytest

from sklearn.cluster import MiniBatchKMeans

from relevanceai import Client
from relevanceai import ClusterOps
from relevanceai.dataset import Dataset

from tests.globals.constants import NOT_IMPLEMENTED


class TestClusterOps:
    vector_field = "sample_1_vector_"

    def test_dataset_cluster_1(self, test_dataset: Dataset):
        alias = "cluster_test_1"
        test_dataset.cluster(
            model="kmeans",
            cluster_config=dict(n_clusters=3),
            alias=alias,
            vector_fields=[self.vector_field],
        )
        assert f"_cluster_.{self.vector_field}.{alias}" in test_dataset.schema

    def test_dataset_cluster_2(self, test_dataset: Dataset):
        alias = "cluster_test_2"
        test_dataset.cluster(
            model="kmeans",
            cluster_config=dict(n_clusters=10),
            alias=alias,
            vector_fields=[self.vector_field],
        )
        assert f"_cluster_.{self.vector_field}.{alias}" in test_dataset.schema

    def test_dataset_cluster_3(self, test_dataset: Dataset):
        alias = "cluster_test_3"
        test_dataset.cluster(
            model="optics",
            alias=alias,
            vector_fields=[self.vector_field],
        )
        assert f"_cluster_.{self.vector_field}.{alias}" in test_dataset.schema

    def test_dataset_cluster_4(self, test_dataset: Dataset):
        alias = "cluster_test_4"
        test_dataset.cluster(
            model=MiniBatchKMeans(n_clusters=10),
            alias=alias,
            vector_fields=[self.vector_field],
        )
        assert f"_cluster_.{self.vector_field}.{alias}" in test_dataset.schema

    def test_ClusterOps(self, test_client: Client, test_dataset: Dataset):
        vector_field = "sample_1_vector_"
        alias = "kmeans-10"

        operator = test_client.ClusterOps(model="kmeans", n_clusters=10)
        operator(test_dataset, vector_fields=[vector_field])

        schema = test_dataset.schema
        assert f"_cluster_" in schema
        assert f"_cluster_.{vector_field}" in schema
        assert f"_cluster_.{vector_field}.{alias}" in schema

    @pytest.mark.skip(NOT_IMPLEMENTED)
    def test_list_closest(self, test_client: Client, test_dataset: Dataset):
        assert False

    @pytest.mark.skip(NOT_IMPLEMENTED)
    def test_list_furthest(self, test_client: Client, test_dataset: Dataset):
        assert False

    def test_merge(self, test_client: Client, test_dataset: Dataset):
        test_dataset.cluster(
            model="kmeans",
            n_clusters=3,
            vector_fields=["sample_1_vector_"],
        )

        ops = ClusterOps.from_dataset(
            dataset=test_dataset,
            alias="kmeans-3",
            vector_fields=["sample_1_vector_"],
        )

        ops.merge(cluster_labels=[0, 1], alias="kmeans-3")
        centroids = ops.services.cluster.centroids.list(
            dataset_id=test_dataset.dataset_id,
            alias="kmeans-3",
            vector_fields=["sample_1_vector_"],
        )["results"]
        assert len(centroids) == 2
