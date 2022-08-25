import pytest
import time
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
            ingest_in_background=False,
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
            include_cluster_report=False,
        )
        assert f"_cluster_.{self.vector_field}.{alias}" in test_dataset.schema

    @pytest.mark.skip(reason="minibatch kmeans not implemented yet")
    def test_dataset_cluster_4(self, test_dataset: Dataset):
        alias = "cluster_test_4"
        test_dataset.cluster(
            model=MiniBatchKMeans(n_clusters=10),
            alias=alias,
            vector_fields=[self.vector_field],
        )
        assert f"_cluster_.{self.vector_field}.{alias}" in test_dataset.schema

    @pytest.mark.skip(reason="broken from refactor")
    def testClusterUtils(self, test_client: Client, test_dataset: Dataset):
        vector_field = "sample_1_vector_"
        alias = "kmeans-10"

        operator = test_dataset.cluster(
            vector_fields=[vector_field],
            model="kmeans",
            model_kwargs=dict(n_clusters=10),
            alias=alias,
        )
        operator(test_dataset)

        schema = test_dataset.schema
        assert f"_cluster_" in schema
        assert f"_cluster_.{vector_field}" in schema
        assert f"_cluster_.{vector_field}.{alias}" in schema

    def test_list_closest(self, test_client: Client, test_dataset: Dataset):
        n_clusters = 10

        clusterer = test_dataset.cluster(
            alias="new_clustering",
            model=MiniBatchKMeans(n_clusters=n_clusters),
            vector_fields=["sample_1_vector_"],
        )
        cluster_ids = ["cluster_0", "cluster_6", "cluster_3"]
        time.sleep(5)
        closests = clusterer.list_closest(
            cluster_ids=cluster_ids,
            approx=0,
            page_size=10,
            include_vector=False,
            include_count=False,
            similarity_metric="l1",
            select_fields=["_id"],
        )["results"]

        for id in cluster_ids:
            assert id in closests

        # clusterer = test_client.ClusterOps(
        #     alias="new_clustering_2",
        #     model="kmeans",
        #     n_clusters=n_clusters,
        #     vector_fields=["sample_2_vector_"],
        # )
        # clusterer.run(test_dataset)
        # closests = clusterer.list_closest(
        #     centroid_vector_fields=["sample_2_vector_"],
        #     similarity_metric="cosine",
        #     select_fields=["_id"],
        #     include_vector=True,
        # )["results"]
        # assert len(closests) == n_clusters

    @pytest.mark.skip(NOT_IMPLEMENTED)
    def test_list_furthest(self, test_client: Client, test_dataset: Dataset):
        assert False

    @pytest.mark.xfail()
    def test_merge(self, test_client: Client, test_dataset: Dataset):
        # TODO: fix this

        ALIAS = "new_merge_clustering"
        test_dataset.cluster(
            model="kmeans",
            model_kwargs={"n_clusters": 3},
            alias=ALIAS,
            vector_fields=["sample_1_vector_"],
        )

        centroids = test_client.datasets.cluster.centroids.list(
            dataset_id=test_dataset.dataset_id,
            alias=ALIAS,
            vector_fields=["sample_1_vector_"],
        )["results"]
        assert len(centroids) == 3

        ops = ClusterOps.from_dataset(
            dataset=test_dataset,
            alias=ALIAS,
            vector_fields=["sample_1_vector_"],
        )

        ops.merge(cluster_labels=["cluster_0", "cluster_1"], alias="new_clustering")

        centroids = test_client.datasets.cluster.centroids.list(
            dataset_id=test_dataset.dataset_id,
            alias=ALIAS,
            vector_fields=["sample_1_vector_"],
        )["results"]
        assert len(centroids) == 2
