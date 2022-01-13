import pytest


class TestDatset:
    def test_Dataset(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        assert True

    def test_info(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        info = df.info()
        assert True

    def test_shape(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        shape = df.shape
        assert True

    def test_head(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        head = df.head()
        assert True

    def test_describe(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        description = df.describe()
        assert True

    def test_cluster(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        centroids = df.cluster(field="sample_1_vector_", overwrite=True)
        assert True

    def test_groupby_agg(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        agg = df.agg({"sample_1_label": "avg"})
        groupby_agg = df.groupby(["sample_1_description"]).agg(
            {"sample_1_label": "avg"}
        )
        assert True

    def test_centroids(self, test_client, test_clustered_dataset):
        df = test_client.Dataset(test_clustered_dataset)
        closest = df.centroids(["sample_1_vector_"], "kmeans_10").closest()
        furthest = df.centroids(["sample_1_vector_"], "kmeans_10").furthest()
        agg = df.centroids(["sample_1_vector_"], "kmeans_10").agg(
            {"sample_2_label": "avg"}
        )
        groupby_agg = (
            df.centroids(["sample_1_vector_"], "kmeans_10")
            .groupby(["sample_3_description"])
            .agg({"sample_2_label": "avg"})
        )
        assert True

    def test_sample(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        sample_n = df.sample(n=10)
        assert len(sample_n) == 10

    def test_cluster(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        df.cluster("sample_1_vector_", n_clusters=10, overwrite=True)
        db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
        assert "_cluster_" in db_health
        assert "_cluster_.sample_1_vector_.kmeans_10" in db_health

    def test_custom_cluster(self, test_client, test_sample_vector_dataset):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=2, random_state=0)
        df = test_client.Dataset(test_sample_vector_dataset)
        df.cluster("sample_1_vector_", clusterer=kmeans, overwrite=True)
        db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
        assert "_cluster_" in db_health
        assert "_cluster_.sample_1_vector_.default" in db_health
