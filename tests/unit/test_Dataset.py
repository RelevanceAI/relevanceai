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

    def test_sample(self, test_client, test_sample_vector_dataset):
        df = test_client.Dataset(test_sample_vector_dataset)
        sample_n = df.cluster(n=10)
        assert len(sample_n) == 10
