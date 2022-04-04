from relevanceai import Client
from relevanceai import ClusterOps
from relevanceai.dataset import Dataset


class TestBaseOps:
    def test_init(self, test_client: Client):
        credentials = test_client.credentials
        ops = ClusterOps.init(credentials=credentials)
        assert True

    def test_from_credentials(self, test_client: Client):
        credentials = test_client.credentials
        ops = ClusterOps.init(credentials=credentials)
        assert True

    def test_from_token(self, test_token: str):
        ops = ClusterOps.from_token(token=test_token)
        assert ops.credentials.token == test_token

    def test_from_client(self, test_client: Client):
        ops = ClusterOps.from_client(client=test_client)
        assert ops.credentials.token == test_client.credentials.token

    def test_from_dataset(self, test_clustered_df: Dataset):
        ops = ClusterOps.from_dataset(
            dataset=test_clustered_df,
            alias="kmeans-10",
            vector_fields=["sample_1_vector_"],
        )
        closest = ops.list_closest()
        assert True
