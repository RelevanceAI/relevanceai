from relevanceai.http_client import Client, Dataset


def test_series_sample(test_client: Client, vector_dataset_id: str):
    df = test_client.Dataset(vector_dataset_id)
    sample_n = df["sample_1_label"].sample(n=10)
    assert len(sample_n) == 10


def test_series_sample(test_client: Client, vector_dataset_id: str):
    df = test_client.Dataset(vector_dataset_id)
    sample_n = df[["sample_1_label", "sample_2_label"]].sample(n=10)
    assert len(sample_n) == 10
    assert len(sample_n[0].keys()) == 3
