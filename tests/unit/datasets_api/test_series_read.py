from relevanceai.http_client import Client, Dataset


def test_series_sample(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    sample_n = df["sample_1_label"].sample(n=10)
    assert len(sample_n) == 10


def test_series_sample(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    sample_n = df[["sample_1_label", "sample_2_label"]].sample(n=10)
    assert len(sample_n) == 10
    assert len(sample_n[0].keys()) == 3
