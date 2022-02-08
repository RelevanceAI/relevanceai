import pytest
from relevanceai.http_client import Client, Dataset


@pytest.mark.skip(reason="Node has to fix first.")
def test_series_sample(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    sample_n = df["sample_1_label"].sample(n=10)
    assert len(sample_n) == 10


@pytest.mark.skip(reason="Node has to fix first.")
def test_series_sample_2(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    sample_n = df[["sample_1_label", "sample_2_label"]].sample(n=10)
    assert len(sample_n) == 10
    # the first one inserted may not necessarily have all values
    assert (
        len(sample_n[0].keys()) == 3
        or len(sample_n[1].keys()) == 3
        or len(sample_n[2]) == 3
    )
