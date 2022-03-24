from relevanceai.dataset import Dataset


def test_series_sample(test_dataset: Dataset):
    sample_n = test_dataset["sample_1_label"].sample(n=10)
    assert len(sample_n) == 10


def test_series_sample(test_dataset: Dataset):
    sample_n = test_dataset[["sample_1_label", "sample_2_label"]].sample(n=10)
    assert len(sample_n) == 10
    assert len(sample_n[0].keys()) == 3


def test_series_add(test_dataset: Dataset):
    series_1 = test_dataset["sample_1_value"]
    series_2 = test_dataset["sample_2_value"]
    series_3 = test_dataset["sample_1_label"]

    series_1 + series_2

    try:
        series_1 + series_3
    except ValueError:
        assert True

    try:
        series_3 + series_2
    except ValueError:
        assert True
