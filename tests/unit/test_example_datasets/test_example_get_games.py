import pytest


@pytest.mark.skip(reason="bad API_KEY on old aus east")
def test_get_games_dataset_subset():
    from relevanceai.datasets import get_games_dataset

    assert len(get_games_dataset(number_of_documents=100)) == 100
