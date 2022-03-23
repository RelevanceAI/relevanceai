import pytest


def test_get_games_dataset_subset():
    from relevanceai.utils.datasets import get_games_dataset

    assert len(get_games_dataset(number_of_documents=100)) == 100
