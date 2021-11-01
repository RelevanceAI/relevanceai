import pytest

def test_smoke_installation():
    """Simple test to ensure there are no breaking installations.
    """
    # Import the client
    from vecdb import VecDBClient
    assert True

def test_datasets_smoke():
    from vecdb.datasets import get_games_dataset
    from vecdb.datasets import get_dummy_ecommerce_dataset
    from vecdb.datasets import get_online_retail_dataset
    from vecdb.datasets import get_news_dataset
    from vecdb.datasets import get_ecommerce_dataset
    assert True

# def test_datasets_smoke():
#     from vecdb.datasets import get_games_dataset
#     assert True
