# -*- coding: utf-8 -*-
def test_smoke_installation():
    """Simple test to ensure there are no breaking installations."""
    from vecdb import VecDBClient
    assert True


def test_datasets_smoke():
    """Testing dataset imports"""
    from vecdb.datasets import get_games_dataset
    from vecdb.datasets import get_dummy_ecommerce_dataset
    from vecdb.datasets import get_online_retail_dataset
    from vecdb.datasets import get_news_dataset
    from vecdb.datasets import get_ecommerce_dataset
    assert True
