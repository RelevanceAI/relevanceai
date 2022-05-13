"""
Simple tests to ensure import errors don't happen
"""


def test_client():
    from relevanceai import Client

    assert True


def test_datasets():
    from relevanceai.utils.datasets import get_games_dataset
    from relevanceai.utils.datasets import get_ecommerce_1_dataset
    from relevanceai.utils.datasets import get_online_retail_dataset
    from relevanceai.utils.datasets import get_news_dataset
    from relevanceai.utils.datasets import get_ecommerce_3_dataset

    assert True


def test_core():
    from relevanceai.operations.cluster.cluster import ClusterOps
    from relevanceai.operations.dr.base import DimReduction

    assert True
