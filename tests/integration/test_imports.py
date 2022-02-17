"""
Simple tests to ensure import errors don't happen
"""


def test_client_smoke():
    """Simple test to ensure there are no breaking installations."""
    # Import the client
    from relevanceai import Client

    assert True


def test_datasets_smoke():
    from relevanceai.datasets import get_games_dataset
    from relevanceai.datasets import get_ecommerce_1_dataset
    from relevanceai.datasets import get_online_retail_dataset
    from relevanceai.datasets import get_news_dataset
    from relevanceai.datasets import get_ecommerce_3_dataset

    assert True


def test_projector_smoke():
    import relevanceai.vector_tools.constants
    from relevanceai.vector_tools.dim_reduction import DimReduction
    from relevanceai.vector_tools.cluster import Cluster
    from relevanceai.visualise.projector import Projector

    assert True
