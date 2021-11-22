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
    import relevanceai.visualise.constants
    from relevanceai.visualise.dim_reduction import dim_reduce
    from relevanceai.visualise.cluster import cluster
    from relevanceai.visualise.projector import Projector

    assert True
