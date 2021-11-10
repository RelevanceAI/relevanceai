def test_client_smoke():
    """Simple test to ensure there are no breaking installations."""
    # Import the client
    from relevanceai import Client

    assert True


def test_datasets_smoke():
    from relevanceai.datasets import get_games_dataset
    from relevanceai.datasets import get_dummy_ecommerce_dataset
    from relevanceai.datasets import get_online_retail_dataset
    from relevanceai.datasets import get_news_dataset
    from relevanceai.datasets import get_ecommerce_dataset

    assert True


def test_projector_smoke():
    import relevanceai.visualise.constants
    from relevanceai.visualise.dataset import Dataset
    from relevanceai.visualise.dim_reduction import DimReduction
    from relevanceai.visualise.cluster import Cluster
    from relevanceai.visualise.projector import Projector
    
    assert True
