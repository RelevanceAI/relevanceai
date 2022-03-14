"""
Simple tests to ensure import errors don't happen
"""


def test_client_smoke():
    """Simple test to ensure there are no breaking installations."""
    # Import the client
    from relevanceai import Client

    assert True


def test_datasets_smoke():
    from relevanceai.package_utils.datasets import get_games_dataset
    from relevanceai.package_utils.datasets import get_ecommerce_1_dataset
    from relevanceai.package_utils.datasets import get_online_retail_dataset
    from relevanceai.package_utils.datasets import get_news_dataset
    from relevanceai.package_utils.datasets import get_ecommerce_3_dataset

    assert True


def test_projector_smoke():
    import relevanceai.workflows.cluster_ops.constants
    from relevanceai.workflows.dim_reduction_ops.dim_reduction import DimReduction

    assert True
