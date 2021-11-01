"""Datasets to mock
"""
import requests
import vecdb_logging


def get_games_dataset() -> list:
    """Function to download a sample games dataset.
    """
    return requests.get(
        "https://www.freetogame.com/api/games"
    ).json()

def get_ecommerce_dataset(db_name: str = 'ecommerce-5', count: int = 1000, base_url = "https://api-aueast.relevance.ai/v1/"):
    from .batch.client import BatchAPIClient

    project = "dummy-collections"
    api_key = "UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw"   # read access
    client = BatchAPIClient(project, api_key, base_url = base_url)
    response = client.datasets.documents.list(db_name, page_size=count)
    if "message" in response:
        logger = vecdb_logging.create_logger()
        logger.error(response["message"])
    return response