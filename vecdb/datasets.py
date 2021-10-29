"""
Datasets to mock
"""
import requests
import pandas as pd


def get_games_dataset() -> list:
    """Function to download a sample games dataset.

    Dataset from https://www.freetogame.com/
    """
    return requests.get(
        "https://www.freetogame.com/api/games"
    ).json()


def get_ecommerce_dataset() -> list:
    """Function to download a sample ecommerce dataset
    
    Dataset from https://data.world/crowdflower/ecommerce-search-relevance
    """
    df = pd.read_csv('https://query.data.world/s/glc7oe2ssd252scha53mu7dy2e7cft', encoding='ISO-8859-1').dropna()
    return df.to_dict('records')

