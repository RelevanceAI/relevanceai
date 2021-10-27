"""Datasets to mock
"""
import requests

def get_games_dataset() -> list:
    """Function to download a sample games dataset.
    """
    return requests.get(
        "https://www.freetogame.com/api/games"
    ).json()

def get_ecommerce_dataset():
    """Function to download a sample ecommerce dataset
    """
    raise NotImplementedError
