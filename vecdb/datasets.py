"""Datasets to mock
"""
import requests

def get_games_dataset() -> list:
    """Function to download a sample games dataset.
    """
    return requests.get(
        "https://www.freetogame.com/api/games"
    ).json()

def get_online_retail_dataset(number_of_documents: int=1000) -> list:
    """Online retail dataset from UCI machine learning
    Sample document: 
    {'Country': 'United Kingdom',
     'CustomerID': 17850.0,
     'Description': 'WHITE HANGING HEART T-LIGHT HOLDER',
     'InvoiceDate': Timestamp('2010-12-01 08:26:00'),
     'InvoiceNo': 536365,
     'Quantity': 6,
     'StockCode': '85123A',
     'UnitPrice': 2.55}
    
    """
    df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx")
    if number_of_documents is None:
        return df.to_dict(orient='records')
    return df.to_dict(orient='records')[:number_of_documents]

def get_ecommerce_dataset():
    """Ecommerce dataset
    """
    raise NotImplementedError
