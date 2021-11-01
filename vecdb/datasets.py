"""
Datasets to mock
"""
import requests
import pandas as pd
from typing import Union, List


def get_games_dataset() -> list:
    """Function to download a sample games dataset.

    Dataset from https://www.freetogame.com/
    """
    return requests.get(
        "https://www.freetogame.com/api/games"
    ).json()



def get_online_retail_dataset(number_of_documents: Union[None, int] = 1000) -> List:
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




def get_news_dataset(sample=True) -> List:
    """News dataset
    Sample document:
    {'Unnamed: 0': 0,
     'authors': 'Ruth Harris',
     'content': 'Sometimes the power of Christmas will make you do wild and wonderful things. You do not need to believe in the Holy Trinity to believe in the positive power of doing good for others. The simple act of giving without receiving is lost on many of us these days, as worries about money and success hold us back from giving to others who are in need. One congregation in Ohio was moved to action by the power of a sermon given at their church on Christmas Eve. The pastor at Grand Lake United Methodist Church in Celina, Ohio gave an emotional sermon about the importance of understanding the message of Jesus.\n\nFor many religious people the message of Jesus is to help others before yourself, to make sure the people who are suffering get the help they need to enjoy life a little bit. The sermon was really about generosity and what that can look like in our lives. Jesus lived a long time ago and he acted generously in the fashion of his time – but what would a generous act look like in our times? That was the focus of the sermon.\n\nThe potency of the sermon was not lost on the congregation, who were so moved they had to take action! After the sermon ended, the congregation decided to take an offering. A bowl was passed around the room and everyone pitched in what they could on this Christmas Eve with the words of the sermon still ringing in their ears.\n\nWhat did they do with this offering? Members of the congregation drove down to the local Waffle House to visit the ladies working the night shift. What a great choice on this most holy of days when everyone should be with their families!\n\nThe ladies working at Waffle House clearly were not with their families. They had no choice but to work on this holy day because it paid the bills. The congregation understood the sacrifice being made by these ladies, and wanted to help them out. They donated the entire offering to be split amongst the ladies at Waffle House.\n\nIn total that amounted to $3,500 being split amongst the staff. What a beautiful moment! What a perfect example of what the preacher was talking about in his sermon! Doing a good deed like this on Christmas really helped ease the burden felt by the ladies working at Waffle House. Sure, they could not see their families, but at least they got a little gift from the good people of their community.\n\nPerhaps the best part about this whole event was that the congregation did not ask anything in return. It was a simple act of generosity from people who understood the pain being felt by another group and sought to alleviate some of that pain. It speaks volumes about the merits of the Church in our daily lives. This simple act brought the entire community together because it showed empathy and compassion on the most special day of the year.',
     'domain': 'awm.com',
     'id': 141,
     'inserted_at': '2018-02-02 01:19:41.756632',
     'keywords': nan,
     'meta_description': nan,
     'meta_keywords': "['']",
     'scraped_at': '2018-01-25 16:17:44.789555',
     'summary': nan,
     'tags': nan,
     'title': 'Church Congregation Brings Gift to Waitresses Working on Christmas Eve, Has Them Crying (video)',
     'type': 'unreliable',
     'updated_at': '2018-02-02 01:19:41.756664',
     'url': 'http://awm.com/church-congregation-brings-gift-to-waitresses-working-on-christmas-eve-has-them-crying-video/'}
    """
    return pd.read_csv(
        "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv"
    ).to_dict(orient='records')



    
def get_ecommerce_dataset(number_of_documents: Union[None, int] = 1000) -> List:
    """Function to download a sample ecommerce dataset
    Dataset from https://data.world/crowdflower/ecommerce-search-relevance 
    Total Len: 15528
    Sample document:
    {'_unit_id': 711158459,
    'product_description': 'The PlayStation 4 system opens the door to an '
                        'incredible journey through immersive new gaming '
                        'worlds and a deeply connected gaming community. Step '
                        'into living, breathing worlds where you are hero of '
                        '...',
    'product_image': 'http://thumbs2.ebaystatic.com/d/l225/m/mzvzEUIknaQclZ801YCY1ew.jpg',
    'product_link': 'http://www.ebay.com/itm/Sony-PlayStation-4-PS4-Latest-Model-500-GB-Jet-Black-Console-/321459436277?pt=LH_DefaultDomain_0&hash=item4ad879baf5',
    'product_price': '$329.98 ',
    'product_title': 'Sony PlayStation 4 (PS4) (Latest Model)- 500 GB Jet Black '
                    'Console',
    'query': 'playstation 4',
    'rank': 1,
    'relevance': 3.67,
    'relevance:variance': 0.471,
    'source': 'eBay',
    'url': 'http://www.ebay.com/sch/i.html?_from=R40&_trksid=p2050601.m570.l1313.TR11.TRC1.A0.H0.Xplant.TRS0&_nkw=playstation%204'}
    """
    df = pd.read_csv(
        'https://query.data.world/s/glc7oe2ssd252scha53mu7dy2e7cft', encoding='ISO-8859-1'
    ).iloc[:number_of_documents, :].dropna()
    df['product_image'] = df['product_image'].str.replace('http://', 'https://')
    df['product_link'] = df['product_link'].str.replace('http://', 'https://')
    df['url'] = df['url'].str.replace('http://', 'https://')
    return df.to_dict('records')
