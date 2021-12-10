"""
Datasets to mock
"""

from typing import List, Union, Dict, Any
import sys
import pandas as pd
import requests

THIS_MODULE = sys.modules[__name__]
DATASETS = [
    "games",
    "ecommerce_1",
    "ecommerce_2",
    "ecommerce_3",
    "online_retail",
    "news",
    "flipkart",
    "realestate",
]


def select_fields_from_json(json, select_fields):
    return [{key: i[key] for key in select_fields} for i in json]


class ExampleDatasets:
    def __init__(self):
        self.datasets = DATASETS

    def list_datasets(self):
        """List of example datasets available to download"""
        return self.datasets

    def get_dataset(self, name, number_of_documents=None, select_fields=[]):
        """
        Download an example dataset
        Parameters
        ----------
        name: string
            Name of example dataset
        number_of_documents: int
            Number of documents to download
        select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
        """
        if name in self.datasets:
            return getattr(THIS_MODULE, f"get_{name}_dataset")(
                number_of_documents, select_fields
            )
        else:
            raise ValueError("Not a valid dataset")

    @staticmethod
    def _get_dummy_dataset(db_name, number_of_documents, select_fields=[]):
        from .http_client import Client

        project = "dummy-collections"
        api_key = (
            "UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw"  # read access
        )
        client = Client(
            project,
            api_key,
        )
        docs = client.get_documents(
            db_name,
            number_of_documents=number_of_documents,
            select_fields=select_fields,
        )
        return docs

    @staticmethod
    def _get_online_dataset(
        url, number_of_documents, select_fields=[], encoding=None, csv=True
    ):
        if csv:
            data = pd.read_csv(url, index_col=0, encoding=encoding).to_dict(
                orient="records"
            )
        else:
            data = pd.read_excel(url, index_col=0).to_dict(orient="records")
        if number_of_documents:
            data = data[:number_of_documents]
        if len(select_fields) > 0:
            data = select_fields_from_json(data, select_fields)
        return data

    @staticmethod
    def _get_api_dataset(url, number_of_documents, select_fields=[]):
        data = requests.get(url).json()
        if number_of_documents:
            data = data[:number_of_documents]
        if len(select_fields) > 0:
            data = select_fields_from_json(data, select_fields)
        return data


def get_games_dataset(
    number_of_documents: Union[None, int] = 365, select_fields: list = []
) -> List:
    """
    Download an example games dataset (https://www.freetogame.com/) \n
    Total Len: 365 \n
    Sample document:

    >>> {'id': 1,
    >>> 'title': 'Dauntless',
    >>> 'thumbnail': 'https://www.freetogame.com/g/1/thumbnail.jpg',
    >>> 'short_description': 'A free-to-play, co-op action RPG with gameplay similar to Monster Hunter.',
    >>> 'game_url': 'https://www.freetogame.com/open/dauntless',
    >>> 'genre': 'MMORPG',
    >>> 'platform': 'PC (Windows)',
    >>> 'publisher': 'Phoenix Labs',
    >>> 'developer': 'Phoenix Labs, Iron Galaxy',
    >>> 'release_date': '2019-05-21',
    >>> 'freetogame_profile_url': 'https://www.freetogame.com/dauntless'
    >>> }

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 365

    return ExampleDatasets._get_api_dataset(
        "https://www.freetogame.com/api/games", number_of_documents, select_fields
    )


def get_ecommerce_1_dataset(
    number_of_documents: int = 1000, select_fields: list = []
) -> List[Dict[Any, Any]]:
    """
    Download an example e-commerce dataset \n
    Total Len: 14058 \n
    Sample document:

    >>> {'_id': 'b7fc9acbc9ddd18855f96863d37a4fe9',
    >>> 'uniq_id': 'b7fc9acbc9ddd18855f96863d37a4fe9',
    >>> 'crawl_timestamp': '2016-04-24 18:34:50 +0000',
    >>> 'product_url': 'http://www.flipkart.com/babeezworld-baby-boy-s-romper/p/itmehyhguebbzb6h?pid=DRPEHYHGNAF5UYUQ',
    >>> 'product_name': "Babeezworld Baby Boy's Romper",
    >>> 'product_category_tree': '["Baby Care >> Infant Wear >> Baby Boys\' Clothes >> Dungarees & Jumpsuits >> Dungarees >> Babeezworld Dungarees >> Babeezworld Baby Boy\'s Romper"]',
    >>> 'pid': 'DRPEHYHGNAF5UYUQ',
    >>> 'retail_price': 999,
    >>> 'discounted_price': 499,
    >>> 'image': '["http://img5a.flixcart.com/image/dungaree-romper/x/f/r/1012blue-yellow-3-6-babeezworld-3-6-months-original-imaehydgqkkadjud.jpeg", "http://img5a.flixcart.com/image/dungaree-romper/x/f/r/1012blue-yellow-3-6-babeezworld-3-6-months-original-imaehbf3h3jsmzhb.jpeg", "http://img5a.flixcart.com/image/dungaree-romper/x/f/r/1012blue-yellow-3-6-babeezworld-3-6-months-original-imaehbf4nupsmhzt.jpeg", "http://img5a.flixcart.com/image/dungaree-romper/z/n/u/1012blue-yellow-6-9-babeezworld-6-9-months-original-imaehbfbkynega2z.jpeg", "http://img6a.flixcart.com/image/dungaree-romper/z/n/u/1012blue-yellow-6-9-babeezworld-6-9-months-original-imaehbfbjjffht4e.jpeg"]',
    >>> 'is_FK_Advantage_product': False,
    >>> 'description': "Key Features of Babeezworld Baby Boy's Romper Fabric: Cotton Brand Color: Blue:Yellow,Babeezworld Baby Boy's Romper Price: Rs. 499 Babeezworld presents a cute baby dungaree set for your little one.This dungaree set comes with a trendy round neck cotton t-shirt with shoulder loops for ease and comfort fit. Set is made of soft cotton material and has adjustable straps with two button closures. The front has a special character which gives it a stylish look. This dungaree set is an ideal pick for this summer and is available in multiple colors.,Specifications of Babeezworld Baby Boy's Romper Top Details Sleeve Half Sleeve Number of Contents in Sales Package Pack of 2 Fabric Cotton Type Romper Neck Round Neck General Details Pattern Printed Ideal For Baby Boy's Fabric Care Wash with Similar Colors, Use Detergent for Colors",
    >>> 'product_rating': 'No rating available',
    >>> 'overall_rating': 'No rating available',
    >>> 'brand': 'Babeezworld',
    >>> 'product_specifications': '{"product_specification"=>[{"key"=>"Sleeve", "value"=>"Half Sleeve"}, {"key"=>"Number of Contents in Sales Package", "value"=>"Pack of 2"}, {"key"=>"Fabric", "value"=>"Cotton"}, {"key"=>"Type", "value"=>"Romper"}, {"key"=>"Neck", "value"=>"Round Neck"}, {"key"=>"Pattern", "value"=>"Printed"}, {"key"=>"Ideal For", "value"=>"Baby Boy\'s"}, {"value"=>"Wash with Similar Colors, Use Detergent for Colors"}]}',
    >>> 'image_first': 'http://img5a.flixcart.com/image/dungaree-romper/x/f/r/1012blue-yellow-3-6-babeezworld-3-6-months-original-imaehydgqkkadjud.jpeg',
    >>> 'category': 'Baby Care ',
    >>> 'insert_date_': '2021-08-13T11:38:52.110Z'
    >>>  ...
    >>>  }

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 1000
    return ExampleDatasets._get_dummy_dataset(
        "ecommerce-5", number_of_documents, select_fields
    )


def get_ecommerce_2_dataset(
    number_of_documents: int = 1000,
    select_fields: list = [
        "_id",
        "product_image",
        "product_link",
        "product_title",
        "product_price",
        "query",
        "source",
    ],
):
    """
    Download an example e-commerce dataset \n
    Total Len: 739 \n
    Sample document:

    >>> {'_id': '711160239',
    >>> '_unit_id': 711160239,
    >>> 'relevance': 3.67,
    >>> 'relevance:variance': 0.47100000000000003,
    >>> 'product_image': 'https://thumbs4.ebaystatic.com/d/l225/pict/321567405391_1.jpg',
    >>> 'product_link': 'https://www.ebay.com/itm/20-36-Mens-Silver-Stainless-Steel-Braided-Wheat-Chain-Necklace-Jewelry-3-4-5-6MM-/321567405391?pt=LH_DefaultDomain_0&var=&hash=item4adee9354f',
    >>> 'product_price': '$7.99 to $12.99',
    >>> 'product_title': '20-36Mens Silver Stainless Steel Braided Wheat Chain Necklace Jewelry 3/4/5/6MM"',
    >>> 'query': 'steel necklace',
    >>> 'rank': 23,
    >>> 'source': 'eBay',
    >>> 'url': 'https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2050601.m570.l1313.TR11.TRC1.A0.H0.Xplant.TRS0&_nkw=steel%20necklace',
    >>> 'product_description': 'eBay item number:321567405391\n\n\n\tSeller assumes all responsibility for this listing
    >>> ...
    >>> }

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 1000
    docs = ExampleDatasets._get_dummy_dataset(
        "quickstart_data_sample", number_of_documents, select_fields
    )
    for d in docs:
        if "image_first" in d:
            d["image"] = d.pop("image_first")
    return docs


def get_online_retail_dataset(
    number_of_documents: Union[None, int] = 1000, select_fields: list = []
) -> List:
    """
    Download an example online retail dataset from UCI machine learning \n
    Total Len: 541909 \n

    Sample document:

    >>> {'Country': 'United Kingdom',
    >>> 'CustomerID': 17850.0,
    >>> 'Description': 'WHITE HANGING HEART T-LIGHT HOLDER',
    >>> 'InvoiceDate': Timestamp('2010-12-01 08:26:00'),
    >>> 'InvoiceNo': 536365,
    >>> 'Quantity': 6,
    >>> 'StockCode': '85123A',
    >>> 'UnitPrice': 2.55}

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 1000
    return ExampleDatasets._get_online_dataset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
        number_of_documents,
        select_fields,
        csv=False,
    )


def get_news_dataset(
    number_of_documents: Union[None, int] = 250, select_fields: list = []
) -> List:
    """
    Download an example news dataset \n
    Total Len: 250 \n
    Sample document:

    >>> {'authors': 'Ruth Harris',
    >>> 'content': 'Sometimes the power of Christmas will make you do wild and wonderful things. You do not need to believe in the Holy Trinity to believe in the positive power of doing good for others.
    >>> 'domain': 'awm.com',
    >>> 'id': 141,
    >>> 'inserted_at': '2018-02-02 01:19:41.756632',
    >>> 'keywords': nan,
    >>> 'meta_description': nan,
    >>> 'meta_keywords': "['']",
    >>> 'scraped_at': '2018-01-25 16:17:44.789555',
    >>> 'summary': nan,
    >>> 'tags': nan,
    >>> 'title': 'Church Congregation Brings Gift to Waitresses Working on Christmas Eve, Has Them Crying (video)',
    >>> 'type': 'unreliable',
    >>> 'updated_at': '2018-02-02 01:19:41.756664',
    >>> 'url': 'http://awm.com/church-congregation-brings-gift-to-waitresses-working-on-christmas-eve-has-them-crying-video/'}

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 250
    return ExampleDatasets._get_online_dataset(
        "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv",
        number_of_documents,
        select_fields,
    )


def get_ecommerce_3_dataset(
    number_of_documents: Union[None, int] = 1000, select_fields: list = []
) -> List:
    """
    Download an example ecommerce dataset (https://data.world/crowdflower/ecommerce-search-relevance) \n
    Total Len: 15528 \n
    Sample document:

    >>> {'_unit_id': 711158459,
    >>> 'product_description': 'The PlayStation 4 system opens the door to an '
    >>>                     'incredible journey through immersive new gaming '
    >>>                     'worlds and a deeply connected gaming community. Step '
    >>>                     'into living, breathing worlds where you are hero of '
    >>>                     '...',
    >>> 'product_image': 'http://thumbs2.ebaystatic.com/d/l225/m/mzvzEUIknaQclZ801YCY1ew.jpg',
    >>> 'product_link': 'http://www.ebay.com/itm/Sony-PlayStation-4-PS4-Latest-Model-500-GB-Jet-Black-Console-/321459436277?pt=LH_DefaultDomain_0&hash=item4ad879baf5',
    >>> 'product_price': '$329.98 ',
    >>> 'product_title': 'Sony PlayStation 4 (PS4) (Latest Model)- 500 GB Jet Black '
    >>>                 'Console',
    >>> 'query': 'playstation 4',
    >>> 'rank': 1,
    >>> 'relevance': 3.67,
    >>> 'relevance:variance': 0.471,
    >>> 'source': 'eBay',
    >>> 'url': 'http://www.ebay.com/sch/i.html?_from=R40&_trksid=p2050601.m570.l1313.TR11.TRC1.A0.H0.Xplant.TRS0&_nkw=playstation%204'}

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 1000
    df = ExampleDatasets._get_online_dataset(
        "https://query.data.world/s/glc7oe2ssd252scha53mu7dy2e7cft",
        number_of_documents,
        select_fields,
        encoding="ISO-8859-1",
    )
    df = pd.DataFrame(df)
    if "product_image" in df.columns:
        df["product_image"] = df["product_image"].str.replace("http://", "https://")
    if "product_link" in df.columns:
        df["product_link"] = df["product_link"].str.replace("http://", "https://")
    if "url" in df.columns:
        df["url"] = df["url"].str.replace("http://", "https://")
    if "_unit_id" in df.columns:
        df["_id"] = df["_unit_id"].astype(str)
    docs = [
        {k: v for k, v in doc.items() if not pd.isna(v)}
        for doc in df.to_dict(orient="records")
    ]
    return docs


def get_flipkart_dataset(
    number_of_documents: Union[None, int] = 19920, select_fields: list = []
) -> List:
    """
    Download an example flipkart ecommerce dataset \n
    Total Len: 19920 \n
    Sample document:

    >>> {'_id': 0,
    >>> 'product_name': "Alisha Solid Women's Cycling Shorts",
    >>> 'description': "Key Features of Alisha Solid Women's Cycling Shorts Cotton Lycra Navy, Red, Navy,Specifications of Alisha Solid Women's Cycling Shorts Shorts Details Number of Contents in Sales Package Pack of 3 Fabric Cotton Lycra Type Cycling Shorts General Details Pattern Solid Ideal For Women's Fabric Care Gentle Machine Wash in Lukewarm Water, Do Not Bleach Additional Details Style Code ALTHT_3P_21 In the Box 3 shorts",
    >>> 'retail_price': 999.0}

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 19920
    return ExampleDatasets._get_online_dataset(
        "https://raw.githubusercontent.com/arditoibryan/Projects/master/20211108_flipkart_df/flipkart.csv",
        number_of_documents,
        select_fields,
    )


def get_realestate_dataset(number_of_documents: int = 50, select_fields: list = []):
    """
    Download an example real-estate dataset \n
    Total Len: 5885 \n
    Sample document:

    >>> {'propertyDetails': {'area': 'North Shore - Lower',
    >>> 'carspaces': 1,
    >>> 'streetNumber': '28',
    >>> 'latitude': -33.8115768,
    >>> 'allPropertyTypes': ['ApartmentUnitFlat'],
    >>> 'postcode': '2066',
    >>> 'unitNumber': '6',
    >>> 'bathrooms': 1.0,
    >>> 'bedrooms': 1.0,
    >>> 'features': ['BuiltInWardrobes', 'InternalLaundry','Intercom', 'Dishwasher'],
    >>> 'street': 'Epping Road',
    >>> 'propertyType': 'ApartmentUnitFlat',
    >>> 'suburb': 'LANE COVE',
    >>> 'state': 'NSW',
    >>> 'region': 'Sydney Region',
    >>> 'displayableAddress': '6/28 Epping Road, Lane Cove',
    >>> 'longitude': 151.166611},
    >>> 'listingSlug': '6-28-epping-road-lane-cove-nsw-2066-14688794',
    >>> 'id': 14688794,
    >>> 'headline': 'Extra large one bedroom unit',
    >>> 'summaryDescription': '<b></b><br />This modern and spacious one-bedroom apartment situated on the top floor, the quiet rear side of a small 2 story boutique block, enjoys a wonderfully private, leafy, and greenly outlook from 2 sides and balcony. A short stroll to city buse...',
    >>> 'advertiser': 'Ray White Lane Cove',
    >>> 'image_url': 'https://bucket-api.domain.com.au/v1/bucket/image/14688794_1_1_201203_101135-w1600-h1065',
    >>> 'insert_date_': '2021-03-01T14:19:22.805086',
    >>> 'labels': [],
    >>> 'image_url_5': 'https://bucket-api.domain.com.au/v1/bucket/image/14688794_5_1_201203_101135-w1600-h1067',
    >>> 'image_url_4': 'https://bucket-api.domain.com.au/v1/bucket/image/14688794_4_1_201203_101135-w1600-h1067',
    >>> 'priceDetails': {'displayPrice': 'Deposit Taken ! Inspection Cancelled thank you !!!'}
    >>> ...
    >>> }

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 50
    return ExampleDatasets._get_dummy_dataset(
        "realestate", number_of_documents, select_fields
    )


def get_mission_statements_dataset(
    number_of_documents: Union[None, int] = 1433, select_fields: list = []
) -> List:
    """Function to download a sample company mission statement dataset.
    Total Len: 1433
    Sample document:
    {'_id': 0,
    'company': 'Starbucks',
    'text': 'Establish Starbucks as the premier purveyor of the finest coffee in the world while maintaining our uncompromising principles while we grow.'},
    """

    """
    Download an example ompany mission statement dataset \n
    Total Len: 1433 \n 
    Sample document:

    >>> {'_id': 0,
    >>> 'company': 'Starbucks',
    >>> 'text': 'Establish Starbucks as the premier purveyor of the finest coffee in the world while maintaining our uncompromising principles while we grow.'
    >>> },
    

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
            Fields to include in the dataset, empty array/list means all fields.
    """
    if number_of_documents is None:
        number_of_documents = 514330
    return ExampleDatasets._get_online_dataset(
        "https://raw.githubusercontent.com/arditoibryan/Projects/master/20211111_company_statements/companies_preprocessed.csv",
        number_of_documents,
        select_fields,
    )


def get_machine_learning_research_dataset():
    """Here we get our Machine Learning research dataset."""
    raise NotImplementedError


get_dummy_ecommerce_dataset = get_ecommerce_1_dataset
get_sample_ecommerce_dataset = get_ecommerce_2_dataset
get_ecommerce_dataset = get_ecommerce_3_dataset
