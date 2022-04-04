# -*- coding: utf-8 -*-
"""
Relevance AI Platform offers free datasets for users.
These datasets have been licensed under Apache 2.0.
"""

from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal

import random
import string
import sys
import pandas as pd
import requests

from relevanceai.constants import DATASETS

THIS_MODULE = sys.modules[__name__]


def select_fields_from_json(json, select_fields):
    return [{key: i[key] for key in select_fields} for i in json]


class ExampleDatasets:
    def __init__(self):
        self.datasets = DATASETS

    def list_datasets(self):
        """List of example datasets available to download"""
        return self.datasets

    def get_dataset(
        self, name, number_of_documents=None, select_fields: Optional[List] = None
    ):
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
        select_fields = [] if select_fields is None else select_fields
        if name in self.datasets:
            return getattr(THIS_MODULE, f"get_{name}_dataset")(
                number_of_documents, select_fields
            )
        else:
            raise ValueError("Not a valid dataset")

    @staticmethod
    def _get_dummy_dataset(
        db_name,
        number_of_documents,
        select_fields: Optional[List[str]] = None,
        include_vector: bool = True,
    ):
        from relevanceai.utils.logger import FileLogger
        from relevanceai import Client

        select_fields = [] if select_fields is None else select_fields
        with FileLogger(fn=".relevanceairetrievingdata.logs", verbose=False):
            project = "3a4b969f4d5fae6f850e"
            api_key = "LVpyeWlYOEI4X2lpWW1za3J6Qmg6dldnTVZCczlUZ09pMG5LM2NyejVtdw"  # read access
            region = "us-east-1"
            firebase_uid = "tQ5Yu5frJhOQ8Ge3PpeFoh2325F3"
            token = ":".join([project, api_key, region, firebase_uid])
            client = Client(token=token)
            documents = client._get_documents(
                db_name,
                number_of_documents=number_of_documents,
                select_fields=select_fields,
                include_vector=include_vector,
            )
            client.config.reset()
            return documents

    @staticmethod
    def _get_online_dataset(
        url,
        number_of_documents,
        select_fields: Optional[List] = None,
        encoding=None,
        csv=True,
    ):
        select_fields = [] if select_fields is None else select_fields
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
    def _get_api_dataset(
        url, number_of_documents, select_fields: Optional[List] = None
    ):
        select_fields = [] if select_fields is None else select_fields
        data = requests.get(url).json()
        if number_of_documents:
            data = data[:number_of_documents]
        if len(select_fields) > 0:
            data = select_fields_from_json(data, select_fields)
        return data


def get_games_dataset(
    number_of_documents: Union[None, int] = 365, select_fields: Optional[List] = None
) -> List:
    """
    Download an example games dataset (https://www.freetogame.com/) \n
    Total Len: 365 \n

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
        Fields to include in the dataset, empty array/list means all fields.

    Example
    -------
    .. code-block::

        {
            'id': 1,
            'title': 'Dauntless',
            'thumbnail': 'https://www.freetogame.com/g/1/thumbnail.jpg',
            'short_description': 'A free-to-play, co-op action RPG with gameplay similar to Monster Hunter.',
            'game_url': 'https://www.freetogame.com/open/dauntless',
            'genre': 'MMORPG',
            'platform': 'PC (Windows)',
            'publisher': 'Phoenix Labs',
            'developer': 'Phoenix Labs, Iron Galaxy',
            'release_date': '2019-05-21',
            'freetogame_profile_url': 'https://www.freetogame.com/dauntless'
        }
    """
    select_fields = [] if select_fields is None else select_fields
    if number_of_documents is None:
        number_of_documents = 365

    return ExampleDatasets._get_dummy_dataset(
        "dummy-games-dataset",
        number_of_documents=number_of_documents,
        select_fields=select_fields,
    )


def get_ecommerce_dataset_encoded(
    number_of_documents: int = 739, select_fields: Optional[List] = None
) -> List[Dict[Any, Any]]:
    """
    Download an example e-commerce dataset \n
    Total Len: 739 \n

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
        Fields to include in the dataset, empty array/list means all fields.

    Example
    -------
    .. code-block::

        {
            '_id': 'b7fc9acbc9ddd18855f96863d37a4fe9',
            'product_image': 'https://thumbs4.ebaystatic.com/d/l225/pict/321567405391_1.jpg',
            'product_image_clip_vector_': [...],
            'product_link': 'https://www.ebay.com/itm/20-36-Mens-Silver-Stainless-Steel-Braided-Wheat-Chain-Necklace-Jewelry-3-4-5-6MM-/321567405391?pt=LH_DefaultDomain_0&var=&hash=item4adee9354f',
            'product_price': '$7.99 to $12.99',
            'product_title': '20-36Mens Silver Stainless Steel Braided Wheat Chain Necklace Jewelry 3/4/5/6MM"',
            'product_title_clip_vector_': [...],
            'query': 'steel necklace',
            'source': 'eBay'
        }
    """
    select_fields = [] if select_fields is None else select_fields
    if number_of_documents is None:
        number_of_documents = 739
    return ExampleDatasets._get_dummy_dataset(
        "ecommerce_1", number_of_documents, select_fields
    )


def get_ecommerce_dataset_clean(
    number_of_documents: int = 1000, select_fields: Optional[List] = None
):
    """
    Download an example e-commerce dataset \n
    Total Len: 739 \n

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
        Fields to include in the dataset, empty array/list means all fields.

    Example
    -------
    .. code-block::

        {
            '_id': '711160239',
            'product_image': 'https://thumbs4.ebaystatic.com/d/l225/pict/321567405391_1.jpg',
            'product_link': 'https://www.ebay.com/itm/20-36-Mens-Silver-Stainless-Steel-Braided-Wheat-Chain-Necklace-Jewelry-3-4-5-6MM-/321567405391?pt=LH_DefaultDomain_0&var=&hash=item4adee9354f',
            'product_price': '$7.99 to $12.99',
            'product_title': '20-36Mens Silver Stainless Steel Braided Wheat Chain Necklace Jewelry 3/4/5/6MM"',
            'query': 'steel necklace',
            'source': 'eBay'
        }
    """
    select_fields = (
        [
            "_id",
            "product_image",
            "product_link",
            "product_title",
            "product_price",
            "query",
            "source",
        ]
        if select_fields is None
        else select_fields
    )
    if number_of_documents is None:
        number_of_documents = 1000
    documents = ExampleDatasets._get_dummy_dataset(
        "ecommerce_2", number_of_documents, select_fields
    )
    for d in documents:
        if "image_first" in d:
            d["image"] = d.pop("image_first")
    return documents


def get_online_retail_dataset(
    number_of_documents: Union[None, int] = 1000, select_fields: Optional[List] = None
) -> List:
    """
    Download an example online retail dataset from UCI machine learning \n
    Total Len: 541909 \n

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
        Fields to include in the dataset, empty array/list means all fields.

    Example
    -------
    .. code-block::

        {
            'Country': 'United Kingdom',
            'CustomerID': 17850.0,
            'Description': 'WHITE HANGING HEART T-LIGHT HOLDER',
            'InvoiceDate': Timestamp('2010-12-01 08:26:00'),
            'InvoiceNo': 536365,
            'Quantity': 6,
            'StockCode': '85123A',
            'UnitPrice': 2.55
        }
    """
    select_fields = [] if select_fields is None else select_fields
    if number_of_documents is None:
        number_of_documents = 1000
    return ExampleDatasets._get_online_dataset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
        number_of_documents,
        select_fields,
        csv=False,
    )


def get_news_dataset(
    number_of_documents: Union[None, int] = 250, select_fields: Optional[List] = None
) -> List:
    """
    Download an example news dataset \n
    Total Len: 250 \n

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
        Fields to include in the dataset, empty array/list means all fields.

    Example
    -------
    .. code-block::

        {
            'authors': 'Ruth Harris',
            'content': 'Sometimes the power of Christmas will make you do wild and wonderful things. You do not need to believe in the Holy Trinity to believe in the positive power of doing good for others.
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
            'url': 'http://awm.com/church-congregation-brings-gift-to-waitresses-working-on-christmas-eve-has-them-crying-video/'
        }
    """
    select_fields = [] if select_fields is None else select_fields
    if number_of_documents is None:
        number_of_documents = 250
    return ExampleDatasets._get_online_dataset(
        "https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv",
        number_of_documents,
        select_fields,
    )


def get_online_ecommerce_dataset(
    number_of_documents: Union[None, int] = 1000, select_fields: Optional[List] = None
) -> List:
    """
    Download an example ecommerce dataset (https://data.world/crowdflower/ecommerce-search-relevance) \n
    Total Len: 15528 \n

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
        Fields to include in the dataset, empty array/list means all fields.

    Example
    -------
    .. code-block::

        {
            '_unit_id': 711158459,
            'product_description': 'The PlayStation 4 system opens the door to an '
                                'incredible journey through immersive new gaming '
                                'worlds and a deeply connected gaming community. Step '
                                'into living, breathing worlds where you are hero of '
                                '...',
            'product_image': 'http://thumbs2.ebaystatic.com/d/l225/m/mzvzEUIknaQclZ801YCY1ew.jpg',
            'product_link': 'http://www.ebay.com/itm/Sony-PlayStation-4-PS4-Latest-Model-500-GB-Jet-Black-Console-/321459436277?pt=LH_DefaultDomain_0&hash=item4ad879baf5',
            'product_price': '$329.98 ',
            'product_title': 'Sony PlayStation 4 (PS4) (Latest Model)- 500 GB Jet Black 'Console'',
            'query': 'playstation 4',
            'rank': 1,
            'relevance': 3.67,
            'relevance:variance': 0.471,
            'source': 'eBay',
            'url': 'http://www.ebay.com/sch/i.html?_from=R40&_trksid=p2050601.m570.l1313.TR11.TRC1.A0.H0.Xplant.TRS0&_nkw=playstation%204'
        }
    """
    select_fields = [] if select_fields is None else select_fields
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
    documents = [
        {k: v for k, v in doc.items() if not pd.isna(v)}
        for doc in df.to_dict(orient="records")
    ]
    return documents


def get_flipkart_dataset(
    number_of_documents: Union[None, int] = 19920, select_fields: Optional[List] = None
) -> List:
    """
    Download an example flipkart ecommerce dataset \n
    Total Len: 19920 \n

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
        Fields to include in the dataset, empty array/list means all fields.

    Example
    -------
    .. code-block::

        {
            '_id': 0,
            'product_name': "Alisha Solid Women's Cycling Shorts",
            'description': "Key Features of Alisha Solid Women's Cycling Shorts Cotton Lycra Navy, Red, Navy,Specifications of Alisha Solid Women's Cycling Shorts Shorts Details Number of Contents in Sales Package Pack of 3 Fabric Cotton Lycra Type Cycling Shorts General Details Pattern Solid Ideal For Women's Fabric Care Gentle Machine Wash in Lukewarm Water, Do Not Bleach Additional Details Style Code ALTHT_3P_21 In the Box 3 shorts",
            'retail_price': 999.0
        }
    """
    select_fields = [] if select_fields is None else select_fields
    if number_of_documents is None:
        number_of_documents = 19920
    return ExampleDatasets._get_dummy_dataset(
        "dummy-flipkart",
        number_of_documents,
        select_fields,
    )


def get_realestate_dataset(
    number_of_documents: int = 50, select_fields: Optional[List] = None
):
    """
    Download an example real-estate dataset \n
    Total Len: 5885 \n

    Parameters
    ----------
    number_of_documents: int
        Number of documents to download
    select_fields : list
        Fields to include in the dataset, empty array/list means all fields.

    Example
    -------
    .. code-block::

        {
            'propertyDetails': {'area': 'North Shore - Lower',
            'carspaces': 1,
            'streetNumber': '28',
            'latitude': -33.8115768,
            'allPropertyTypes': ['ApartmentUnitFlat'],
            'postcode': '2066',
            'unitNumber': '6',
            'bathrooms': 1.0,
            'bedrooms': 1.0,
            'features': ['BuiltInWardrobes', 'InternalLaundry','Intercom', 'Dishwasher'],
            'street': 'Epping Road',
            'propertyType': 'ApartmentUnitFlat',
            'suburb': 'LANE COVE',
            'state': 'NSW',
            'region': 'Sydney Region',
            'displayableAddress': '6/28 Epping Road, Lane Cove',
            'longitude': 151.166611},
            'listingSlug': '6-28-epping-road-lane-cove-nsw-2066-14688794',
            'id': 14688794,
            'headline': 'Extra large one bedroom unit',
            'summaryDescription': '<b></b><br />This modern and spacious one-bedroom apartment situated on the top floor, the quiet rear side of a small 2 story boutique block, enjoys a wonderfully private, leafy, and greenly outlook from 2 sides and balcony. A short stroll to city buse...',
            'advertiser': 'Ray White Lane Cove',
            'image_url': 'https://bucket-api.domain.com.au/v1/bucket/image/14688794_1_1_201203_101135-w1600-h1065',
            'insert_date_': '2021-03-01T14:19:22.805086',
            'labels': [],
            'image_url_5': 'https://bucket-api.domain.com.au/v1/bucket/image/14688794_5_1_201203_101135-w1600-h1067',
            'image_url_4': 'https://bucket-api.domain.com.au/v1/bucket/image/14688794_4_1_201203_101135-w1600-h1067',
            'priceDetails': {'displayPrice': 'Deposit Taken ! Inspection Cancelled thank you !!!'}
        ...
        }
    """
    select_fields = [] if select_fields is None else select_fields
    if number_of_documents is None:
        number_of_documents = 50

    documents = ExampleDatasets._get_dummy_dataset(
        "realestate2", number_of_documents, select_fields
    )

    # todo: fix the clustering results in the original dataset
    # insert_documents fails if they are included
    for doc in documents:
        if "_clusters_" in doc:
            del doc["_clusters_"]

    return documents


def mock_documents(number_of_documents: int = 100, vector_length=5):
    """
    Utility function to mock documents. Aimed at helping users reproduce errors
    if required.
    The schema for the documents is as follows:

    .. code-block::

        {'_chunk_': 'chunks',
        '_chunk_.label': 'text',
        '_chunk_.label_chunkvector_': {'chunkvector': 5},
        'insert_date_': 'date',
        'sample_1_description': 'text',
        'sample_1_label': 'text',
        'sample_1_value': 'numeric',
        'sample_1_vector_': {'vector': 5},
        'sample_2_description': 'text',
        'sample_2_label': 'text',
        'sample_2_value': 'numeric',
        'sample_2_vector_': {'vector': 5},
        'sample_3_description': 'text',
        'sample_3_label': 'text',
        'sample_3_value': 'numeric',
        'sample_3_vector_': {'vector': 5}}

    Parameters
    ------------

    number_of_documents: int
        The number of documents to mock
    vector_length: int
        The length of vectors

    .. code-block::

        from relevanceai.package_utils.datasets import mock_documents
        documents = mock_documents(10)

    """

    def generate_random_string(string_length: int = 5) -> str:
        """Generate a random string of letters and numbers"""
        return "".join(
            random.choice(string.ascii_uppercase + string.digits)
            for _ in range(string_length)
        )

    def generate_random_vector(vector_length: int = vector_length) -> List[float]:
        """Generate a random list of floats"""
        return [random.random() for _ in range(vector_length)]

    def generate_random_label(label_value: int = 5) -> str:
        return f"label_{random.randint(0, label_value)}"

    def generate_random_integer(min: int = 0, max: int = 100) -> int:
        return random.randint(min, max)

    def vector_document() -> Dict:
        document = {
            "sample_1_label": generate_random_label(),
            "sample_2_label": generate_random_label(),
            "sample_3_label": generate_random_label(),
            "sample_1_description": generate_random_string(),
            "sample_2_description": generate_random_string(),
            "sample_3_description": generate_random_string(),
            "sample_1_vector_": generate_random_vector(),
            "sample_2_vector_": generate_random_vector(),
            "sample_3_vector_": generate_random_vector(),
            "sample_1_value": generate_random_integer(),
            "sample_2_value": generate_random_integer(),
            "sample_3_value": generate_random_integer(),
            "_chunk_": [
                {
                    "label": generate_random_label(),
                    "label_chunkvector_": generate_random_vector(),
                }
            ],
        }

        from relevanceai.utils import make_id

        document["_id"] = make_id(document)
        return document

    return [vector_document() for _ in range(number_of_documents)]


def get_titanic_dataset(
    output_format: Literal["pandas_dataframe", "json", "csv"] = "json"
):
    """
    Titanic Dataset.

    # Sample document
    {'Unnamed: 0': 0,
    'PassengerId': 892,
    'Survived': 0,
    'Pclass': 3,
    'Age': 34.5,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.8292,
    'male': 1,
    'Q': 1,
    'S': 0,
    'value_vector_': '[3.0, 34.5, 0.0, 0.0, 7.8292, 1.0, 1.0, 0.0]'}
    """
    FN = "https://gist.githubusercontent.com/boba-and-beer/0bf5f7840a856f2d2adb2b80f96db481/raw/0891fed9c19ddc07e3393ee4127aa3b9d809b4f2/titanic_train_data.csv"
    if output_format == "csv":
        return FN
    df = pd.read_csv(FN)
    if output_format == "pandas_dataframe":
        return df
    docs = df.to_dict(orient="records")
    for d in docs:
        d["value_vector_"] = eval(d["value_vector_"])
    return docs


def get_coco_dataset(
    number_of_documents: int = 1000,
    include_vector: bool = True,
    select_fields: Optional[list] = None,
):
    """
    Get the coco dataset
    """
    select_fields = [] if select_fields is None else select_fields
    if number_of_documents is None:
        number_of_documents = 50

    documents = ExampleDatasets._get_dummy_dataset(
        "toy_image_caption_coco_image_encoded",
        number_of_documents,
        select_fields,
        include_vector=include_vector,
    )

    # todo: fix the clustering results in the original dataset
    # insert_documents fails if they are included
    for doc in documents:
        if "_clusters_" in doc:
            del doc["_clusters_"]

    return documents


def get_palmer_penguins_dataset(
    number_of_documents: int = None,
    select_fields: Optional[List] = None,
    shuffle: bool = True,
) -> List[Dict]:
    adelie_data = ExampleDatasets._get_online_dataset(
        url="https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.219.3&entityid=002f3893385f710df69eeebe893144ff",
        number_of_documents=number_of_documents,
        select_fields=select_fields,
    )
    gentoo_data = ExampleDatasets._get_online_dataset(
        url="https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.220.3&entityid=e03b43c924f226486f2f0ab6709d2381",
        number_of_documents=number_of_documents,
        select_fields=select_fields,
    )
    chinstrap_data = ExampleDatasets._get_online_dataset(
        url="https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.221.2&entityid=fe853aa8f7a59aa84cdd3197619ef462",
        number_of_documents=number_of_documents,
        select_fields=select_fields,
    )
    data = adelie_data + gentoo_data + chinstrap_data
    if shuffle:
        random.shuffle(data)
    return data


def get_iris_dataset(
    number_of_documents: int = None,
    select_fields: Optional[List] = None,
    shuffle: bool = True,
) -> List[Dict]:
    iris_data = ExampleDatasets._get_online_dataset(
        url="https://raw.githubusercontent.com/venky14/Machine-Learning-with-Iris-Dataset/master/Iris.csv",
        number_of_documents=number_of_documents,
        select_fields=select_fields,
    )
    if shuffle:
        random.shuffle(iris_data)
    return iris_data


### For backwards compatability

get_ecommerce_1_dataset = get_dummy_ecommerce_dataset = get_ecommerce_dataset_encoded
get_ecommerce_2_dataset = get_sample_ecommerce_dataset = get_ecommerce_dataset_clean
get_ecommerce_3_dataset = get_ecommerce_dataset = get_online_ecommerce_dataset
