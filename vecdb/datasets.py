# -*- coding: utf-8 -*-
"""
Datasets to mock
"""
from typing import List, Union, Dict, Any, Literal, Callable
import collections.abc


import pandas as pd
import requests
from vecdb_logging import logger

JSONDict = Dict[str, Any]




def get_games_dataset(
    number_of_documents: Union[None, int] = 365
) -> List[JSONDict]:
    """Function to download a sample games dataset.
    Dataset from https://www.freetogame.com/
    Total Len: 365
    Sample document:
    {'id': 1,
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
    data = requests.get('https://www.freetogame.com/api/games').json()
    if number_of_documents:
        data = data[:number_of_documents]
    return data


def get_dummy_ecommerce_dataset(
    number_of_documents: Union[None, int] = 100,
    db_name: Literal['ecommerce-5', 'ecommerce-6'] = 'ecommerce-5',
    base_url='https://api-aueast.relevance.ai/v1/',
) -> List[JSONDict]:
    """Function to a dummy ecommerce dataset from VecDB dummy collections
    Dataset from ecommerce-5
    Total Len: 14058
    Sample document:
    ### Real vector values are vectors of 512 length - min, mean, max is shown below for brevity
    {
        "_id": "b7fc9acbc9ddd18855f96863d37a4fe9",
        "uniq_id": "b7fc9acbc9ddd18855f96863d37a4fe9",
        "crawl_timestamp": "2016-04-24 18:34:50 +0000",
        "product_url": "http://www.flipkart.com/babeezworld-baby-boy-s-romper/p/itmehyhguebbzb6h?pid=DRPEHYHGNAF5UYUQ",
        "product_name": "Babeezworld Baby Boy's Romper",
        "product_category_tree": "[\"Baby Care >> Infant Wear >> Baby Boys' Clothes >> Dungarees & Jumpsuits >> Dungarees >> Babeezworld Dungarees >> Babeezworld Baby Boy's Romper\"]",
        "pid": "DRPEHYHGNAF5UYUQ",
        "retail_price": 999,
        "discounted_price": 499,
        "image": "[\"http://img5a.flixcart.com/image/dungaree-romper/x/f/r/1012blue-yellow-3-6-babeezworld-3-6-months-original-imaehydgqkkadjud.jpeg\", \"http://img5a.flixcart.com/image/dungaree-romper/x/f/r/1012blue-yellow-3-6-babeezworld-3-6-months-original-imaehbf3h3jsmzhb.jpeg\", \"http://img5a.flixcart.com/image/dungaree-romper/x/f/r/1012blue-yellow-3-6-babeezworld-3-6-months-original-imaehbf4nupsmhzt.jpeg\", \"http://img5a.flixcart.com/image/dungaree-romper/z/n/u/1012blue-yellow-6-9-babeezworld-6-9-months-original-imaehbfbkynega2z.jpeg\", \"http://img6a.flixcart.com/image/dungaree-romper/z/n/u/1012blue-yellow-6-9-babeezworld-6-9-months-original-imaehbfbjjffht4e.jpeg\"]",
        "is_FK_Advantage_product": false,
        "description": "Key Features of Babeezworld Baby Boy's Romper Fabric: Cotton Brand Color: Blue:Yellow,Babeezworld Baby Boy's Romper Price: Rs. 499 Babeezworld presents a cute baby dungaree set for your little one.This dungaree set comes with a trendy round neck cotton t-shirt with shoulder loops for ease and comfort fit. Set is made of soft cotton material and has adjustable straps with two button closures. The front has a special character which gives it a stylish look. This dungaree set is an ideal pick for this summer and is available in multiple colors.,Specifications of Babeezworld Baby Boy's Romper Top Details Sleeve Half Sleeve Number of Contents in Sales Package Pack of 2 Fabric Cotton Type Romper Neck Round Neck General Details Pattern Printed Ideal For Baby Boy's Fabric Care Wash with Similar Colors, Use Detergent for Colors",
        "product_rating": "No rating available",
        "overall_rating": "No rating available",
        "brand": "Babeezworld",
        "product_specifications": "{\"product_specification\"=>[{\"key\"=>\"Sleeve\", \"value\"=>\"Half Sleeve\"}, {\"key\"=>\"Number of Contents in Sales Package\", \"value\"=>\"Pack of 2\"}, {\"key\"=>\"Fabric\", \"value\"=>\"Cotton\"}, {\"key\"=>\"Type\", \"value\"=>\"Romper\"}, {\"key\"=>\"Neck\", \"value\"=>\"Round Neck\"}, {\"key\"=>\"Pattern\", \"value\"=>\"Printed\"}, {\"key\"=>\"Ideal For\", \"value\"=>\"Baby Boy's\"}, {\"value\"=>\"Wash with Similar Colors, Use Detergent for Colors\"}]}",
        "image_first": "http://img5a.flixcart.com/image/dungaree-romper/x/f/r/1012blue-yellow-3-6-babeezworld-3-6-months-original-imaehydgqkkadjud.jpeg",
        "category": "Baby Care ",
        "insert_date_": "2021-08-13T11:38:52.110Z",
        "product_name_default_vector_": [
            -0.09005960822105408,
            -0.00011151187237601334,
            0.09022483229637146
        ],
        "product_category_tree_default_vector_": [
            -0.06799716502428055,
            6.681176930101174e-05,
            0.06799962371587753
        ],
        "description_default_vector_": [
            -0.0638396292924881,
            0.002004953355722705,
            0.06384047120809555
        ],
        "product_specifications_default_vector_": [
            -0.0928298607468605,
            0.0010741057754444228,
            0.0936967059969902
        ],
        "image_first_default_vector_": [
            -5.581348419189453,
            -0.004497126452065459,
            1.9208835363388062
        ],
        "product_nametextmulti_vector_": [
            -0.12710408866405487,
            0.00036892396830268126,
            0.1368495225906372
        ],
        "descriptiontextmulti_vector_": [
            -0.12531952559947968,
            0.000991348989572316,
            0.12568922340869904
        ],
        "product_name_imagetext_vector_": [
            -0.8955078125,
            0.008709213696420193,
            3.013671875
        ],
        "description_imagetext_vector_": [
            1e-07,
            9.999999999999997e-08,
            1e-07
        ],
        "description_chinese": "巴贝兹世界男婴的隆珀面料的主要特点：棉花品牌颜色：蓝色：黄色，巴贝兹世界男婴的隆珀价格：Rs. 499巴贝兹世界为您的小家伙呈现一个可爱的婴儿粪便集。这套dungaree配备了一个时尚的圆脖子棉T恤与肩环的轻松和舒适适合。套装由柔软的棉质材料制成，带可调节带，有两个按钮关闭。前面有一个特殊的性格，给它一个时尚的外观。这套 dungaree 套装是今年夏天的理想选择，有多种颜色可供选择。",
        "product_name_chinese": "巴贝兹世界男婴的隆珀",
        "product_name_chinese_textmulti_vector_": [
            -0.11721953004598618,
            4.798636290104241e-05,
            0.1346004456281662
        ],
        "description_chinese_textmulti_vector_": [
            -0.11921291798353195,
            0.0002485030701109281,
            0.11918215453624725
        ]
    },
    """
    # def _apply_recursive(obj, func, keys=Union[None, List[str]]):
    #     '''
    #     Applies a function in-place to a nested dict or list recursively to a select group of keys
    #     '''
    #     if isinstance(obj, dict):
    #         for k, v in obj.iteritems():
    #             if isinstance(v, dict):
    #                 if keys is None: obj.keys()
    #                 if k in keys:
    #                     _apply_recursive(v, func, keys)
    #     elif isinstance(v, list):
    #         obj[k] = map(func, v)
    #     else:
    #         obj[k] = func(v)

    # def _apply_recursive(obj, func, keys=Union[None, List[str]]):
    #     if isinstance(obj, dict):  # if dict, apply to each key
    #         if keys is None: obj.keys()
    #         return {k: _apply_recursive(func, v) for k, v in obj.items() if k in keys}
    #     elif isinstance(obj, list):  # if list, apply to each element
    #         return [_apply_recursive(func, elem) for elem in obj]
    #     else:
    #         return func(obj)
    def _apply_fn_dict_value(
        doc: JSONDict,
        func: Callable,
        keys: Union[None, List[str]] = None
    ) -> JSONDict:
        if keys is None: keys = doc.keys()
        for k, v in doc.items():
            if k in keys:
                doc[k] = func(v)
        return doc


    from http_client import VecDBClient

    project = 'dummy-collections'
    api_key = (
        'UzdYRktIY0JxNmlvb1NpOFNsenU6VGdTU0s4UjhUR0NsaDdnQTVwUkpKZw'    # Read access
    )
    vi = VecDBClient(project, api_key, base_url=base_url)
    page_size = number_of_documents if (number_of_documents and number_of_documents <=1000) else 1000
    resp = vi.datasets.documents.list(db_name, page_size=page_size)     # Initial test
    '''
    Paginating the dataset
    '''
    data = resp['documents']
    data = [_apply_fn_dict_value(doc=doc,
                            func=lambda x: x.replace('http://', 'https://'),
                            keys=['product_url', 'image', 'image_first'])
            for doc in data]
    _cursor = resp['cursor']
    while _cursor:
        resp = vi.datasets.documents.list(db_name, page_size=page_size, cursor=_cursor)
        if 'message' in resp:
            import warnings
            warnings.warn(resp['message'])
        _data = resp['documents']
        _cursor = resp['cursor']
        if (_data == []) or (_cursor is []): break
        _data = [_apply_fn_dict_value(doc=doc,
                            func=lambda x: x.replace('http://', 'https://'),
                            keys=['product_url', 'image', 'image_first'])
                for doc in _data]
        data += _data
        if (number_of_documents and (len(data) >= int(number_of_documents))): break
    return data


def get_online_retail_dataset(
    number_of_documents: Union[None, int] = 100
) -> List[JSONDict]:
    """Online retail dataset from UCI machine learning
    Total Len: 406829
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
    return (
        pd.read_excel(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx',
        )
        .dropna()
        .iloc[:number_of_documents, :]
        .to_dict(orient='records')
    )


def get_news_dataset(
    number_of_documents: Union[None, int] = 250
) -> List[JSONDict]:
    """News dataset
    Total Len: 250
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
    return (
        pd.read_csv(
            'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv',
            index_col=0
        )
        .iloc[:number_of_documents, :]
        .to_dict(orient='records')
    )


def get_ecommerce_dataset(
    number_of_documents: Union[None, int] = 1000
) -> List[JSONDict]:
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
    df = (
        pd.read_csv(
            'https://query.data.world/s/glc7oe2ssd252scha53mu7dy2e7cft',
            encoding='ISO-8859-1',
        )
        .dropna()
        .iloc[:number_of_documents, :]
    )
    df['product_image'] = df['product_image'].str.replace('http://', 'https://')
    df['product_link'] = df['product_link'].str.replace('http://', 'https://')
    df['url'] = df['url'].str.replace('http://', 'https://')
    return df.to_dict('records')


from pprint import pprint

data = get_dummy_ecommerce_dataset(number_of_documents=10)
pprint(data[0]['image'])
print(len(data))
