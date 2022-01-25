:py:mod:`relevanceai.datasets`
==============================

.. py:module:: relevanceai.datasets

.. autoapi-nested-parse::

   Datasets to mock



Module Contents
---------------

.. py:data:: THIS_MODULE
   

   

.. py:data:: DATASETS
   :annotation: = ['games', 'ecommerce_1', 'ecommerce_2', 'ecommerce_3', 'online_retail', 'news', 'flipkart', 'realestate']

   

.. py:function:: select_fields_from_json(json, select_fields)


.. py:class:: ExampleDatasets

   .. py:method:: list_datasets(self)

      List of example datasets available to download


   .. py:method:: get_dataset(self, name, number_of_documents=None, select_fields=[])

      Download an example dataset
      :param name: Name of example dataset
      :type name: string
      :param number_of_documents: Number of documents to download
      :type number_of_documents: int
      :param select_fields: Fields to include in the dataset, empty array/list means all fields.
      :type select_fields: list



.. py:function:: get_games_dataset(number_of_documents: Union[None, int] = 365, select_fields: list = []) -> List

   Download an example games dataset (https://www.freetogame.com/)

   Total Len: 365

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

   :param number_of_documents: Number of documents to download
   :type number_of_documents: int
   :param select_fields: Fields to include in the dataset, empty array/list means all fields.
   :type select_fields: list


.. py:function:: get_ecommerce_1_dataset(number_of_documents: int = 739, select_fields: list = []) -> List[Dict[Any, Any]]

   Download an example e-commerce dataset

   Total Len: 739

   Sample document:

   >>> {'_id': 'b7fc9acbc9ddd18855f96863d37a4fe9',
   >>> 'product_image': 'https://thumbs4.ebaystatic.com/d/l225/pict/321567405391_1.jpg',
   >>> 'product_image_clip_vector_': [...],
   >>> 'product_link': 'https://www.ebay.com/itm/20-36-Mens-Silver-Stainless-Steel-Braided-Wheat-Chain-Necklace-Jewelry-3-4-5-6MM-/321567405391?pt=LH_DefaultDomain_0&var=&hash=item4adee9354f',
   >>> 'product_price': '$7.99 to $12.99',
   >>> 'product_title': '20-36Mens Silver Stainless Steel Braided Wheat Chain Necklace Jewelry 3/4/5/6MM"',
   >>> 'product_title_clip_vector_': [...],
   >>> 'query': 'steel necklace',
   >>> 'source': 'eBay'
   >>> }

   :param number_of_documents: Number of documents to download
   :type number_of_documents: int
   :param select_fields: Fields to include in the dataset, empty array/list means all fields.
   :type select_fields: list


.. py:function:: get_ecommerce_2_dataset(number_of_documents: int = 1000, select_fields: list = ['_id', 'product_image', 'product_link', 'product_title', 'product_price', 'query', 'source'])

   Download an example e-commerce dataset

   Total Len: 739

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
   >>> 'product_description': 'eBay item number:321567405391


       Seller assumes all responsibility for this listing
   >>> ...
   >>> }

   :param number_of_documents: Number of documents to download
   :type number_of_documents: int
   :param select_fields: Fields to include in the dataset, empty array/list means all fields.
   :type select_fields: list


.. py:function:: get_online_retail_dataset(number_of_documents: Union[None, int] = 1000, select_fields: list = []) -> List

   Download an example online retail dataset from UCI machine learning

   Total Len: 541909


   Sample document:

   >>> {'Country': 'United Kingdom',
   >>> 'CustomerID': 17850.0,
   >>> 'Description': 'WHITE HANGING HEART T-LIGHT HOLDER',
   >>> 'InvoiceDate': Timestamp('2010-12-01 08:26:00'),
   >>> 'InvoiceNo': 536365,
   >>> 'Quantity': 6,
   >>> 'StockCode': '85123A',
   >>> 'UnitPrice': 2.55}

   :param number_of_documents: Number of documents to download
   :type number_of_documents: int
   :param select_fields: Fields to include in the dataset, empty array/list means all fields.
   :type select_fields: list


.. py:function:: get_news_dataset(number_of_documents: Union[None, int] = 250, select_fields: list = []) -> List

   Download an example news dataset

   Total Len: 250

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

   :param number_of_documents: Number of documents to download
   :type number_of_documents: int
   :param select_fields: Fields to include in the dataset, empty array/list means all fields.
   :type select_fields: list


.. py:function:: get_ecommerce_3_dataset(number_of_documents: Union[None, int] = 1000, select_fields: list = []) -> List

   Download an example ecommerce dataset (https://data.world/crowdflower/ecommerce-search-relevance)

   Total Len: 15528

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

   :param number_of_documents: Number of documents to download
   :type number_of_documents: int
   :param select_fields: Fields to include in the dataset, empty array/list means all fields.
   :type select_fields: list


.. py:function:: get_flipkart_dataset(number_of_documents: Union[None, int] = 19920, select_fields: list = []) -> List

   Download an example flipkart ecommerce dataset

   Total Len: 19920

   Sample document:

   >>> {'_id': 0,
   >>> 'product_name': "Alisha Solid Women's Cycling Shorts",
   >>> 'description': "Key Features of Alisha Solid Women's Cycling Shorts Cotton Lycra Navy, Red, Navy,Specifications of Alisha Solid Women's Cycling Shorts Shorts Details Number of Contents in Sales Package Pack of 3 Fabric Cotton Lycra Type Cycling Shorts General Details Pattern Solid Ideal For Women's Fabric Care Gentle Machine Wash in Lukewarm Water, Do Not Bleach Additional Details Style Code ALTHT_3P_21 In the Box 3 shorts",
   >>> 'retail_price': 999.0}

   :param number_of_documents: Number of documents to download
   :type number_of_documents: int
   :param select_fields: Fields to include in the dataset, empty array/list means all fields.
   :type select_fields: list


.. py:function:: get_realestate_dataset(number_of_documents: int = 50, select_fields: list = [])

   Download an example real-estate dataset

   Total Len: 5885

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

   :param number_of_documents: Number of documents to download
   :type number_of_documents: int
   :param select_fields: Fields to include in the dataset, empty array/list means all fields.
   :type select_fields: list


.. py:function:: get_mission_statements_dataset(number_of_documents: Union[None, int] = 1433, select_fields: list = []) -> List

   Function to download a sample company mission statement dataset.
   Total Len: 1433
   Sample document:
   {'_id': 0,
   'company': 'Starbucks',
   'text': 'Establish Starbucks as the premier purveyor of the finest coffee in the world while maintaining our uncompromising principles while we grow.'},


.. py:function:: get_machine_learning_research_dataset()

   Here we get our Machine Learning research dataset.


.. py:data:: get_dummy_ecommerce_dataset
   

   

.. py:data:: get_sample_ecommerce_dataset
   

   

.. py:data:: get_ecommerce_dataset
   

   

