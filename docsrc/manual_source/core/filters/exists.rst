Filters (Complex)
===================

.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/filters-1.png?raw=true" width="1009" alt="604547f-combined_filters.png" />
.. <figcaption>Example output of filtering Lenovo products all inserted into the database after 01/01/2020</figcaption>
.. <figure>

Filters are great tools to retrieve a subset of documents whose data match the criteria specified in the filter.
For instance, in an e-commerce dataset, we can retrieve all products:
* with prices between 200 and 300 dollars
* with the phrase "free return" included in `description` field
* that are produced after January 2020

📘 Filters help us find what we need.

Filters are great tools to retrieve a subset of documents whose data match certain criteria. This allows us to have a more fine-grained overview of the data since only documents that meet the filtering criteria will be displayed.

How to form a filter?

Filters at Relevance AI are defined as Python dictionaries with four main keys:
- `field` (i.e. the data filed in the document you want to filter on)
- `condition` (i.e. operators such as greater than or equal)
- `filter_type` (i.e. the type of filter you want to apply - whether it be date/numeric/text etc.)
- `condition_value` (dependent on the filter type but decides what value to filter on)


.. code-block::

    filter = [
        {
            "field": description,
            "filter_type": contains,
            "condition": ==,
            "condition_value": Durian Club
        }
    ]

Filtering operators
======================

Relevance AI covers all common operators:
* "==" (a == b, a equals b)
* "!="  (a != b, a not equals b)
* ">=" (a >= b, a greater that or equals b)
* ">"   (a > b, a greater than b)
* "<"   (a < b, a smaller than b)
* "<=" (a <= b, a smaller than or equals b)

Filter types

Supported filter types at Relevance AI are listed below.

* contains
* exact_match
* word_match
* categories
* exists
* date
* numeric
* ids
* support for mixing together multiple filters such as in OR situations

We will explain each filter type followed by a sample code snippet in the next pages. There is also a 
:ref:`guide <https://docs.relevance.ai/docs/combining-filters-and-vector-search>` on how to combine filters and vector search.

.. image:: https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/exists.png?raw=true
  :width: 400
  :alt: Alternative text

Exists
==========

This filter returns entries in a database if a certain field (as opposed to the field values in previously mentioned filter types) exists or doesn't exist in them. For instance, filtering out documents in which there is no field 'purchase-info'. *Note that this filter is case-sensitive.*

You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
Once you have signed up, click on the value under `Activation token` and paste it here

.. code-block:: python

    from relevanceai import Client
    client = Client()

    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)

    filters = [
        {
            "field": "brand",
            "filter_type": "exists",
            "condition": "==",
            "condition_value": " "
        }
    ]

    filtered_data = ds.get_documents(filters)

.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/exact-match.png?raw=true" width="2062" alt="Exact match.png" />
.. <figcaption>Filtering documents with "Durian Leather 2 Seater Sofa" as the product_name.</figcaption>
.. <figure>

Exact Match
==============

This filter works with string values and only returns documents with a field value that exactly matches the filtered criteria. For instance under filtering by 'Samsung galaxy s21', the result will only contain products explicitly having 'Samsung galaxy s21' in their specified field. *Note that this filter is case-sensitive.*

.. code-block::

    from relevanceai import Client


    DATASET_ID = "ecommerce-sample-dataset"
    df = client.Dataset(DATASET_ID)

    filter = [
        {
            "field": product_name,
            "filter_type": exact_match,
            "condition": ==,
            "condition_value": Durian Leather 2 Seater Sofa
        }
    ]

    filtered_data = ds.get_where(filter)

.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/category.png?raw=true" width="658" alt="categories.png" />
.. <figcaption>Filtering documents with "LG" or "Samsung" as the brand.</figcaption>
.. <figure>

Categories 
==============

This filter checks the entries in a database and returns ones in which a field value exists in a given filter list. For instance, if the product name is any of Sony, Samsung, or LG. *Note that this filter is case-sensitive.*

.. code-block::

    filter = [
        {
            "field": brand,
            "filter_type": categories,
            "condition": >=,
            "condition_value": ['LG', 'samsung']
        }
    ]

    filtered_data = ds.get_where(filter)

.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/contains.png?raw=true" width="2048" alt="contains.png" />
.. <figcaption>Filtering documents containing "Durian BID" in description using filter_type `contains`.</figcaption>
.. <figure>


Contains
============

This filter returns a document only if it contains a string value. Note that substrings are covered in this category. For instance, if a product name is composed of a name and a number (e.g. ABC-123), one might remember the name but not the number. This filter can easily return all products including the ABC string.
*Note that this filter is case-sensitive.*

You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
Once you have signed up, click on the value under `Activation token` and paste it here

.. code-block::

    from relevanceai import Client
    client = Client()

    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)

    filter = [
        {
            "field": "description",
            "filter_type": "contains",
            "condition": "==",
            "condition_value": "Durian BID"
        }
    ]

    filtered_data = ds.get_where(filter)


.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/date.png?raw=true" width="600"  alt="date.png" />
.. <figcaption>Filtering documents which were added to the database after January 2021.</figcaption>
.. <figure>

Date
============

This filter performs date analysis and filters documents based on their date information. For instance, it is possible to filter out any documents with a production date before January 2021.

.. code-block::

    from relevanceai import Client
    client = Client()

    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)

    filter = [
        {
            "field": "insert_date_",
            "filter_type": "date",
            "condition": "==",
            "condition_value": "2020-07-01"
        }
    ]

Note that the default format is "yyyy-mm-dd" but can be changed to "yyyy-dd-mm" through the `format` parameter as shown in the example below.

.. code-block::

    filters = [
        {
            "field": "insert_date_",
            "filter_type": "date",
            "condition": "==",
            "condition_value": "2020-07-01",
            "format": "yyyy-dd-MM"
        }
    ]

    filtered_data = ds.get_documents(filters)

.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/word-match.png?raw=true" width="1974" alt="wordmatch.png" />
.. <figcaption>Filtering documents matching "Home curtain" in the description field.</figcaption>
.. <figure>

Word Match
============

This filter has similarities to both `exact_match` and `contains`. It returns a document only if it contains a **word** value matching the filter; meaning substrings are covered in this category but as long as they can be extracted with common word separators like the white-space (blank). For instance, the filter value "Home Gallery",  can lead to extraction of a document with "Buy Home Fashion Gallery Polyester ..." in the description field as both words are explicitly seen in the text. *Note that this filter is case-sensitive.*

.. code-block:: 

    from relevanceai import Client
    client = Client()

    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)

    filter = [
        {
            "field": description,
            "filter_type": "word_match",
            "condition": "==",
            "condition_value": "Home curtain"
        }
    ]

    filtered_data = ds.get_where(filter)


.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/id.png?raw=true" width="612" alt="id.png" />
.. <figcaption>Filtering documents based on their id.</figcaption>
.. <figure>

IDs
============

This filter returns documents whose unique id exists in a given list. It may look similar to 'categories'. The main difference is the search speed.

.. code-block::

    from relevanceai import Client
    client = Client()

    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)

    filter = [
        {
            "field": _id,
            "filter_type": ids,
            "condition": ==,
            "condition_value": 7790e058cbe1b1e10e20cd22a1e53d36
        }
    ]

    filtered_data = ds.get_documents(filter)

Numeric
============

.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/numeric.png?raw=true" width="446" alt="Numeric.png" />
.. <figcaption>Filtering documents with retail price higher than 5000.</figcaption>
.. <figure>

This filter is to perform the filtering operators on a numeric value. For instance, returning the documents with a price larger than 1000 dollars.

You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
Once you have signed up, click on the value under `Activation token` and paste it here

.. code-block::

    from relevanceai import Client
    client = Client()

    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)

    filter = [
        {
            "field": "retail_price",
            "filter_type": "numeric",
            "condition": ">",
            "condition_value": 5000
        }
    ]

    filtered_data = ds.get_documents(filter)

Or
===========

The `or` filter helps you filter for multiple conditions. Unlike other filters, the only values used here are `filter_type` and `condition_value`.

.. code-block::

    from relevanceai import Client
    client = Client()

    filters = [
        {
        'filter_type' : 'or',
        "condition_value": [
            {
                'field' : 'price',
                'filter_type' : 'numeric',
                "condition":"<=", "condition_value":90
            },
            {
                'field' : 'price',
                'filter_type' : 'numeric',
                "condition":">=",
                "condition_value": 150
            }
        ]}
    ]

    filtered_data = df.get_documents(filter)

(A or B) and (C or D)
------------------------

Below, we show an example of how to use 2 lists of filters with `or` logic.

.. code-block::

    from relevanceai import Client
    client = Client()

    filter = [{
        'filter_type' : 'or',
        "condition_value": [
            {
                'field' : 'price',
                'filter_type' : 'numeric',
                "condition":"<=",
                "condition_value":90
            },
            {
                'field' : 'price',
                'filter_type' : 'numeric',
                "condition":">=",
                "condition_value": 150
            }
        ]},
        'filter_type' : 'or',
        "condition_value": [
            {
                'field' : 'animal',
                'filter_type' : 'category',
                "condition":"==",
                "condition_value":"cat"
            },
            {
                'field' : 'animal',
                'filter_type' : 'category',
                "condition":"==",
                "condition_value": "dog"
            }
        ]}
    ]

    filtered_data = ds.get_where(filter)

(A or B or C) and D
-------------------------

Below, we show an example of how to use 2 lists of filters with `or` logic.

.. code-block::

    from relevanceai import Client
    client = Client()

    filter = [{
        'filter_type' : 'or',
        "condition_value": [
            {
                'field' : 'price',
                'filter_type' : 'numeric',
                "condition":"<=",
                "condition_value":90
            },
            {
                'field' : 'price',
                'filter_type' : 'numeric',
                "condition":">=",
                "condition_value": 150
            },
            {
                'field' : 'value',
                'filter_type' : 'numeric',
                "condition":">=",
                "condition_value": 2
            },
            ],
            {
                'field' : 'animal',
                'filter_type' : 'category',
                "condition":"==",
                "condition_value":"cat"
            },
    ]

    filtered_data = ds.get_documents(filter)

Regex
=========

.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/regex.png?raw=true" width="2048" alt="7cbd106-contains.png" />
.. <figcaption>Filtering documents containing "Durian (\w+)" in description using filter_type `regexp`.</figcaption>
.. <figure>

This filter returns a document only if it matches regexp (i.e. regular expression). Note that substrings are covered in this category. For instance, if a product name is composed of a name and a number (e.g. ABC-123), one might remember the name but not the number. This filter can easily return all products including the ABC string.

Relevance AI has the same regular expression schema as Apache Lucene's ElasticSearch to parse queries.

*Note that this filter is case-sensitive.*

.. code-block::

    from relevanceai import Client
    client = Client()

    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)

    filter = [
        {
            "field": description,
            "filter_type": regexp,
            "condition": ==,
            "condition_value": .*Durian (\w+)
        }
    ]
    filtered_data = ds.get_where(filter)


.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/multiple-filters.png?raw=true" width="1009" alt="combined filters.png" />
.. <figcaption>Filtering results when using multiple filters: categories, contains, and date.</figcaption>
.. <figure>

Combining filters
=====================

It is possible to combine multiple filters. For instance, the sample code below shows a filter that searches for

* a Lenovo flip cover  
* produced after January 2020 
* by either Lapguard or 4D brand.  
A screenshot of the results can be seen on top.  


You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
Once you have signed up, click on the value under `Activation token` and paste it here


.. code-block::

    from relevanceai import Client
    client = Client()


    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)
    filter = [
        {
            "field": description,
            "filter_type" : contains,
            "condition": ==,
            "condition_value": Lenovo
        },
        {
            "field" : brand,
            "filter_type" : categories,
            "condition": ==,
            "condition_value": ['Lapguard', '4D']
        },
        {
            "field" : "insert_date_",
            "filter_type" : date,
            "condition": >=,
            "condition_value": 2020-01-01
        }
    ]

    filtered_data = ds.get_where(filter)

.. <figure>
.. <img src="https://github.com/RelevanceAI/RelevanceAI-readme-docs/blob/v1.4.3/docs_template/GENERAL_FEATURES/_assets/combine.png?raw=true" width="1014" alt="filter+vectors.png" />
.. <figcaption>Including filters in a vector search.</figcaption>
.. <figure>

Including filters in vector search
======================================

Filtering provides you with a subset of a database containing data entities that match the certain criteria set as filters. What if we need to search through this subset? The difficult way is to ingest (save) the subset as a new dataset, then make the search on the new dataset. However, RelevanceAI has provided the filtering option in almost all search endpoints. This makes the whole process much faster and more straightforward.
In the code snippet below we show a hybrid search sample which is done on a subset of a huge database via filtering. In this scenario, the user is looking for white sneakers but only the ones produced after mid-2020 and from two brands Nike and Adidas.

Note that the code below needs
1. Relevance AI's Python SDK to be installed.
2. A dataset named `ecommerce-search-example`
3. Vectorized description saved under the name `descriptiontextmulti_vector_`

Please refer to a full guide on how to [create and upload a database](doc:creating-a-dataset) and how to use vectorizers to update a dataset with vectors at [How to vectorize](doc:vectorize-text).

.. code-block::

    from relevanceai import Client
    client = Client()
    DATASET_ID = "ecommerce-sample-dataset"
    ds = client.Dataset(DATASET_ID)
    query = "white sneakers"
    query_vec_txt = "enc_imagetext".encode(query)

    filter = [
        {
            "field" : "brand",
            "filter_type" : "contains",
            "condition": ",
            "condition_value": "Asian"
        },
        {
            "field" : "insert_date_",
            "filter_type" : "date",
            "condition": ">,
            "condition_value": "2020-07-01"
        }
    ]

    multivector_query=[
        {
            "vector": "query_vec_txt",
            "fields": "descriptiontextmulti_vector_"
        }
    ]

    results = ds.vector_search(
        multivector_query=multivector_query,
        page_size=5,
        filter=filter
    )

