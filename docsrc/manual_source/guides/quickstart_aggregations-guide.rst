|Open In Colab|

Installation
============

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/general-features/aggregations/_notebooks/RelevanceAI_ReadMe_Quickstart_Aggregations.ipynb

.. code:: python

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0


Setup
=====

.. code:: python

    from relevanceai import Client

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()



Data
====

.. code:: python

    import pandas as pd
    from relevanceai.utils.datasets import get_realestate_dataset

    # Retrieve our sample dataset. - This comes in the form of a list of documents.
    documents = get_realestate_dataset()

    # ToDo: Remove this cell when the dataset is updated

    for d in documents:
      if '_clusters_' in d:
        del d['_clusters_']

    pd.DataFrame.from_dict(documents).head()




.. code:: python

    ds = client.Dataset("quickstart_aggregation")
    ds.insert_documents(documents)


1. Grouping the Data
====================

In general, the group-by field is structured as

::

   {"name": ALIAS,
   "field": FIELD,
   "agg": TYPE-OF-GROUP}

Categorical Data
----------------

.. code:: python

    location_group = {"name": "location", "field": "propertyDetails.area", "agg": "category"}


Numerical Data
--------------

.. code:: python

    bedrooms_group = {"name": "bedrooms", "field": "propertyDetails.bedrooms", "agg": "numeric"}


Putting it Together
-------------------

.. code:: python

    groupby = [location_group, bedrooms_group]


2. Creating Aggregation Metrics
===============================

In general, the aggregation field is structured as

::

   {"name": ALIAS,
   "field": FIELD,
   "agg": TYPE-OF-AGG}

Average, Minimum and Maximum
----------------------------

.. code:: python

    avg_price_metric = {"name": "avg_price", "field": "priceDetails.price", "agg": "avg"}
    max_price_metric = {"name": "max_price", "field": "priceDetails.price", "agg": "max"}
    min_price_metric = {"name": "min_price", "field": "priceDetails.price", "agg": "min"}


Sum
---

.. code:: python

    sum_bathroom_metric = {"name": "bathroom_sum", "field": "propertyDetails.bathrooms", "agg": "sum"}


Putting it Together
-------------------

.. code:: python

    metrics = [ avg_price_metric, max_price_metric, min_price_metric, sum_bathroom_metric ]


3. Combining Grouping and Aggregating
=====================================

.. code:: python

    results = ds.aggregate(metrics=metrics, groupby=groupby)


.. code:: python

    from jsonshower import show_json
    show_json(results, text_fields=list(results['results'][0].keys()))
