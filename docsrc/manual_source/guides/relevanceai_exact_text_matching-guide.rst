Installation
============

.. code:: python

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0


Setup
=====

First, you need to set up a client object to interact with RelevanceAI.

.. code:: python

    from relevanceai import Client

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()



Data
====

You will need to have a dataset under your Relevance AI account. You can
either use our e-commerce dataset as shown below or follow the tutorial
on how to create your own dataset.

Our e-commerce dataset includes fields such as ``product_title``, as
well as the vectorized version of the field
``product_title_clip_vector_``. Loading these documents can be done via:

Load the data
-------------

.. code:: python

    from relevanceai.utils.datasets import get_ecommerce_dataset_encoded

    documents = get_ecommerce_dataset_encoded()
    {k:v for k, v in documents[0].items() if '_vector_' not in k}


Upload the data to Relevance AI
-------------------------------

Run the following cell, to upload these documents into your personal
Relevance AI account under the name ``quickstart_search``

.. code:: python

    ds = client.Dataset("quickstart_search")
    ds.insert_documents(documents)


Check the data
--------------

.. code:: python

    ds.schema


Traditional Search
==================

We call the hybrid search endpoint with no vectors to perform a pure
traditional search

.. code:: python

    results = ds.hybrid_search(
        multivector_query=[],
        text="HP 2.4GHz",
        fields=["product_title"],
        page_size=5
    )
