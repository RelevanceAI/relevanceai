|Open In Colab| # Quickstart to get features such as: - hybrid search -
multivector search - filtered search - etc out of the box

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/getting-started/example-applications/_notebooks/RelevanceAI-ReadMe-Multi-Vector-Search.ipynb

What I Need
===========

-  Project & API Key (grab your API key from https://cloud.relevance.ai/
   in the settings area)
-  Python 3
-  Relevance AI Installed as shown below. You can also visit our
   `Installation guide <https://docs.relevance.ai/docs>`__

Installation Requirements
=========================

.. code:: ipython3

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0


Multivector search
==================

Client
------

To use Relevance AI, a client object must be instantiated. This needs an
API_key and a project name. These can be generated/access directly at
https://cloud.relevance.ai/ or simply by running the cell below and
following the link and the guide:

.. code:: ipython3

    from relevanceai import Client

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()



1) Data + Encode
----------------

For this quickstart we will be using a sample e-commerce dataset.
Alternatively, you can use your own dataset for the different steps.

.. code:: ipython3

    from relevanceai.utils.datasets import get_ecommerce_dataset_encoded

    documents = get_ecommerce_dataset_encoded()
    {k:v for k, v in documents[0].items() if '_vector_' not in k}


2) Insert
---------

Uploading our documents into the dataset ``quickstart_sample``.

In case you are uploading your own dataset, keep in mind that each
document should have a field called ’_id’. Such an id can be easily
allocated using the uuid package:

::

   import uuid

   for d in docs:
     d['_id'] = uuid.uuid4().__str__()    # Each document must have a field '_id'

.. code:: ipython3

    ds = client.Dataset("quickstart_multi_vector_search")
    ds.insert_documents(documents)


3) Search
---------

In the cell below, we will

1. get a random document from our dataset as a query data
2. form a multivector search to find other documents similart to our
   query

.. code:: ipython3

    # Query sample data
    sample_id = documents[0]['_id']
    document = ds.get_documents_by_ids([sample_id])["documents"][0]
    image_vector = document['product_image_clip_vector_']
    text_vector = document['product_title_clip_vector_']

    # Create a multivector query
    multivector_query = [
        {"vector": image_vector, "fields": ['product_image_clip_vector_']},
        {"vector": text_vector, "fields": ['product_title_clip_vector_']}
    ]


    results = ds.vector_search(
        multivector_query=multivector_query,
        page_size=5
    )


.. code:: ipython3

    from relevanceai import show_json

    print('=== QUERY === ')
    display(show_json([document], image_fields=["product_image"], text_fields=["product_title"]))

    print('=== RESULTS ===')
    show_json(results, image_fields=["product_image"], text_fields=["product_title"])
