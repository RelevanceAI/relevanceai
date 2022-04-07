|Open In Colab|

Installation
============

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/general-features/creating-a-dataset/_notebooks/RelevanceAI_ReadMe_Creating_A_Dataset.ipynb

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



Creating a dataset
~~~~~~~~~~~~~~~~~~

To create a new empty dataset pass the name under which you wish to save
the dataset to the ``create`` function as shown below. In this example,
we have used ``ecommerce-sample-dataset`` as the name.

.. code:: python

    from relevanceai.utils.datasets import get_ecommerce_dataset_encoded

    documents = get_ecommerce_dataset_encoded()
    {k:v for k, v in documents[0].items() if '_vector_' not in k}


.. code:: python

    ds = client.Dataset("ecommerce-sample-dataset")
    ds.insert_documents(documents)


See `Inserting and updating documents <doc:inserting-data>`__ for more
details on how to insert/upload documents into a dataset.

-  Id field: Relevance AI platform identifies unique data entries within
   a dataset using a field called ``_id`` (i.e. every document in the
   dataset must include an ``_id`` field with a unique value per
   document).
-  Vector fields: the name of vector fields must end in ``_vector_``

List your datasets
~~~~~~~~~~~~~~~~~~

You can see a list of all datasets you have uploaded to your account in
the dashboard.

Alternatively, you can use the list endpoint under Python SDK as shown
below:

.. code:: python

    client.list_datasets()


Monitoring a specific dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RelevanceAI’s dashboard at https://cloud.relevance.ai is the most
straightforward place to monitor your data.

Alternatively, you can monitor the health of a dataset using the command
below which returns the count of total missing and existing fields in
the data points in the named dataset.

.. code:: python

    ds.health()


.. code:: python

    ds.schema


Deleting a dataset
~~~~~~~~~~~~~~~~~~

Deleting an existing dataset can be done on the dashboard using the
delete option available for each dataset. Or through the Python SDK:

.. code:: python

    client.delete_dataset(dataset_id="ecommerce-sample-dataset")


Inserting Documents
-------------------

Inserting new documents into a dataset is simple as the command below.

.. code:: python

    ds.insert_documents(documents=documents)


Upserting Documents
-------------------

To only update specific documents, use ``upsert_documents`` as shown in
the example below: Keep in mind that if you are updating a previously
inserted document, you need to include the ``_id`` fiedld as a reference
when upserting.

.. code:: python

    SAMPLE_DOCUMENT = {
        '_id': '711160239',
        'product_image': 'https://thumbs4.ebaystatic.com/d/l225/pict/321567405391_1.jpg',
        'product_image_clip_vector_': [0.1, 0.1, 0.1],
        'product_link': 'https://www.ebay.com/itm/20-36-Mens-Silver-Stainless-Steel-Braided-Wheat-Chain-Necklace-Jewelry-3-4-5-6MM-/321567405391?pt=LH_DefaultDomain_0&var=&hash=item4adee9354f',
        'product_price': '$7.99 to $12.99',
        'product_title': '20-36Mens Silver Stainless Steel Braided Wheat Chain Necklace Jewelry 3/4/5/6MM"',
        'product_title_clip_vector_': [0.1, 0.1, 0.1],
        'query': 'steel necklace',
        'source': 'eBay'
    }


.. code:: python

    ds.upsert_documents(documents=[SAMPLE_DOCUMENT])
