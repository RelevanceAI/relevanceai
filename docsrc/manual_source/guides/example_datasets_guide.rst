✨ Example Dataset Guide
------------------------

Relevance AI allows users to quickly get datasets!

You can explore a list of available datasets using the function below.

.. code:: ipython3

    from relevanceai.utils import list_example_datasets

.. code:: ipython3

    list_example_datasets()

Getting An Example Dataset
--------------------------

You can retrieve a dataset using:

.. code:: ipython3

    from relevanceai.utils import example_documents

    docs = example_documents("dummy-coco")

Inserting Into Your Dataset
===========================

.. code:: ipython3

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("example-dataset")
    ds.upsert_documents(docs)

There you have it! Go forth and now insert as many example documents as
you would like!
