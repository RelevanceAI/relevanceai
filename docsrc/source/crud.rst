Create Read Update Delete
=============================

Create/Insertion
---------

Creating via insertion (preferred)
************************************

.. code-block:: python

    from relevanceai import Client 
    client = Client()
    documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
    df = client.Dataset("sample")
    df.read_documents(dataset_id, documents)

Creating (without insertion)
********************************

.. code-block:: python

    from relevanceai import Client 
    client = Client()
    documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
    df = client.Dataset("sample")
    df.read_documents(dataset_id, documents)

Read
------

Getting by ID
***************

.. code-block:: python

    from relevanceai import Client
    client = Client()
    df.get("id_1")


Update
-------

.. code-block:: python

    from relevanceai import Client
    client = Client()
    documents = [
        {
            "_id": "id_1",
            "value": 10
        },
        {
            "_id": "id_2,
            "value": 20
        }
    ]
    client.update_documents(dataset_id, documents)


Delete
-------

To delete a dataset, just run:

.. code-block:: python

    from relevanceai import Client 
    client = Client()
    df = client.Dataset("sample_dataset")
    df.delete()

Listing Datasets
------------------

.. code-block:: python

    from relevanceai import Client
    client = Client()
    client.list_datasets()
