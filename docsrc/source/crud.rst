Create Read Update Delete
=============================

Create/Insertion
---------

Creating via upsert (preferred)
************************************

Upserting means that the document is updated if it is there but will not
overwrite otherwise it will insert if it is not.

.. code-block:: python

    from relevanceai import Client 
    client = Client()
    documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
    df = client.Dataset("sample")
    df.upsert_documents(dataset_id, documents)

Creating (without insertion)
********************************

Sometimes if the automatic schema detection is not working appropriately, it may
be appropriate to specify the schema yourself. In this cases, you can use this

.. code-block:: python

    from relevanceai import Client 
    client = Client()
    documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
    df = client.Dataset("sample")
    df.create()

Creating via insertion
************************************

.. code-block:: python

    from relevanceai import Client 
    client = Client()
    documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
    df = client.Dataset("sample")
    df.upsert(dataset_id, documents)

Insertion vs upsert
**************************

Users can choose to insert or they can upsert. The key difference between the 
2 is that `insert` will overwrite the document if the ID of the document is the
same whereas `upsert` will cerate a separte document if the ID of the document
is different.

Read
------

Getting by ID
***************

.. code-block:: python

    from relevanceai import Client, Dataset
    client = Client()
    df = client.Dataset("sample_dataset")
    df.get("sample_id", include_vector=False)


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
