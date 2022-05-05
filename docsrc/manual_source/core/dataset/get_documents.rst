Get Documents
=================

The easiest way to get documents is to firstly instantiate the `Dataset` object and then 
get run `get`.

For example: 

.. code-block::

    from relevanceai import Client 
    client = Client()
    ds = client.Dataset("sample")
    ds.sample() # to get a sample of documents
    ds.get("sample_id") # to get documents

Getting All Documents 
-----------------------

.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("sample")
    ds.get_all_documents()

Getting Documents With Filters
-----------------------

You can also get documents with filters using

.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("ecommerce-example-encoded")
    documents = ds.get_documents(filters=ds['product_title'].exists())

Getting Documents by IDs
-----------------------

Retrieve a document by its ID (“_id” field). This will retrieve the document faster than a filter applied on the “_id” field.
You can do so by running the following:

.. code-block::

    from relevanceai import Client, Dataset
    client = Client()
    dataset_id = "sample_dataset_id"
    df = client.Dataset(dataset_id)
    df.get_documents_by_ids(["sample_id"], include_vector=False)
