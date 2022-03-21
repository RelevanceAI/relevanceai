..
   Manually maintained. Relevant functions are copied from docsrc/source/autoapi/relevanceai/dataset_api/dataset/index.rst

Dataset
=============================

Dataset is the class that Relevance AI uses to resolve a lot of complexity.

It is instantiated like this:

.. code-block::

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("sample_dataset_id")
    ds.head()

You can also easily access metadata using the following: 

.. code-block::

    ds = client.Dataset("_mock_dataset_")
    ds.metadata['value'] = 3
    ds.metadata['strong_values'] = 10
    import time
    time.sleep(1)
    ds.metadata

.. autoclass:: relevanceai.dataset.crud.dataset_write.Write
    :members:
    :special-members: relevanceai.dataset.crud.write.Write.insert_csv

.. autoclass:: relevanceai.dataset.crud.dataset_read.Read
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset.search.search.Search
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset.statistics.interface.Stats
    :members:

.. autoclass:: relevanceai.dataset.export.interface.Export
    :members:

.. autoclass:: relevanceai.dataset.crud.dataset_series.Series
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset.crud.dataset_metadata._Metadata
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset.ops.dataset_operations.Operations
    :members:
    :exclude-members: label_with_model_from_dataset, label_vector, label_document
