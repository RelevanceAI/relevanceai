..
   Manually maintained. Relevant functions are copied from docsrc/source/autoapi/relevanceai/dataset_api/dataset/index.rst

Dataset
=============================

Dataset is the class that Relevance AI uses to resolve a lot of complexity.

It is instantiated like this:

.. code-block::

    from relevanceai import Client 
    client = Client()
    df = client.Dataset("sample_dataset")
    df.head()

.. autoclass:: relevanceai.dataset_api.dataset_write.Write
    :members:
    :special-members: relevanceai.dataset_api.dataset.Write.insert_csv

.. autoclass:: relevanceai.dataset_api.dataset_read.Read
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset_api.dataset_stats.Stats
    :members:

.. autoclass:: relevanceai.dataset_api.dataset_export.Export
    :members:

.. autoclass:: relevanceai.dataset_api.dataset_series.Series
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset_api.dataset_operations.Operations
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset_api.dataset_search.Search
   :members:
   :exclude-members: __init__

.. autoclass:: relevanceai.dataset_api.dataset_dr.Dimensionality_Reduction
   :members:
   :exclude-members: __init__
