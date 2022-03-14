..
   Manually maintained. Relevant functions are copied from docsrc/source/autoapi/relevanceai/dataset_api/dataset/index.rst

Dataset
=============================

Dataset is the class that Relevance AI uses to resolve a lot of complexity.

It is instantiated like this:

.. code-block::

    from relevanceai import Client
    client = Client()
    df = client.Dataset("sample_dataset_id")
    df.head()

.. autoclass:: relevanceai.dataset_crud.dataset_write.Write
    :members:
    :special-members: relevanceai.dataset_crud.dataset.Write.insert_csv

.. autoclass:: relevanceai.dataset_crud.dataset_read.Read
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset_crud.dataset_stats.Stats
    :members:

.. autoclass:: relevanceai.export.dataset_export.Export
    :members:

.. autoclass:: relevanceai.dataset_crud.dataset_series.Series
    :members:
    :exclude-members: __init__

.. autoclass:: relevanceai.dataset_ops.dataset_operations.Operations
    :members:
    :exclude-members: label_with_model_from_dataset, label_vector, label_document
