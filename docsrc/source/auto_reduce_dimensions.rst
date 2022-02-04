Auto Dimensionality Reduction
===============================

.. code-block::

    from relevanceai import Client

    client = Client()

    dataset_id = "sample_dataset"
    df = client.Dataset(dataset_id)

    df.auto_reduce_dimensions("pca-3",
        vector_fields=["sample_vector_"])

.. automethod:: relevanceai.dataset_api.dataset_operations.auto_reduce_dimensions
