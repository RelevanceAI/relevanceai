Auto Dimensionality Reduction
===============================

.. code-block::

    from relevanceai import Client

    client = Client()

    dataset_id = "sample_dataset_id"
    df = client.Dataset(dataset_id)

    df.auto_reduce_dimensions("pca-3",
        vector_fields=["sample_vector_"])

    df.auto_reduce_dimensions("tsne-3",
        vector_fields=["sample_vector_"])

    df.auto_reduce_dimensions("umap-3",
        vector_fields=["sample_vector_"])

    df.auto_reduce_dimensions("ivis-3",
        vector_fields=["sample_vector_"])

.. automethod:: relevanceai.dataset_ops.dataset_operations.Operations.auto_reduce_dimensions
