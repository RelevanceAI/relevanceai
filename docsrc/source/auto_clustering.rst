Auto Clustering
================

Auto clustering is the easiest way to cluster.

.. code-block::

    from relevanceai import Client
    client = Client()
    df = client.Dataset("sample")

    # Now to run KMeans with 10 clusters
    clusterer = df.auto_cluster(
        alias="kmeans-10", 
        vector_fields=["sample_vector_"]
    )

    clusterer.list_closest_to_center()

You can read more about how to cluster using the `auto_cluster` below!

.. automethod:: relevanceai.dataset_api.dataset_operations.Operations.auto_cluster

For more advanced clustering methods and to use your own custom clustering
method, read the other sections under `Clusterer`.

