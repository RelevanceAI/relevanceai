Auto Clustering
================

Auto clustering is the easiest way to cluster.

.. code-block::

    from relevanceai import Client

    client = Client()

    dataset_id = "sample_dataset"
    df = client.Dataset(dataset_id)

    # run kmeans with default 10 clusters
    clusterer = df.auto_cluster("kmeans", vector_fields=[vector_field])
    clusterer.list_closest_to_center()

    # Run k means clustering with 8 clusters
    clusterer = df.auto_cluster("kmeans-8", vector_fields=[vector_field])

    # Run minibatch k means clustering with 8 clusters
    clusterer = df.auto_cluster("minibatchkmeans-8", vector_fields=[vector_field])

    # Run minibatch k means clustering with 20 clusters
    clusterer = df.auto_cluster("minibatchkmeans-20", vector_fields=[vector_field])

You can read more about how to cluster using the `auto_cluster` below!

.. automethod:: relevanceai.dataset_api.dataset_operations.Operations.auto_cluster

For more advanced clustering methods and to use your own custom clustering
method, read the other sections under `Clusterer`.

