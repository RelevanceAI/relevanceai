SubClustering 
=================

Subclustering allows users to cluster and then cluster again.

A quick example can be shown below with the `auto_cluster` functionality.

.. code-block::

    from relevanceai import Client

    client = Client()

    from relevanceai.datasets import mock_documents

    ds = client.Dataset('sample')
    ds.upsert_documents(mock_documents(100))
    # Run initial kmeans to get clusters
    ds.auto_cluster('kmeans-3', vector_fields=["sample_1_vector_"])
    # Run separate K Means to get subclusters
    ds.auto_cluster(
        'kmeans-2',
        vector_fields=["sample_1_vector_"],
        parent_alias="kmeans-3"
    )

From here, you will then see a list of labels similar to this: 

.. code-block::

    ['cluster-0-1',
    'cluster-0-1',
    'cluster-0-1',
    'cluster-0-0',
    'cluster-0-0',
    'cluster-0-1',
    'cluster-0-1',
    'cluster-0-0',
    'cluster-0-0',
    'cluster-0-1',
    'cluster-0-0',
    'cluster-0-1',
    ...]

You can infinitely continue subclustering based on previous aliases:

.. code-block::

    cluster_ops = ds.auto_cluster(
        "kmeans-4",
        vector_fields=["sample_1_vector_"],
        parent_alias="kmeans-2
    )

    # This should create labels that look like this: 

    ['cluster-0-0-3',
    'cluster-0-0-1',
    'cluster-0-0-1',
    'cluster-0-0-2',
    'cluster-0-0-0',
    'cluster-0-0-0',
    'cluster-0-0-0',
    'cluster-0-0-3',
    'cluster-0-0-2',
    'cluster-0-0-1',
    'cluster-0-0-1',
    'cluster-0-1-1',
    'cluster-0-1-1',
    ...]
