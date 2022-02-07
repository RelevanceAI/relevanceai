Reloading ClusterOps
======================

You can reload reload ClusterOps instances in 2 ways.

.. code-block::

    # State the vector fields and alias in the ClusterOps object
    clusterer = client.ClusterOps(alias="kmeans-16", dataset_id="sample_dataset",
        vector_fields=['sample_vector_'])
    clusterer.list_closest_to_center()

    # state the vector fields and alias in teh operational call
    clusterer = client.ClusterOps(alias="kmeans-16")
    clusterer.list_closest_to_center(dataset="sample_dataset",
        vector_fields=["documentation_vector_])
