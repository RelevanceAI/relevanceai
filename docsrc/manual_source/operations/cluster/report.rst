Cluster Report
=================

The cluster report provides users with the ability to assess their clusters.

You can print an internal report as such:

.. code-block:: python

    from relevanceai import Client
    client = Client()

    ds = client.Dataset("clothes") 
    vector_fields = ["image_path_clip_vector_"]
    alias = "minibatchkmeans-10"

    # Run Clustering
    cluster_ops = ds.cluster(
        model="communitydetection", 
        vector_fields=["image_path_clip_vector_"]
    )
    cluster_ops.internal_report
