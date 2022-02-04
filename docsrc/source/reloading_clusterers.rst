Reloading Clusterers
======================

You can reload reload clusterers in 2 ways. 

.. code-block::

    clusterer = client.ClusterOps("kmeans-16", dataset_id="_github_repo_vectorai", vector_fields=['documentation_vector_'])
    clusterer.list_closest_to_center()
