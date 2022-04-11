☀️ Cluster Centroid Heat Maps
=============================

In order to better interpret your clusters, you may need to visualise
them using heatmaps. These heatmaps allow users to see which clusters
are the closest.

.. code:: ipython3

    In [1]: %load_ext autoreload
    In [2]: %autoreload 2

.. code:: ipython3

    from relevanceai import Client

.. code:: ipython3

    client = Client()

You can retrieve the ecommerce dataset from
https://relevanceai.readthedocs.io/en/development/core/available_datasets.html#relevanceai.utils.datasets.get_ecommerce_1_dataset.

.. code:: ipython3

    ds = client.Dataset("ecommerce")

.. code:: ipython3

    from relevanceai.operations.viz.cluster import ClusterVizOps
    cluster_ops = ClusterVizOps.from_dataset(ds, alias="main-cluster", vector_fields=["product_image_clip_vector_"])

.. code:: ipython3

    cluster_ops.centroid_heatmap()


.. parsed-literal::

    Your closest centroids are:
    0.74 cluster-5, cluster-1
    0.73 cluster-5, cluster-4
    0.71 cluster-4, cluster-1
    0.65 cluster-4, cluster-2
    0.65 cluster-7, cluster-2
    0.64 cluster-7, cluster-4
    0.64 cluster-7, cluster-5
    0.63 cluster-5, cluster-2




.. parsed-literal::

    [Text(0.5, 1.0, 'cosine plot')]




.. image:: cluster_centroid_heatmap_guide_files/cluster_centroid_heatmap_guide_8_2.png


Now we can see if our clusters are useful when we check the dashboard
and inspect those clusters:

.. code:: ipython3

    closest = cluster_ops.closest()['results']


.. parsed-literal::

    You can now visit the dashboard at https://cloud.relevance.ai/sdk/cluster/centroids/closest


Below, we can now see if 2 separate clusters. One for boots and one for
shoes and if we need that granularity.

.. figure:: docsrc/manual_source/guides/cluster_centroid_heatmap_guide_files/cluster-centroid-dashboard-cluster-1.png
   :alt: image.png

   image.png

.. figure:: docsrc/manual_source/guides/cluster_centroid_heatmap_guide_files/cluster-centroid-dashboard-cluster-5.png
   :alt: image.png

   image.png
