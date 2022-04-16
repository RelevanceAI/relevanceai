🍔 K Means Clustering Step By Step
=================================

Installation
============

.. code:: python

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0

Setup
=====

First, you need to set up a client object to interact with RelevanceAI.

.. code:: python

    from relevanceai import Client

    client = Client()

Data
====

You will need to have a dataset under your Relevance AI account. You can
either use our e-commerce dataset as shown below or follow the tutorial
on how to create your own dataset.

Our e-commerce dataset includes fields such as ``product_title``, as
well as the vectorized version of the field
``product_title_clip_vector_``. Loading these documents can be done via:

Load the data
-------------

.. code:: python

    from relevanceai.utils.datasets import get_ecommerce_dataset_encoded

    documents = get_ecommerce_dataset_encoded()
    {k: v for k, v in documents[0].items() if "_vector_" not in k}

Upload the data to Relevance AI
-------------------------------

Run the following cell, to upload these documents into your personal
Relevance AI account under the name ``quickstart_clustering_kmeans``

.. code:: python

    ds = client.Dataset("quickstart_kmeans_clustering")
    ds.insert_documents(documents)

Check the data
--------------

.. code:: python

    ds.health()

.. code:: python

    ds.schema

Clustering
==========

We apply the Kmeams clustering algorithm to the vector field,
``product_title_clip_vector_``

.. code:: python

    from sklearn.cluster import KMeans

    VECTOR_FIELD = "product_title_clip_vector_"
    KMEAN_NUMBER_OF_CLUSTERS = 5
    ALIAS = "kmeans_" + str(KMEAN_NUMBER_OF_CLUSTERS)

    model = KMeans(n_clusters=KMEAN_NUMBER_OF_CLUSTERS)
    clusterer = client.ClusterOps(alias=ALIAS, model=model)
    clusterer.operate(
        dataset_id="quickstart_kmeans_clustering",
        vector_fields=["product_title_clip_vector_"],
    )

.. code:: python

    # List closest to center of the cluster

    clusterer.list_closest(
        dataset_id="quickstart_kmeans_clustering", vector_field="product_title_clip_vector_"
    )

.. code:: python

    # List furthest from the center of the cluster

    clusterer.list_furthest(
        dataset_id="quickstart_kmeans_clustering", vector_field="product_title_clip_vector_"
    )

We download a small sample and show the clustering results using our
json_shower.

.. code:: python

    from relevanceai import show_json

    sample_documents = ds.sample(n=5)
    samples = [
        {
            "product_title": d["product_title"],
            "cluster": d["_cluster_"][VECTOR_FIELD][ALIAS],
        }
        for d in sample_documents
    ]

    show_json(samples, text_fields=["product_title", "cluster"])
