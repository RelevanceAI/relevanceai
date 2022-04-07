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

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
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
    {k:v for k, v in documents[0].items() if '_vector_' not in k}


Upload the data to Relevance AI
-------------------------------

Run the following cell, to upload these documents into your personal
Relevance AI account under the name ``quickstart_clustering_metadata``

.. code:: python

    ds = client.Dataset('quickstart_clustering_metadata')
    ds.insert_documents(documents)


Check the data
--------------

.. code:: python

    ds.health()


Clustering
==========

Cluster dataset
---------------

We apply the Kmeams clustering algorithm to the vector field,
``product_title_clip_vector_``, to perform clustersing.

.. code:: python

    clusterer = ds.cluster(ALIAS-10, ["product_title_clip_vector_"])


.. code:: python

    from relevanceai.clusterer import KMeansModel

    VECTOR_FIELD = "product_title_clip_vector_"
    KMEAN_NUMBER_OF_CLUSTERS = 10
    ALIAS = "kmeans_" + str(KMEAN_NUMBER_OF_CLUSTERS)

    model = KMeansModel(k=KMEAN_NUMBER_OF_CLUSTERS)
    clusterer = client.ClusterOps(alias=ALIAS, model=model)
    clusterer.fit_predict_update(df, [VECTOR_FIELD])


Clustering results are automatically inserted into your datase. Here, we
download a small sample and show the clustering results using our
json_shower.

.. code:: python

    from relevanceai import show_json

    sample_documents = ds.sample(n=5)
    samples = [{
        'product_title':d['product_title'],
        'cluster':d['_cluster_'][VECTOR_FIELD][ALIAS]
    } for d in sample_documents]

    show_json(samples, text_fields=['product_title', 'cluster'])



Metadata
--------

.. code:: python

    clusterer.metadata
