|Open In Colab|

Setup
=====

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/clustering-features/cluster-evaluation/_notebooks/RelevanceAI-ReadMe-Cluster-Metrics.ipynb

.. code:: python

    # remove `!` if running the line in a terminal
    !pip install -U RelevanceAI[notebook]==2.0.0


.. code:: python

    from relevanceai import Client

    """
    You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api
    Once you have signed up, click on the value under `Activation token` and paste it here
    """
    client = Client()



Data
====

.. code:: python

    from relevanceai.utils.datasets import get_ecommerce_dataset_encoded

    documents = get_ecommerce_dataset_encoded()
    {k:v for k, v in documents[0].items() if '_vector_' not in k}


.. code:: python

    ds = client.Dataset("quickstart_kmeans_clustering")
    ds.insert_documents(documents)


.. code:: python

    ds.health()


1. Create clusters
==================

.. code:: python

    from relevanceai.clusterer import KMeansModel

    VECTOR_FIELD = "product_title_clip_vector_"
    KMEAN_NUMBER_OF_CLUSTERS = 10
    ALIAS = "kmeans_" + str(KMEAN_NUMBER_OF_CLUSTERS)

    model = KMeansModel(k=KMEAN_NUMBER_OF_CLUSTERS)
    clusterer = client.ClusterOps(alias=ALIAS, model=model)
    clusterer.fit_predict_update(df, [VECTOR_FIELD])


2. Cluster evaluation
=====================


Within cluster label (coming soon)
----------------------------------

.. code:: python

    # against a ground-truth column

    # GROUND_TRUTH_FIELD = "query"
