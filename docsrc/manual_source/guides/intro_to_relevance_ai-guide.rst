|Open In Colab|

Quickstart
==========

Use Relevance AI to experiment, build and share the best vectors to
solve similarity and relevance based problems across teams.

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/RelevanceAI/RelevanceAI-readme-docs/blob/v2.0.0/docs/GETTING_STARTED/_notebooks/Intro_to_Relevance_AI.ipynb

Quick Start
===========

1. Setup Relevance AI

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



2. Vector Storage
=================

Store as many vectors and metadata with ease.

.. code:: python

    documents = [
    	{"_id": "1", "example_vector_": [0.1, 0.1, 0.1], "data": "Documentation"},
    	{"_id": "2", "example_vector_": [0.2, 0.2, 0.2], "data": "Best document!"},
    	{"_id": "3", "example_vector_": [0.3, 0.3, 0.3], "data": "Document example"},
    	{"_id": "5", "example_vector_": [0.4, 0.4, 0.4], "data": "This is a doc"},
    	{"_id": "4", "example_vector_": [0.5, 0.5, 0.5], "data": "This is another doc"},
    ]

    ds = client.Dataset("quickstart")
    ds.insert_documents(documents)


.. code:: python

    ds.schema


2. Clustering vectors
=====================

-  Cluster your vectors
-  Aggregate and understand your clusters
-  Store and compare multiple different clusters

.. code:: python

    from sklearn.cluster import KMeans

    cluster_model = KMeans(n_clusters=2)
    ds.cluster(cluster_model, ["example_vector_"])


You can run clustering via Relevance AI’s clustering app too:

.. code:: python

    ds.launch_cluster_app()


3. Searching vectors
====================

-  Perform nearest neighbor search on your vectors
-  Comes with essential search features such as keyword matching,
   filters and facets
-  Store and compare multiple different search configuration

.. code:: python

    results = ds.vector_search(
        multivector_query=[
    		{"vector": [0.2, 0.2, 0.2], "fields": ["example_vector"]}
    	],
        page_size=3
    )
