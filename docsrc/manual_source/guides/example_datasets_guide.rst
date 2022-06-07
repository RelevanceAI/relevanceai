✨ Example Dataset Guide
------------------------

Relevance AI allows users to quickly get datasets!

You can explore a list of available datasets using the function below.

.. code:: ipython3

    from relevanceai.utils import list_example_datasets

.. code:: ipython3

    list_example_datasets()


.. parsed-literal::

    Connecting to us-east-1...
    Welcome to RelevanceAI. Logged in as 3a4b969f4d5fae6f850e.




.. parsed-literal::

    {'datasets': ['_cluster_reports',
      '_mock_dataset2_',
      '_mock_dataset_',
      '_sample_test_dataset__ecom',
      '_sample_test_dataset__iris',
      '_sample_test_dataset__penguins',
      '_test_sample_hdbscan',
      'basic_subclustering',
      'community-detection-test',
      'dummy-coco',
      'dummy-ecommerce',
      'dummy-ecommerce-clean',
      'dummy-flipkart',
      'dummy-games-dataset',
      'dummy-iris',
      'dummy-mock_dataset',
      'dummy-news',
      'dummy-palmer-penguins',
      'dummy-realestate',
      'dummy-titanic',
      'ecommerce-sample-dataset',
      'ecommerce_1',
      'ecommerce_2',
      'ecommerce_3',
      'faiss_kmeans_clustering',
      'flipkart',
      'game-reviews',
      'games',
      'games-review-long-text',
      'games-review-long-text-chunk-flatten',
      'iris',
      'legal-effort',
      'legal-effort-2',
      'legal-work-codes',
      'mock-ds',
      'news',
      'online_retail',
      'palmers-penguins',
      'pokedex',
      'quickstart_aggregation',
      'quickstart_auto_clustering_kmeans',
      'quickstart_clip',
      'quickstart_clustering_aggregation',
      'quickstart_clustering_list_closest',
      'quickstart_clustering_list_furthest',
      'quickstart_clustering_metadata',
      'quickstart_example_encoding',
      'quickstart_kmeans_clustering',
      'quickstart_multi_vector_search',
      'quickstart_search',
      'realestate',
      'realestate2',
      'realestate3',
      'requested_read_key_storage',
      'retail_reviews_large',
      'retail_reviews_small',
      'sample',
      'test-vectorize',
      'things_that_went_well',
      'titanic',
      'toy_image_caption_coco_image_encoded',
      'workflows-data',
      'workflows-recipes'],
     'count': 63}



Getting An Example Dataset
--------------------------

You can retrieve a dataset using:

.. code:: ipython3

    from relevanceai.utils import example_documents

    docs = example_documents("dummy-coco")


.. parsed-literal::

    Connecting to us-east-1...
    Welcome to RelevanceAI. Logged in as 3a4b969f4d5fae6f850e.



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


Inserting Into Your Dataset
===========================

.. code:: ipython3

    from relevanceai import Client
    client = Client()
    ds = client.Dataset("example-dataset")
    ds.upsert_documents(docs)

There you have it! Go forth and now insert as many example documents as
you would like!
