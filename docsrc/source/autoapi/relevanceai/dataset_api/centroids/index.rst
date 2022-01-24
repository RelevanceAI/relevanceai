:py:mod:`relevanceai.dataset_api.centroids`
===========================================

.. py:module:: relevanceai.dataset_api.centroids


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.dataset_api.centroids.Centroids




.. py:class:: Centroids(project: str, api_key: str, dataset_id: str)

   Bases: :py:obj:`relevanceai.api.client.BatchAPIClient`

   Batch API client

   .. py:method:: __call__(self, vector_fields: list, alias: str, cluster_field: str = '_cluster_')

      Instaniates Centroids Class which stores centroid information to be called

      :param vector_fields: The vector field where a clustering task was run.
      :type vector_fields: list
      :param alias: Alias is used to name a cluster
      :type alias: string
      :param cluster_field: Name of clusters in documents
      :type cluster_field: string


   .. py:method:: closest(self, cluster_ids: List = [], centroid_vector_fields: List = [], select_fields: List = [], approx: int = 0, sum_fields: bool = True, page_size: int = 1, page: int = 1, similarity_metric: str = 'cosine', filters: List = [], min_score: int = 0, include_vector: bool = False, include_count: bool = True)

      List of documents closest from the centre.

      :param cluster_ids: Any of the cluster ids
      :type cluster_ids: list
      :param centroid_vector_fields: Vector fields stored
      :type centroid_vector_fields: list
      :param select_fields: Fields to include in the search results, empty array/list means all fields
      :type select_fields: list
      :param approx: Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate
      :type approx: int
      :param sum_fields: Whether to sum the multiple vectors similarity search score as 1 or seperate
      :type sum_fields: bool
      :param page_size: Size of each page of results
      :type page_size: int
      :param page: Page of the results
      :type page: int
      :param similarity_metric: Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
      :type similarity_metric: string
      :param filters: Query for filtering the search results
      :type filters: list
      :param min_score: Minimum score for similarity metric
      :type min_score: int
      :param include_vectors: Include vectors in the search results
      :type include_vectors: bool
      :param include_count: Include the total count of results in the search results
      :type include_count: bool


   .. py:method:: furthest(self, cluster_ids: List = [], centroid_vector_fields: List = [], select_fields: List = [], approx: int = 0, sum_fields: bool = True, page_size: int = 1, page: int = 1, similarity_metric: str = 'cosine', filters: List = [], min_score: int = 0, include_vector: bool = False, include_count: bool = True)

      List of documents furthest from the centre.

      :param cluster_ids: Any of the cluster ids
      :type cluster_ids: list
      :param select_fields: Fields to include in the search results, empty array/list means all fields
      :type select_fields: list
      :param approx: Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate
      :type approx: int
      :param sum_fields: Whether to sum the multiple vectors similarity search score as 1 or seperate
      :type sum_fields: bool
      :param page_size: Size of each page of results
      :type page_size: int
      :param page: Page of the results
      :type page: int
      :param similarity_metric: Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
      :type similarity_metric: string
      :param filters: Query for filtering the search results
      :type filters: list
      :param min_score: Minimum score for similarity metric
      :type min_score: int
      :param include_vectors: Include vectors in the search results
      :type include_vectors: bool
      :param include_count: Include the total count of results in the search results
      :type include_count: bool



