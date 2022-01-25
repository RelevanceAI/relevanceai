:py:mod:`relevanceai.api.endpoints.services.recommend`
======================================================

.. py:module:: relevanceai.api.endpoints.services.recommend

.. autoapi-nested-parse::

   Recommmend services.



Module Contents
---------------

.. py:class:: RecommendClient(project, api_key)



   Base class for all relevanceai client utilities

   .. py:method:: vector(self, dataset_id: str, positive_document_ids: dict = {}, negative_document_ids: dict = {}, vector_fields=[], approximation_depth: int = 0, vector_operation: str = 'sum', sum_fields: bool = True, page_size: int = 20, page: int = 1, similarity_metric: str = 'cosine', facets: list = [], filters: list = [], min_score: float = 0, select_fields: list = [], include_vector: bool = False, include_count: bool = True, asc: bool = False, keep_search_history: bool = False, hundred_scale: bool = False)

      Vector Search based recommendations are done by extracting the vectors of the documents ids specified performing some vector operations and then searching the dataset with the resultant vector. This allows us to not only do recommendations but personalized and weighted recommendations.

      Here are a couple of different scenarios and what the queries would look like for those:


      Recommendations Personalized by single liked product:

      >>> positive_document_ids=['A']

      -> Document ID A Vector = Search Query


      Recommendations Personalized by multiple liked product:

      >>> positive_document_ids=['A', 'B']

      -> Document ID A Vector + Document ID B Vector = Search Query


      Recommendations Personalized by multiple liked product and disliked products:

      >>> positive_document_ids=['A', 'B'], negative_document_ids=['C', 'D']

      -> (Document ID A Vector + Document ID B Vector) - (Document ID C Vector + Document ID C Vector) = Search Query


      Recommendations Personalized by multiple liked product and disliked products with weights:

      >>> positive_document_ids={'A':0.5, 'B':1}, negative_document_ids={'C':0.6, 'D':0.4}

      -> (Document ID A Vector * 0.5 + Document ID B Vector * 1) - (Document ID C Vector * 0.6 + Document ID D Vector * 0.4) = Search Query


      You can change the operator between vectors with vector_operation:

      e.g. >>> positive_document_ids=['A', 'B'], negative_document_ids=['C', 'D'], vector_operation='multiply'

      -> (Document ID A Vector * Document ID B Vector) - (Document ID C Vector * Document ID D Vector) = Search Query

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param positive_document_ids: Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
      :type positive_document_ids: dict
      :param negative_document_ids: Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
      :type negative_document_ids: dict
      :param vector_fields: The vector field to search in. It can either be an array of strings (automatically equally weighted) (e.g. ['check_vector_', 'yellow_vector_']) or it is a dictionary mapping field to float where the weighting is explicitly specified (e.g. {'check_vector_': 0.2, 'yellow_vector_': 0.5})
      :type vector_fields: list
      :param approximation_depth: Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
      :type approximation_depth: int
      :param vector_operation: Aggregation for the vectors when using positive and negative document IDs, choose from ['mean', 'sum', 'min', 'max', 'divide', 'mulitple']
      :type vector_operation: string
      :param sum_fields: Whether to sum the multiple vectors similarity search score as 1 or seperate
      :type sum_fields: bool
      :param page_size: Size of each page of results
      :type page_size: int
      :param page: Page of the results
      :type page: int
      :param similarity_metric: Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
      :type similarity_metric: string
      :param facets: Fields to include in the facets, if [] then all
      :type facets: list
      :param filters: Query for filtering the search results
      :type filters: list
      :param min_score: Minimum score for similarity metric
      :type min_score: int
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list
      :param include_vector: Include vectors in the search results
      :type include_vector: bool
      :param include_count: Include the total count of results in the search results
      :type include_count: bool
      :param asc: Whether to sort results by ascending or descending order
      :type asc: bool
      :param keep_search_history: Whether to store the history into VecDB. This will increase the storage costs over time.
      :type keep_search_history: bool
      :param hundred_scale: Whether to scale up the metric by 100
      :type hundred_scale: bool


   .. py:method:: diversity(self, dataset_id: str, cluster_vector_field: str, n_clusters: int, positive_document_ids: dict = {}, negative_document_ids: dict = {}, vector_fields=[], approximation_depth: int = 0, vector_operation: str = 'sum', sum_fields: bool = True, page_size: int = 20, page: int = 1, similarity_metric: str = 'cosine', facets: list = [], filters: list = [], min_score: float = 0, select_fields: list = [], include_vector: bool = False, include_count: bool = True, asc: bool = False, keep_search_history: bool = False, hundred_scale: bool = False, search_history_id: str = None, n_init: int = 5, n_iter: int = 10, return_as_clusters: bool = False)

      Vector Search based recommendations are done by extracting the vectors of the documents ids specified performing some vector operations and then searching the dataset with the resultant vector. This allows us to not only do recommendations but personalized and weighted recommendations.

      Diversity recommendation increases the variety within the recommendations via clustering. Search results are clustered and the top k items in each cluster are selected. The main clustering parameters are cluster_vector_field and n_clusters, the vector field on which to perform clustering and number of clusters respectively.

      Here are a couple of different scenarios and what the queries would look like for those:


      Recommendations Personalized by single liked product:

      >>> positive_document_ids=['A']

      -> Document ID A Vector = Search Query

      Recommendations Personalized by multiple liked product:

      >>> positive_document_ids=['A', 'B']

      -> Document ID A Vector + Document ID B Vector = Search Query

      Recommendations Personalized by multiple liked product and disliked products:

      >>> positive_document_ids=['A', 'B'], negative_document_ids=['C', 'D']

      -> (Document ID A Vector + Document ID B Vector) - (Document ID C Vector + Document ID C Vector) = Search Query

      Recommendations Personalized by multiple liked product and disliked products with weights:

      >>> positive_document_ids={'A':0.5, 'B':1}, negative_document_ids={'C':0.6, 'D':0.4}

      -> (Document ID A Vector * 0.5 + Document ID B Vector * 1) - (Document ID C Vector * 0.6 + Document ID D Vector * 0.4) = Search Query

      You can change the operator between vectors with vector_operation:

      e.g. >>> positive_document_ids=['A', 'B'], negative_document_ids=['C', 'D'], vector_operation='multiply'

      -> (Document ID A Vector * Document ID B Vector) - (Document ID C Vector * Document ID D Vector) = Search Query

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param cluster_vector_field: The field to cluster on.
      :type cluster_vector_field: str
      :param n_clusters: Number of clusters to be specified.
      :type n_clusters: int
      :param positive_document_ids: Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
      :type positive_document_ids: dict
      :param negative_document_ids: Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
      :type negative_document_ids: dict
      :param vector_fields: The vector field to search in. It can either be an array of strings (automatically equally weighted) (e.g. ['check_vector_', 'yellow_vector_']) or it is a dictionary mapping field to float where the weighting is explicitly specified (e.g. {'check_vector_': 0.2, 'yellow_vector_': 0.5})
      :type vector_fields: list
      :param approximation_depth: Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
      :type approximation_depth: int
      :param vector_operation: Aggregation for the vectors when using positive and negative document IDs, choose from ['mean', 'sum', 'min', 'max', 'divide', 'mulitple']
      :type vector_operation: string
      :param sum_fields: Whether to sum the multiple vectors similarity search score as 1 or seperate
      :type sum_fields: bool
      :param page_size: Size of each page of results
      :type page_size: int
      :param page: Page of the results
      :type page: int
      :param similarity_metric: Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
      :type similarity_metric: string
      :param facets: Fields to include in the facets, if [] then all
      :type facets: list
      :param filters: Query for filtering the search results
      :type filters: list
      :param min_score: Minimum score for similarity metric
      :type min_score: int
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list
      :param include_vector: Include vectors in the search results
      :type include_vector: bool
      :param include_count: Include the total count of results in the search results
      :type include_count: bool
      :param asc: Whether to sort results by ascending or descending order
      :type asc: bool
      :param keep_search_history: Whether to store the history into VecDB. This will increase the storage costs over time.
      :type keep_search_history: bool
      :param hundred_scale: Whether to scale up the metric by 100
      :type hundred_scale: bool
      :param search_history_id: Search history ID, only used for storing search histories.
      :type search_history_id: str
      :param n_init: Number of runs to run with different centroid seeds
      :type n_init: int
      :param n_iter: Number of iterations in each run
      :type n_iter: int
      :param return_as_clusters: If True, return as clusters as opposed to results list
      :type return_as_clusters: bool



