:py:mod:`relevanceai.clusterer.clusterer`
=========================================

.. py:module:: relevanceai.clusterer.clusterer

.. autoapi-nested-parse::

   Clusterer class to run clustering.



Module Contents
---------------

.. py:class:: Clusterer(model: relevanceai.clusterer.cluster_base.ClusterBase, alias: str, project: str, api_key: str, cluster_field: str = '_cluster_')



   Clusterer allows users to be able to

   :param alias: The name to call your cluster.  This will be used to store your clusters in the form of {cluster_field{.vector_field.alias}
   :type alias: str
   :param k: The number of clusters in your K Means
   :type k: str
   :param cluster_field: The field from which to store the cluster. This will be used to store your clusters in the form of {cluster_field{.vector_field.alias}
   :type cluster_field: str
   :param You can read about the other parameters here:
   :type You can read about the other parameters here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

   .. rubric:: Example

   >>> from relevanceai import Client
   >>> client = Client()
   >>> clusterer = client.KMeansClusterer()
   >>> df = client.Dataset("sample")
   >>> clusterer.fit(df, vector_fields=["sample_vector_"])

   .. py:method:: fit(self, dataset: Union[relevanceai.dataset_api.Dataset, str], vector_fields: List)

      This function takes in the dataset and the relevant vector fields.
      Under the hood, it runs fit_dataset. Sometimes, you may want to modify the behavior
      to adapt it to your needs.

      :param dataset: The dataset to fit the clusterer on
      :type dataset: Union[Dataset, str]
      :param vector_fields: The vector fields to fit it on
      :type vector_fields: List[str],

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> from relevanceai import ClusterBase
      >>> import random
      >>> class RandomClusterer(ClusterBase):
      >>>     def __init__(self):
      >>>     pass
      >>> # update this to update documents
      >>> def fit_transform(self, X):
      >>>     return random.randint(0, 100)
      >>> clusterer = client.KMeansClusterer()
      >>> df = client.Dataset("sample")
      >>> clusterer.fit(df)


   .. py:method:: fit_dataset(self, dataset: Union[relevanceai.dataset_api.Dataset, str], vector_fields: List, filters: List = [])

      This function fits a cluster model onto a dataset.

      :param dataset: The dataset object to fit it on
      :type dataset: Union[Dataset, str],
      :param vector_fields: The vector fields to fit the model on
      :type vector_fields: list
      :param filters: The filters to run it on
      :type filters: list

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> from relevanceai import ClusterBase
      >>> import random
      >>> class RandomClusterer(ClusterBase):
      >>>     def __init__(self):
      >>>     pass
      >>> # update this to update documents
      >>> def fit_transform(self, X):
      >>>     return random.randint(0, 100)
      >>> clusterer = client.KMeansClusterer()
      >>> df = client.Dataset("sample")
      >>> clusterer.fit(df)


   .. py:method:: fit_documents(self, vector_fields: list, documents: List[Dict], return_only_clusters: bool = True, inplace: bool = True)

      Train clustering algorithm on documents and then store the labels
      inside the documents.

      :param vector_field: The vector field of the documents
      :type vector_field: list
      :param docs: List of documents to run clustering on
      :type docs: list
      :param alias: What the clusters can be called
      :type alias: str
      :param cluster_field: What the cluster fields should be called
      :type cluster_field: str
      :param return_only_clusters: If True, return only clusters, otherwise returns the original document
      :type return_only_clusters: bool
      :param inplace: If True, the documents are edited inplace otherwise, a copy is made first
      :type inplace: bool
      :param kwargs: Any other keyword argument will go directly into the clustering algorithm
      :type kwargs: dict

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> from relevanceai import ClusterBase
      >>> import random
      >>> class RandomClusterer(ClusterBase):
      >>>     def __init__(self):
      >>>     pass
      >>> # update this to update documents
      >>> def fit_documents(self, documents, *args, **kw):
      >>>     X = self.get_field_across_documents("sample_vector_", documents)
      >>>     y = self.get_field_across_documents("entropy", documents)
      >>>     cluster_labels = self.fit_transform(documents, entropy)
      >>>     self.set_cluster_labels_across_documents(cluster_labels, documents)
      >>> def fit_transform(self, X, y):
      >>>     cluster_labels = []
      >>>     for y_value in y:
      >>>     if y_value == "auto":
      >>>         cluster_labels.append(1)
      >>>     else:
      >>>         cluster_labels.append(random.randint(0, 100))
      >>>     return cluster_labels
      >>> clusterer = client.KMeansClusterer()
      >>> df = client.Dataset("sample")
      >>> clusterer.fit(df, ["sample_vector_"])


   .. py:method:: set_cluster_labels_across_documents(self, cluster_labels: list, documents: List[Dict], inplace: bool = True, return_only_clusters: bool = True)


   .. py:method:: list_furthest_from_center(self)

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
      :param facets: Fields to include in the facets, if [] then all
      :type facets: list
      :param min_score: Minimum score for similarity metric
      :type min_score: int
      :param include_vectors: Include vectors in the search results
      :type include_vectors: bool
      :param include_count: Include the total count of results in the search results
      :type include_count: bool
      :param include_facets: Include facets in the search results
      :type include_facets: bool

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> df = client.Dataset("_github_repo_vectorai")
      >>> cluster = client.ClusterWorkFlow()
      >>> clusterer.fit(df)
      >>> clusterer.list_furthest_from_center()


   .. py:method:: list_closest_to_center(self, cluster_ids: List = [], centroid_vector_fields: List = [], select_fields: List = [], approx: int = 0, sum_fields: bool = True, page_size: int = 1, page: int = 1, similarity_metric: str = 'cosine', filters: List = [], min_score: int = 0, include_vector: bool = False, include_count: bool = True)

      List of documents closest from the centre.

      :param cluster_ids: Any of the cluster ids
      :type cluster_ids: lsit
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
      :param facets: Fields to include in the facets, if [] then all
      :type facets: list
      :param min_score: Minimum score for similarity metric
      :type min_score: int
      :param include_vectors: Include vectors in the search results
      :type include_vectors: bool
      :param include_count: Include the total count of results in the search results
      :type include_count: bool
      :param include_facets: Include facets in the search results
      :type include_facets: bool

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> df = client.Dataset("sample_dataset")
      >>> clusterer = client.KMeansClusterer()
      >>> clusterer.fit(df, ["sample_vector_"])
      >>> clusterer.list_closest_to_center()


   .. py:method:: aggregate(self, metrics: list = [], sort: list = [], groupby: list = [], filters: list = [], page_size: int = 20, page: int = 1, asc: bool = False, flatten: bool = True)

      Takes an aggregation query and gets the aggregate of each cluster in a collection. This helps you interpret each cluster and what is in them.
      It can only can be used after a vector field has been clustered.


      For more information about aggregations check out services.aggregate.aggregate.

      :param metrics: Fields and metrics you want to calculate
      :type metrics: list
      :param groupby: Fields you want to split the data into
      :type groupby: list
      :param filters: Query for filtering the search results
      :type filters: list
      :param page_size: Size of each page of results
      :type page_size: int
      :param page: Page of the results
      :type page: int
      :param asc: Whether to sort results by ascending or descending order
      :type asc: bool
      :param flatten: Whether to flatten
      :type flatten: bool

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> df = client.Dataset("sample_dataset")
      >>> clusterer = client.KMeansClusterer()
      >>> clusterer.fit(df, ["sample_vector_"])
      >>> clusterer.aggregate(
      >>>     groupby=[],
      >>>     metrics=[
      >>>         {"name": "average_score", "field": "final_score", "agg": "avg"},
      >>>     ]
      >>> )


   .. py:method:: metadata(self)
      :property:

      If metadata is none, retrieves metadata about a dataset. notably description, data source, etc
      Otherwise, you can store the metadata about your cluster here.

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> df = client.Dataset("_github_repo_vectorai")
      >>> kmeans = client.KMeansClusterer(df)
      >>> kmeans.fit(df, vector_fields=["sample_1_vector_"])
      >>> kmeans.metadata
      # {"k": 10}

