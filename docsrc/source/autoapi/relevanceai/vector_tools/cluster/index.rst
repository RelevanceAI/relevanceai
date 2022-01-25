:py:mod:`relevanceai.vector_tools.cluster`
==========================================

.. py:module:: relevanceai.vector_tools.cluster


Module Contents
---------------

.. py:class:: ClusterBase



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors)
      :abstractmethod:


   .. py:method:: fit_documents(self, vector_fields: list, docs: list, alias: str = 'default', cluster_field: str = '_cluster_', return_only_clusters: bool = True, inplace: bool = True)

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


   .. py:method:: to_metadata(self)
      :abstractmethod:

      You can also store the metadata of this clustering algorithm


   .. py:method:: metadata(self)
      :property:



.. py:class:: CentroidCluster



   Using verbose loguru as base logger for now

   .. py:attribute:: get_centroid_docs
      

      

   .. py:method:: fit_transform(self, vectors)
      :abstractmethod:


   .. py:method:: get_centers(self) -> Union[numpy.ndarray, List[list]]
      :abstractmethod:

      Get centers for the centroid-based clusters


   .. py:method:: get_centroid_documents(self, centroid_vector_field_name='centroid_vector_') -> List

      Get the centroid documents to store.
      If single vector field returns this:
          {
              "_id": "document-id-1",
              "centroid_vector_": [0.23, 0.24, 0.23]
          }
      If multiple vector fields returns this:
      Returns multiple
      ```
      {
          "_id": "document-id-1",
          "blue_vector_": [0.12, 0.312, 0.42],
          "red_vector_": [0.23, 0.41, 0.3]
      }
      ```



.. py:class:: DensityCluster



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors)
      :abstractmethod:



.. py:class:: MiniBatchKMeans(k: Union[None, int] = 10, init: str = 'k-means++', verbose: bool = False, compute_labels: bool = True, max_no_improvement: int = 2)



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors: Union[numpy.ndarray, List])

      Fit and transform transform the vectors


   .. py:method:: get_centers(self)

      Returns centroids of clusters


   .. py:method:: to_metadata(self)

      Editing the metadata of the function



.. py:class:: KMeans(k=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')



   Using verbose loguru as base logger for now

   .. py:method:: to_metadata(self)

      Editing the metadata of the function



.. py:class:: HDBSCANClusterer(algorithm: str = 'best', alpha: float = 1.0, approx_min_span_tree: bool = True, gen_min_span_tree: bool = False, leaf_size: int = 40, memory=Memory(cachedir=None), metric: str = 'euclidean', min_samples: int = None, p: float = None, min_cluster_size: Union[None, int] = 10)



   Using verbose loguru as base logger for now

   .. py:method:: fit_transform(self, vectors: numpy.ndarray) -> numpy.ndarray



.. py:class:: Cluster(project, api_key)



   Batch API client

   .. py:method:: cluster(vectors: numpy.ndarray, cluster: Union[relevanceai.vector_tools.constants.CLUSTER, ClusterBase], cluster_args: Dict = {}, k: Union[None, int] = None) -> numpy.ndarray
      :staticmethod:

      Cluster vectors


   .. py:method:: kmeans_cluster(self, dataset_id: str, vector_fields: list, alias: str, filters: List = [], k: Union[None, int] = 10, init: str = 'k-means++', n_init: int = 10, max_iter: int = 300, tol: float = 0.0001, verbose: bool = True, random_state: Optional[int] = None, copy_x: bool = True, algorithm: str = 'auto', cluster_field: str = '_cluster_', update_documents_chunksize: int = 50, overwrite: bool = False, page_size: int = 1)

      This function performs all the steps required for Kmeans clustering:
      1- Loads the data
      2- Clusters the data
      3- Updates the data with clustering info
      4- Adds the centroid to the hidden centroid collection

      :param dataset_id: name of the dataser
      :type dataset_id: string
      :param vector_fields: a list containing the vector field to be used for clustering
      :type vector_fields: list
      :param alias: "kmeans", string to be used in naming of the field showing the clustering results
      :type alias: string
      :param filters: a list to filter documents of the dataset,
      :type filters: list
      :param k: K in Kmeans
      :type k: int
      :param init: "k-means++" -> Kmeans algorithm parameter
      :type init: string
      :param n_init: number of reinitialization for the kmeans algorithm
      :type n_init: int
      :param max_iter: max iteration in the kmeans algorithm
      :type max_iter: int
      :param tol: tol in the kmeans algorithm
      :type tol: int
      :param verbose: True by default
      :type verbose: bool
      :param random_state = None: None by default -> Kmeans algorithm parameter
      :param copy_x: True bydefault
      :type copy_x: bool
      :param algorithm: "auto" by default
      :type algorithm: string
      :param cluster_field: "_cluster_", string to name the main cluster field
      :type cluster_field: string
      :param overwrite: False by default, To overwite an existing clusering result
      :type overwrite: bool

      .. rubric:: Example

      >>> client.vector_tools.cluster.kmeans_cluster(
          dataset_id="sample_dataset",
          vector_fields=vector_fields
      )


   .. py:method:: hdbscan_cluster(self, dataset_id: str, vector_fields: list, filters: List = [], algorithm: str = 'best', alpha: float = 1.0, approx_min_span_tree: bool = True, gen_min_span_tree: bool = False, leaf_size: int = 40, memory=Memory(cachedir=None), metric: str = 'euclidean', min_samples=None, p=None, min_cluster_size: Union[None, int] = 10, alias: str = 'hdbscan', cluster_field: str = '_cluster_', update_documents_chunksize: int = 50, overwrite: bool = False)

      This function performs all the steps required for hdbscan clustering:
      1- Loads the data
      2- Clusters the data
      3- Updates the data with clustering info
      4- Adds the centroid to the hidden centroid collection

      :param dataset_id: name of the dataser
      :type dataset_id: string
      :param vector_fields: a list containing the vector field to be used for clustering
      :type vector_fields: list
      :param filters: a list to filter documents of the dataset
      :type filters: list
      :param algorithm: hdbscan configuration parameter default to "best"
      :type algorithm: str
      :param alpha: hdbscan configuration parameter default to 1.0
      :type alpha: float
      :param approx_min_span_tree: hdbscan configuration parameter default to True
      :type approx_min_span_tree: bool
      :param gen_min_span_tree: hdbscan configuration parameter default to False
      :type gen_min_span_tree: bool
      :param leaf_size: hdbscan configuration parameter default to 40
      :type leaf_size: int
      :param memory = Memory(cachedir=None): hdbscan configuration parameter on memory management
      :param metric: hdbscan configuration parameter default to "euclidean"
      :type metric: str = "euclidean"
      :param min_samples = None: hdbscan configuration parameter default to None
      :param p = None: hdbscan configuration parameter default to None
      :param min_cluster_size: minimum cluster size, 10 by default
      :param alias: "hdbscan", string to be used in naming of the field showing the clustering results
      :type alias: string
      :param cluster_field: "_cluster_", string to name the main cluster field
      :type cluster_field: string
      :param overwrite: False by default, To overwite an existing clusering result
      :type overwrite: bool

      .. rubric:: Example

      >>> client.vector_tools.cluster.hdbscan_cluster(
          dataset_id="sample_dataset",
          vector_fields=["sample_1_vector_"] # Only 1 vector field is supported for now
      )



