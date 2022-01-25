:py:mod:`relevanceai.clusterer.cluster_base`
============================================

.. py:module:: relevanceai.clusterer.cluster_base

.. autoapi-nested-parse::

   Document utilities



Module Contents
---------------

.. py:class:: ClusterBase



   A Cluster Base for models to be copied off.

   .. py:method:: fit_transform(self, vectors: list) -> List[Union[str, float, int]]
      :abstractmethod:

      Edit this method to implement a ClusterBase.

      :param vectors: The vectors that are going to be clustered
      :type vectors: list

      .. rubric:: Example

      .. code-block::

          class KMeansModel(ClusterBase):
              def __init__(self, k=10, init="k-means++", n_init=10,
                  max_iter=300, tol=1e-4, verbose=0, random_state=None,
                      copy_x=True,algorithm="auto"):
                      self.init = init
                      self.n_init = n_init
                      self.max_iter = max_iter
                      self.tol = tol
                      self.verbose = verbose
                      self.random_state = random_state
                      self.copy_x = copy_x
                      self.algorithm = algorithm
                      self.n_clusters = k

          def _init_model(self):
              from sklearn.cluster import KMeans
              self.km = KMeans(
                  n_clusters=self.n_clusters,
                  init=self.init,
                  verbose=self.verbose,
                  max_iter=self.max_iter,
                  tol=self.tol,
                  random_state=self.random_state,
                  copy_x=self.copy_x,
                  algorithm=self.algorithm,
              )
              return

          def fit_transform(self, vectors: Union[np.ndarray, List]):
              if not hasattr(self, "km"):
                  self._init_model()
              self.km.fit(vectors)
              cluster_labels = self.km.labels_.tolist()
              # cluster_centroids = km.cluster_centers_
              return cluster_labels


   .. py:method:: fit_documents(self, vector_fields: list, documents: List[dict], alias: str = 'default', cluster_field: str = '_cluster_', return_only_clusters: bool = True, inplace: bool = True)

      Train clustering algorithm on documents and then store the labels
      inside the documents.

      :param vector_field: The vector field of the documents
      :type vector_field: list
      :param documents: List of documents to run clustering on
      :type documents: list
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


   .. py:method:: metadata(self) -> dict
      :property:

      If metadata is set - this willi be stored on RelevanceAI.
      This is useful when you are looking to compare the metadata of your clusters.



