:py:mod:`relevanceai.clusterer.cluster_base`
============================================

.. py:module:: relevanceai.clusterer.cluster_base

.. autoapi-nested-parse::

   Document utilities



Module Contents
---------------

.. py:class:: ClusterBase



   A Cluster Base for models to be copied off.

   .. py:method:: fit_transform(self, vectors) -> List[Union[str, float, int]]
      :abstractmethod:


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



