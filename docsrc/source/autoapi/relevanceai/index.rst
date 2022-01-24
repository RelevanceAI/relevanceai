:py:mod:`relevanceai`
=====================

.. py:module:: relevanceai


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   api/index.rst
   clusterer/index.rst
   data_tools/index.rst
   dataset_api/index.rst
   vector_tools/index.rst
   visualise/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base/index.rst
   concurrency/index.rst
   config/index.rst
   dashboard_mappings/index.rst
   datasets/index.rst
   errors/index.rst
   http_client/index.rst
   json_encoder/index.rst
   logger/index.rst
   print_formats/index.rst
   progress_bar/index.rst
   transport/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.Client
   relevanceai.ClusterBase




Attributes
~~~~~~~~~~

.. autoapisummary::

   relevanceai.__version__
   relevanceai.pypi_data


.. py:class:: Client(project=os.getenv('RELEVANCE_PROJECT'), api_key=os.getenv('RELEVANCE_API_KEY'), authenticate: bool = False)

   Bases: :py:obj:`relevanceai.api.client.BatchAPIClient`, :py:obj:`doc_utils.doc_utils.DocUtils`

   Python Client for Relevance AI's relevanceai

   .. py:attribute:: FAIL_MESSAGE
      :annotation: = Your API key is invalid. Please login again

      

   .. py:attribute:: _cred_fn
      :annotation: = .creds.json

      

   .. py:attribute:: build_and_plot_clusters
      

      CRUD-related utility functions

   .. py:method:: base_url(self)
      :property:


   .. py:method:: base_ingest_url(self)
      :property:


   .. py:method:: _token_to_auth(self)


   .. py:method:: _write_credentials(self, project, api_key)


   .. py:method:: _read_credentials(self)


   .. py:method:: login(self, authenticate: bool = True)

      Preferred login method for demos and interactive usage.


   .. py:method:: auth_header(self)
      :property:


   .. py:method:: make_search_suggestion(self)


   .. py:method:: check_auth(self)


   .. py:method:: list_datasets(self)



.. py:class:: ClusterBase

   Bases: :py:obj:`doc_utils.DocUtils`, :py:obj:`abc.ABC`

   Create an ABC

   .. py:method:: __call__(self, *args, **kwargs)


   .. py:method:: fit_transform(self, vectors) -> List[Union[str, float, int]]
      :abstractmethod:


   .. py:method:: _concat_vectors_from_list(self, list_of_vectors: list)

      Concatenate 2 vectors together in a pairwise fashion


   .. py:method:: _get_vectors_from_documents(self, vector_fields, docs, missing_treatment)


   .. py:method:: __getdoc__(self, documents)
      :abstractmethod:

      What you want in each doc



   .. py:method:: _bulk_get_doc(self, documents)
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


   .. py:method:: metadata(self)
      :property:

      You can also store the metadata of this clustering algorithm


   .. py:method:: _label_cluster(self, label: Union[int, str])


   .. py:method:: _label_clusters(self, labels)



.. py:data:: __version__
   :annotation: = 0.28.2

   

.. py:data:: pypi_data
   

   

