:py:mod:`relevanceai.api.batch.batch_retrieve`
==============================================

.. py:module:: relevanceai.api.batch.batch_retrieve

.. autoapi-nested-parse::

   Batch Retrieve



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.api.batch.batch_retrieve.BatchRetrieveClient




Attributes
~~~~~~~~~~

.. autoapisummary::

   relevanceai.api.batch.batch_retrieve.BYTE_TO_MB
   relevanceai.api.batch.batch_retrieve.LIST_SIZE_MULTIPLIER


.. py:data:: BYTE_TO_MB
   

   

.. py:data:: LIST_SIZE_MULTIPLIER
   :annotation: = 3

   

.. py:class:: BatchRetrieveClient(project: str, api_key: str)

   Bases: :py:obj:`relevanceai.api.endpoints.client.APIClient`, :py:obj:`relevanceai.api.batch.chunk.Chunker`

   API Client

   .. py:method:: get_documents(self, dataset_id: str, number_of_documents: int = 20, filters: list = [], cursor: str = None, batch_size: int = 1000, sort: list = [], select_fields: list = [], include_vector: bool = True)

      Retrieve documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched.

      If you are looking to combine your filters with multiple ORs, simply add the following inside the query {"strict":"must_or"}.
      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param number_of_documents: Number of documents to retrieve
      :type number_of_documents: int
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list
      :param cursor: Cursor to paginate the document retrieval
      :type cursor: string
      :param batch_size: Number of documents to retrieve per iteration
      :type batch_size: int
      :param include_vector: Include vectors in the search results
      :type include_vector: bool
      :param sort: Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
      :type sort: list
      :param filters: Query for filtering the search results
      :type filters: list


   .. py:method:: get_all_documents(self, dataset_id: str, chunk_size: int = 1000, filters: List = [], sort: List = [], select_fields: List = [], include_vector: bool = True, show_progress_bar: bool = True)

      Retrieve all documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched. For more details see documents.get_where.

      .. rubric:: Example

      >>> client = Client()
      >>> client.get_all_documents(dataset_id="sample_dataset"")

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param chunk_size: Number of documents to retrieve per retrieval
      :type chunk_size: list
      :param include_vector: Include vectors in the search results
      :type include_vector: bool
      :param sort: Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
      :type sort: list
      :param filters: Query for filtering the search results
      :type filters: list
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list


   .. py:method:: get_number_of_documents(self, dataset_id, filters=[])

      Get number of documents in a dataset. Filter can be used to select documents that match the conditions set in a filter query. For more details see documents.get_where.

      :param dataset_ids: Unique names of datasets
      :type dataset_ids: list
      :param filters: Filters to select documents
      :type filters: list


   .. py:method:: get_vector_fields(self, dataset_id)

      Returns list of valid vector fields in dataset
      :param dataset_id: Unique name of dataset
      :type dataset_id: string



