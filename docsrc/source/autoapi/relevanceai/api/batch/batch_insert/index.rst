:py:mod:`relevanceai.api.batch.batch_insert`
============================================

.. py:module:: relevanceai.api.batch.batch_insert

.. autoapi-nested-parse::

   Batch Insert



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.api.batch.batch_insert.BatchInsertClient




Attributes
~~~~~~~~~~

.. autoapisummary::

   relevanceai.api.batch.batch_insert.BYTE_TO_MB
   relevanceai.api.batch.batch_insert.LIST_SIZE_MULTIPLIER
   relevanceai.api.batch.batch_insert.SUCCESS_CODES
   relevanceai.api.batch.batch_insert.RETRY_CODES
   relevanceai.api.batch.batch_insert.HALF_CHUNK_CODES


.. py:data:: BYTE_TO_MB
   

   

.. py:data:: LIST_SIZE_MULTIPLIER
   :annotation: = 3

   

.. py:data:: SUCCESS_CODES
   :annotation: = [200]

   

.. py:data:: RETRY_CODES
   :annotation: = [400, 404]

   

.. py:data:: HALF_CHUNK_CODES
   :annotation: = [413, 524]

   

.. py:class:: BatchInsertClient(project, api_key)

   Bases: :py:obj:`relevanceai.utils.Utils`, :py:obj:`relevanceai.api.batch.batch_retrieve.BatchRetrieveClient`, :py:obj:`relevanceai.api.endpoints.client.APIClient`, :py:obj:`relevanceai.api.batch.chunk.Chunker`

   API Client

   .. py:method:: insert_documents(self, dataset_id: str, docs: list, bulk_fn: Callable = None, max_workers: int = 8, retry_chunk_mult: float = 0.5, show_progress_bar: bool = False, chunksize: int = 0, use_json_encoder: bool = True, *args, **kwargs)

      Insert a list of documents with multi-threading automatically enabled.

      - When inserting the document you can optionally specify your own id for a document by using the field name "_id", if not specified a random id is assigned.
      - When inserting or specifying vectors in a document use the suffix (ends with) "_vector_" for the field name. e.g. "product_description_vector_".
      - When inserting or specifying chunks in a document the suffix (ends with) "_chunk_" for the field name. e.g. "products_chunk_".
      - When inserting or specifying chunk vectors in a document's chunks use the suffix (ends with) "_chunkvector_" for the field name. e.g. "products_chunk_.product_description_chunkvector_".

      Documentation can be found here: https://ingest-api-dev-aueast.relevance.ai/latest/documentation#operation/InsertEncode

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param docs: A list of documents. Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
      :type docs: list
      :param bulk_fn: Function to apply to documents before uploading
      :type bulk_fn: callable
      :param max_workers: Number of workers active for multi-threading
      :type max_workers: int
      :param retry_chunk_mult: Multiplier to apply to chunksize if upload fails
      :type retry_chunk_mult: int
      :param chunksize: Number of documents to upload per worker. If None, it will default to the size specified in config.upload.target_chunk_mb
      :type chunksize: int
      :param use_json_encoder: Whether to automatically convert documents to json encodable format
      :type use_json_encoder: bool

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> df = client.Dataset("sample_dataset")
      >>> documents = [{"_id": "10", "value": 5}, {"_id": "332", "value": 10}]
      >>> df.insert_documents(documents)


   .. py:method:: insert_csv(self, dataset_id: str, filepath_or_buffer, chunksize: int = 10000, max_workers: int = 8, retry_chunk_mult: float = 0.5, show_progress_bar: bool = False, index_col: int = None, csv_args: dict = {}, col_for_id: str = None, auto_generate_id: bool = True)

      Insert data from csv file

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param filepath_or_buffer: Any valid string path is acceptable. The string could be a URL. Valid URL schemes include http, ftp, s3, gs, and file.
      :param chunksize: Number of lines to read from csv per iteration
      :type chunksize: int
      :param max_workers: Number of workers active for multi-threading
      :type max_workers: int
      :param retry_chunk_mult: Multiplier to apply to chunksize if upload fails
      :type retry_chunk_mult: int
      :param csv_args: Optional arguments to use when reading in csv. For more info, see https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
      :type csv_args: dict
      :param index_col: Optional argument to specify if there is an index column to be skipped (e.g. index_col = 0)
      :type index_col: None
      :param col_for_id: Optional argument to use when a specific field is supposed to be used as the unique identifier ('_id')
      :type col_for_id: str
      :param auto_generate_id: Automatically generateds UUID if auto_generate_id is True and if the '_id' field does not exist
      :type auto_generate_id: bool = True

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> df = client.Dataset("sample_dataset")
      >>> csv_filename = "temp.csv"
      >>> df.insert_csv(csv_filename)


   .. py:method:: _insert_csv_chunk(self, chunk, dataset_id, max_workers, retry_chunk_mult, show_progress_bar, col_for_id, auto_generate_id)


   .. py:method:: update_documents(self, dataset_id: str, docs: list, bulk_fn: Callable = None, max_workers: int = 8, retry_chunk_mult: float = 0.5, chunksize: int = 0, show_progress_bar=False, use_json_encoder: bool = True, *args, **kwargs)

      Update a list of documents with multi-threading automatically enabled.
      Edits documents by providing a key value pair of fields you are adding or changing, make sure to include the "_id" in the documents.

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> url = "https://api-aueast.relevance.ai/v1/"
      >>> collection = ""
      >>> project = ""
      >>> api_key = ""
      >>> client = Client(project, api_key)
      >>> docs = client.datasets.documents.get_where(collection, select_fields=['title'])
      >>> while len(docs['documents']) > 0:
      >>>     docs['documents'] = model.encode_documents_in_bulk(['product_name'], docs['documents'])
      >>>     client.update_documents(collection, docs['documents'])
      >>>     docs = client.datasets.documents.get_where(collection, select_fields=['product_name'], cursor=docs['cursor'])

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param docs: A list of documents. Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
      :type docs: list
      :param bulk_fn: Function to apply to documents before uploading
      :type bulk_fn: callable
      :param max_workers: Number of workers active for multi-threading
      :type max_workers: int
      :param retry_chunk_mult: Multiplier to apply to chunksize if upload fails
      :type retry_chunk_mult: int
      :param chunksize: Number of documents to upload per worker. If None, it will default to the size specified in config.upload.target_chunk_mb
      :type chunksize: int
      :param use_json_encoder: Whether to automatically convert documents to json encodable format
      :type use_json_encoder: bool


   .. py:method:: pull_update_push(self, dataset_id: str, update_function, updated_dataset_id: str = None, log_file: str = None, updating_args: dict = {}, retrieve_chunk_size: int = 100, max_workers: int = 8, filters: list = [], select_fields: list = [], show_progress_bar: bool = True, use_json_encoder: bool = True)

      Loops through every document in your collection and applies a function (that is specified by you) to the documents.
      These documents are then uploaded into either an updated collection, or back into the original collection.

      :param dataset_id: The dataset_id of the collection where your original documents are
      :type dataset_id: string
      :param update_function: A function created by you that converts documents in your original collection into the updated documents. The function must contain a field which takes in a list of documents from the original collection. The output of the function must be a list of updated documents.
      :type update_function: function
      :param updated_dataset_id: The dataset_id of the collection where your updated documents are uploaded into. If 'None', then your original collection will be updated.
      :type updated_dataset_id: string
      :param updating_args: Additional arguments to your update_function, if they exist. They must be in the format of {'Argument': Value}
      :type updating_args: dict
      :param retrieve_chunk_size: The number of documents that are received from the original collection with each loop iteration.
      :type retrieve_chunk_size: int
      :param max_workers: The number of processors you want to parallelize with
      :type max_workers: int
      :param max_error: How many failed uploads before the function breaks
      :type max_error: int
      :param json_encoder: Whether to automatically convert documents to json encodable format
      :type json_encoder: bool


   .. py:method:: pull_update_push_to_cloud(self, dataset_id: str, update_function, updated_dataset_id: str = None, logging_dataset_id: str = None, updating_args: dict = {}, retrieve_chunk_size: int = 100, retrieve_chunk_size_failure_retry_multiplier: float = 0.5, number_of_retrieve_retries: int = 3, max_workers: int = 8, max_error: int = 1000, filters: list = [], select_fields: list = [], show_progress_bar: bool = True, use_json_encoder: bool = True)

      Loops through every document in your collection and applies a function (that is specified by you) to the documents.
      These documents are then uploaded into either an updated collection, or back into the original collection.

      :param original_dataset_id: The dataset_id of the collection where your original documents are
      :type original_dataset_id: string
      :param logging_dataset_id: The dataset_id of the collection which logs which documents have been updated. If 'None', then one will be created for you.
      :type logging_dataset_id: string
      :param updated_dataset_id: The dataset_id of the collection where your updated documents are uploaded into. If 'None', then your original collection will be updated.
      :type updated_dataset_id: string
      :param update_function: A function created by you that converts documents in your original collection into the updated documents. The function must contain a field which takes in a list of documents from the original collection. The output of the function must be a list of updated documents.
      :type update_function: function
      :param updating_args: Additional arguments to your update_function, if they exist. They must be in the format of {'Argument': Value}
      :type updating_args: dict
      :param retrieve_chunk_size: The number of documents that are received from the original collection with each loop iteration.
      :type retrieve_chunk_size: int
      :param retrieve_chunk_size_failure_retry_multiplier: If fails, retry on each chunk
      :type retrieve_chunk_size_failure_retry_multiplier: int
      :param max_workers: The number of processors you want to parallelize with
      :type max_workers: int
      :param max_error: How many failed uploads before the function breaks
      :type max_error: int
      :param json_encoder: Whether to automatically convert documents to json encodable format
      :type json_encoder: bool


   .. py:method:: insert_df(self, dataset_id, dataframe, *args, **kwargs)

      Insert a dataframe for eachd doc


   .. py:method:: delete_pull_update_push_logs(self, dataset_id=False)


   .. py:method:: _write_documents(self, insert_function, docs: list, bulk_fn: Callable = None, max_workers: int = 8, retry_chunk_mult: float = 0.5, show_progress_bar: bool = False, chunksize: int = 0)


   .. py:method:: rename_fields(self, dataset_id: str, field_mappings: dict, retrieve_chunk_size: int = 100, max_workers: int = 8, show_progress_bar: bool = True)

      Loops through every document in your collection and renames specified fields by deleting the old one and
      creating a new field using the provided mapping
      These documents are then uploaded into either an updated collection, or back into the original collection.

      Example:
      rename_fields(dataset_id,field_mappings = {'a.b.d':'a.b.c'})  => doc['a']['b']['d'] => doc['a']['b']['c']
      rename_fields(dataset_id,field_mappings = {'a.b':'a.c'})  => doc['a']['b'] => doc['a']['c']

      :param dataset_id: The dataset_id of the collection where your original documents are
      :type dataset_id: string
      :param field_mappings: A dictionary in the form f {old_field_name1 : new_field_name1, ...}
      :type field_mappings: dict
      :param retrieve_chunk_size: The number of documents that are received from the original collection with each loop iteration.
      :type retrieve_chunk_size: int
      :param retrieve_chunk_size_failure_retry_multiplier: If fails, retry on each chunk
      :type retrieve_chunk_size_failure_retry_multiplier: int
      :param max_workers: The number of processors you want to parallelize with
      :type max_workers: int
      :param show_progress_bar: Shows a progress bar if True
      :type show_progress_bar: bool



