..
   Manually maintained. Relevant functions are copied from docsrc/source/autoapi/relevanceai/dataset_api/dataset/index.rst

Create
=============================

Creating via upsert (preferred)
--------------------------------------

Upserting means that the document is updated if it is there but will not
overwrite otherwise it will insert if it is not.

.. py:method:: upsert_documents(self, documents: list, bulk_fn: Callable = None, max_workers: int = 8, retry_chunk_mult: float = 0.5, chunksize: int = 0, show_progress_bar=False, use_json_encoder: bool = True)

    Update a list of documents with multi-threading automatically enabled.
    Edits documents by providing a key value pair of fields you are adding or changing, make sure to include the "_id" in the documents.


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
    >>> documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
    >>> df = client.Dataset("sample")
    >>> df.upsert(dataset_id, documents)



Creating (without insertion)
--------------------------------------

Sometimes if the automatic schema detection is not working appropriately, it may
be appropriate to specify the schema yourself. In this cases, you can use this

   .. py:method:: create(self, schema: dict = {})

      A dataset can store documents to be searched, retrieved, filtered and aggregated (similar to Collections in MongoDB, Tables in SQL, Indexes in ElasticSearch).
      A powerful and core feature of VecDB is that you can store both your metadata and vectors in the same document. When specifying the schema of a dataset and inserting your own vector use the suffix (ends with) "_vector_" for the field name, and specify the length of the vector in dataset_schema.


      For example:

      >>>    {
      >>>        "product_image_vector_": 1024,
      >>>        "product_text_description_vector_" : 128
      >>>    }

      These are the field types supported in our datasets: ["text", "numeric", "date", "dict", "chunks", "vector", "chunkvector"].


      For example:

      >>>    {
      >>>        "product_text_description" : "text",
      >>>        "price" : "numeric",
      >>>        "created_date" : "date",
      >>>        "product_texts_chunk_": "chunks",
      >>>        "product_text_chunkvector_" : 1024
      >>>    }

      You don't have to specify the schema of every single field when creating a dataset, as VecDB will automatically detect the appropriate data type for each field (vectors will be automatically identified by its "_vector_" suffix). Infact you also don't always have to use this endpoint to create a dataset as /datasets/bulk_insert will infer and create the dataset and schema as you insert new documents.


      .. note::

         - A dataset name/id can only contain undercase letters, dash, underscore and numbers.
         - "_id" is reserved as the key and id of a document.
         - Once a schema is set for a dataset it cannot be altered. If it has to be altered, utlise the copy dataset endpoint.

      For more information about vectors check out the 'Vectorizing' section, services.search.vector or out blog at https://relevance.ai/blog. For more information about chunks and chunk vectors check out services.search.chunk.

      :param schema: Schema for specifying the field that are vectors and its length
      :type schema: dict

      .. rubric:: Example

      from relevanceai import Client
      client = Client()
      documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
      df = client.Dataset("sample")
      df.create()

Creating via insertion (overwrite)
------------------------------------------

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


Inserting a CSV
--------------------

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

Insertion vs upsert
-----------------------------

Users can choose to insert or they can upsert. The key difference between the 
2 is that `insert` will overwrite the document if the ID of the document is the
same whereas `upsert` will cerate a separte document if the ID of the document
is different.

Apply (Write and Update)
-----------------------------

.. py:method:: apply(self, func: Callable, retrieve_chunk_size: int = 100, max_workers: int = 8, filters: list = [], select_fields: list = [], show_progress_bar: bool = True, use_json_encoder: bool = True, axis: int = 0)

    Apply a function along an axis of the DataFrame.

    Objects passed to the function are Series objects whose index is either the DataFrame’s index (axis=0) or the DataFrame’s columns (axis=1). By default (result_type=None), the final return type is inferred from the return type of the applied function. Otherwise, it depends on the result_type argument.

    :param func: Function to apply to each document
    :type func: function
    :param retrieve_chunk_size: The number of documents that are received from the original collection with each loop iteration.
    :type retrieve_chunk_size: int
    :param max_workers: The number of processors you want to parallelize with
    :type max_workers: int
    :param max_error: How many failed uploads before the function breaks
    :type max_error: int
    :param json_encoder: Whether to automatically convert documents to json encodable format
    :type json_encoder: bool
    :param axis: Axis along which the function is applied.
                - 9 or 'index': apply function to each column
                - 1 or 'columns': apply function to each row
    :type axis: int

    .. rubric:: Example

    >>> from relevanceai import Client
    >>> client = Client()
    >>> df = client.Dataset("sample_dataset")
    >>> def update_doc(doc):
    >>>     doc["value"] = 2
    >>>     return doc
    >>> df.apply(update_doc)

Bulk Apply (Bulk Write and Update)
---------------------------------------

.. py:method:: bulk_apply(self, bulk_func: Callable, retrieve_chunk_size: int = 100, max_workers: int = 8, filters: list = [], select_fields: list = [], show_progress_bar: bool = True, use_json_encoder: bool = True)

    Apply a bulk function along an axis of the DataFrame.

    :param bulk_func: Function to apply to a bunch of documents at a time
    :type bulk_func: function
    :param retrieve_chunk_size: The number of documents that are received from the original collection with each loop iteration.
    :type retrieve_chunk_size: int
    :param max_workers: The number of processors you want to parallelize with
    :type max_workers: int
    :param max_error: How many failed uploads before the function breaks
    :type max_error: int
    :param json_encoder: Whether to automatically convert documents to json encodable format
    :type json_encoder: bool
    :param axis: Axis along which the function is applied.
                - 9 or 'index': apply function to each column
                - 1 or 'columns': apply function to each row
    :type axis: int

    .. rubric:: Example

    >>> from relevanceai import Client
    >>> client = Client()
    >>> df = client.Dataset("sample_dataset")
    >>> def update_documents(document):
            for d in documents:
    >>>         d["value"] = 10
    >>>     return documents
    >>> df.apply(update_documents)


Delete
-------

.. rubric:: Example

>>> from relevanceai import Client
>>> client = Client()
>>> documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
>>> df = client.Dataset("sample")
>>> df.delete()

Reference
-----------
