:py:mod:`relevanceai.dataset_api.dataset`
=========================================

.. py:module:: relevanceai.dataset_api.dataset

.. autoapi-nested-parse::

   Pandas like dataset API



Module Contents
---------------

.. py:class:: Series(project: str, api_key: str, dataset_id: str, field)



   A wrapper class for being able to vectorize documents over field

   .. py:method:: sample(self, n: int = 1, frac: float = None, filters: list = [], random_state: int = 0, output_format='pandas')

      Return a random sample of items from a dataset.

      :param n: Number of items to return. Cannot be used with frac.
      :type n: int
      :param frac: Fraction of items to return. Cannot be used with n.
      :type frac: float
      :param filters: Query for filtering the search results
      :type filters: list
      :param random_state: Random Seed for retrieving random documents.
      :type random_state: int


   .. py:method:: all(self, chunk_size: int = 1000, filters: List = [], sort: List = [], include_vector: bool = True, show_progress_bar: bool = True)


   .. py:method:: vectorize(self, model) -> None

      Vectorises over a field give a model architecture

      :param model: The dataset_id of concern
      :type model: Machine learning model for vectorizing text`


   .. py:method:: apply(self, func: Callable, output_field: str, axis: int = 0)

      Apply a function along an axis of the DataFrame.

      Objects passed to the function are Series objects whose index is either the DataFrame’s index (axis=0) or the DataFrame’s columns (axis=1). By default (result_type=None), the final return type is inferred from the return type of the applied function. Otherwise, it depends on the result_type argument.

      :param func: Function to apply to each document
      :type func: function
      :param axis: Axis along which the function is applied.
                   - 9 or 'index': apply function to each column
                   - 1 or 'columns': apply function to each row
      :type axis: int
      :param output_field: The field from which to output
      :type output_field: str

      .. rubric:: Example

      >>> df["sample_1_label"].apply(lambda x: x + 3)



.. py:class:: Dataset(project: str, api_key: str)



   A Pandas Like datatset API for interacting with the RelevanceAI python package

   .. py:method:: shape(self)
      :property:

      Returns the shape (N x C) of a dataset
      N = number of samples in the Dataset
      C = number of columns in the Dataset

      :returns: (N, C)
      :rtype: Tuple


   .. py:method:: info(self, dtype_count: bool = False) -> pandas.DataFrame

      Return a dictionary that contains information about the Dataset
      including the index dtype and columns and non-null values.

      :param dtype_count: If dtype_count is True, prints a value_counts of the data type
      :type dtype_count: bool

      :returns: Dictionary of information
      :rtype: Dict


   .. py:method:: head(self, n: int = 5, raw_json: bool = False, **kw) -> Union[dict, pandas.DataFrame]

      Return the first `n` rows.
      returns the first `n` rows of your dataset.
      It is useful for quickly testing if your object
      has the right type of data in it.

      :param n: Number of rows to select.
      :type n: int, default 5
      :param raw_json: If True, returns raw JSON and not Pandas Dataframe
      :type raw_json: bool
      :param kw: Additional arguments to feed into show_json

      :returns: The first 'n' rows of the caller object.
      :rtype: Pandas DataFrame or Dict, depending on args

      .. rubric:: Example

      >>> from relevanceai import Client, Dataset
      >>> client = Client()
      >>> df = client.Dataset("sample_dataset", image_fields=["image_url])
      >>> df.head()


   .. py:method:: describe(self) -> dict

      Descriptive statistics include those that summarize the central tendency
      dispersion and shape of a dataset's distribution, excluding NaN values.


   .. py:method:: vectorize(self, field, model)

      Vectorizes a Particular field (text) of the dataset

      :param field: The text field to select
      :type field: str
      :param model: a Type deep learning model that vectorizes text


   .. py:method:: cluster(self, field, n_clusters=10, overwrite=False)

      Performs KMeans Clustering on over a vector field within the dataset.

      :param field: The text field to select
      :type field: str
      :param n_cluster: the number of cluster to find wihtin the vector field
      :type n_cluster: int default = 10


   .. py:method:: sample(self, n: int = 0, frac: float = None, filters: list = [], random_state: int = 0, select_fields: list = [], output_format: str = 'json')

      Return a random sample of items from a dataset.

      :param n: Number of items to return. Cannot be used with frac.
      :type n: int
      :param frac: Fraction of items to return. Cannot be used with n.
      :type frac: float
      :param filters: Query for filtering the search results
      :type filters: list
      :param random_state: Random Seed for retrieving random documents.
      :type random_state: int
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list

      .. rubric:: Example

      >>> from relevanceai import Client, Dataset
      >>> client = Client()
      >>> df = client.Dataset("sample_dataset", image_fields=["image_url])
      >>> df.sample()


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


   .. py:method:: all(self, chunk_size: int = 1000, filters: List = [], sort: List = [], select_fields: List = [], include_vector: bool = True, show_progress_bar: bool = True)

      Retrieve all documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched. For more details see documents.get_where.

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


   .. py:method:: to_csv(self, filename: str, **kwargs)

      Download a dataset from the QC to a local .csv file

      :param filename: path to downloaded .csv file
      :type filename: str
      :param kwargs: see client.get_all_documents() for extra args
      :type kwargs: Optional


   .. py:method:: read_csv(self, filename: str, **kwargs)

      Wrapper for client.insert_csv

      :param filename: path to .csv file
      :type filename: str
      :param kwargs: see client.insert_csv() for extra args
      :type kwargs: Optional


   .. py:method:: cat(self, vector_name: Union[str, None] = None, fields: List = [])

      Concatenates numerical fields along an axis and reuploads this vector for other operations

      :param vector_name: name of the new concatenated vector field
      :type vector_name: str, default None
      :param fields: fields alone which the new vector will concatenate
      :type fields: List


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

      >>> from relevanceai import Client
      >>> client = Client()
      >>> documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
      >>> df = client.Dataset("sample")
      >>> df.create()


   .. py:method:: delete(self)

      Delete a dataset

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> documents = [{"_id": "321", "value": 10}, "_id": "4243", "value": 100]
      >>> df = client.Dataset("sample")
      >>> df.delete()


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


   .. py:method:: get(self, document_ids: Union[List, str], include_vector: bool = True)

      Retrieve a document by its ID ("_id" field). This will retrieve the document faster than a filter applied on the "_id" field.

      :param document_ids: ID of a document in a dataset.
      :type document_ids: Union[list, str]
      :param include_vector: Include vectors in the search results
      :type include_vector: bool

      .. rubric:: Example

      >>> from relevanceai import Client, Dataset
      >>> client = Client()
      >>> df = client.Dataset("sample_dataset")
      >>> df.get("sample_id", include_vector=False)


   .. py:method:: schema(self)

      Returns the schema of a dataset. Refer to datasets.create for different field types available in a VecDB schema.

      .. rubric:: Example

      >>> from relevanceai import Client
      >>> client = Client()
      >>> df = client.Dataset("sample")
      >>> df.schema()



.. py:class:: Datasets(project: str, api_key: str)



   Dataset class for multiple datasets


