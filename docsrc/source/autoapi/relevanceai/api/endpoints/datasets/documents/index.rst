:py:mod:`relevanceai.api.endpoints.datasets.documents`
======================================================

.. py:module:: relevanceai.api.endpoints.datasets.documents


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   relevanceai.api.endpoints.datasets.documents.DocumentsClient




.. py:class:: DocumentsClient(project, api_key)

   Bases: :py:obj:`relevanceai.base._Base`

   Base class for all relevanceai client utilities

   .. py:method:: list(self, dataset_id: str, select_fields=[], cursor: str = None, page_size: int = 20, include_vector: bool = True, random_state: int = 0)

      Retrieve documents from a specified dataset. Cursor is provided to retrieve even more documents. Loop through it to retrieve all documents in the dataset.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list
      :param page_size: Size of each page of results
      :type page_size: int
      :param cursor: Cursor to paginate the document retrieval
      :type cursor: string
      :param include_vector: Include vectors in the search results
      :type include_vector: bool
      :param random_state: Random Seed for retrieving random documents.
      :type random_state: int


   .. py:method:: get(self, dataset_id: str, id: str, include_vector: bool = True)

      Retrieve a document by its ID ("_id" field). This will retrieve the document faster than a filter applied on the "_id" field.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param id: ID of a document in a dataset.
      :type id: string
      :param include_vector: Include vectors in the search results
      :type include_vector: bool


   .. py:method:: bulk_get(self, dataset_id: str, ids: List, include_vector: bool = True, select_fields: List = [])

      Retrieve a document by its ID ("_id" field). This will retrieve the document faster than a filter applied on the "_id" field.

      For single id lookup version of this request use datasets.documents.get.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param ids: IDs of documents in the dataset.
      :type ids: list
      :param include_vector: Include vectors in the search results
      :type include_vector: bool
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list


   .. py:method:: get_where(self, dataset_id: str, filters: list = [], cursor: str = None, page_size: int = 20, sort: list = [], select_fields: list = [], include_vector: bool = True, random_state: int = 0, is_random: bool = False)

      Retrieve documents with filters. Cursor is provided to retrieve even more documents. Loop through it to retrieve all documents in the database. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched.


      The filters query is a json body that follows the schema of:

      >>> [
      >>>    {'field' : <field to filter>, 'filter_type' : <type of filter>, "condition":"==", "condition_value":"america"},
      >>>    {'field' : <field to filter>, 'filter_type' : <type of filter>, "condition":">=", "condition_value":90},
      >>> ]

      These are the available filter_type types: ["contains", "category", "categories", "exists", "date", "numeric", "ids"]


      "contains": for filtering documents that contains a string

      >>> {'field' : 'item_brand', 'filter_type' : 'contains', "condition":"==", "condition_value": "samsu"}

      "exact_match"/"category": for filtering documents that matches a string or list of strings exactly.

      >>> {'field' : 'item_brand', 'filter_type' : 'category', "condition":"==", "condition_value": "sumsung"}

      "categories": for filtering documents that contains any of a category from a list of categories.

      >>> {'field' : 'item_category_tags', 'filter_type' : 'categories', "condition":"==", "condition_value": ["tv", "smart", "bluetooth_compatible"]}

      "exists": for filtering documents that contains a field.

      >>> {'field' : 'purchased', 'filter_type' : 'exists', "condition":"==", "condition_value":" "}

      If you are looking to filter for documents where a field doesn't exist, run this:

      >>> {'field' : 'purchased', 'filter_type' : 'exists', "condition":"!=", "condition_value":" "}

      "date": for filtering date by date range.

      >>> {'field' : 'insert_date_', 'filter_type' : 'date', "condition":">=", "condition_value":"2020-01-01"}

      "numeric": for filtering by numeric range.

      >>> {'field' : 'price', 'filter_type' : 'numeric', "condition":">=", "condition_value":90}

      "ids": for filtering by document ids.

      >>> {'field' : 'ids', 'filter_type' : 'ids', "condition":"==", "condition_value":["1", "10"]}

      These are the available conditions:

      >>> "==", "!=", ">=", ">", "<", "<="

      If you are looking to combine your filters with multiple ORs, simply add the following inside the query {"strict":"must_or"}.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list
      :param cursor: Cursor to paginate the document retrieval
      :type cursor: string
      :param page_size: Size of each page of results
      :type page_size: int
      :param include_vector: Include vectors in the search results
      :type include_vector: bool
      :param sort: Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
      :type sort: list
      :param filters: Query for filtering the search results
      :type filters: list
      :param is_random: If True, retrieves doucments randomly. Cannot be used with cursor.
      :type is_random: bool
      :param random_state: Random Seed for retrieving random documents.
      :type random_state: int


   .. py:method:: paginate(self, dataset_id: str, page: int = 1, page_size: int = 20, include_vector: bool = True, select_fields: list = [])

      Retrieve documents with filters and support for pagination.

      For more information about filters check out datasets.documents.get_where.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param page: Page of the results
      :type page: int
      :param page_size: Size of each page of results
      :type page_size: int
      :param include_vector: Include vectors in the search results
      :type include_vector: bool
      :param select_fields: Fields to include in the search results, empty array/list means all fields.
      :type select_fields: list


   .. py:method:: update(self, dataset_id: str, update: dict, insert_date: bool = True)

      Edits documents by providing a key value pair of fields you are adding or changing, make sure to include the "_id" in the documents.

      For update multiple documents refer to datasets.documents.bulk_update

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param update: A dictionary to edit and add fields to a document. It should be specified in a format of {"field_name": "value"}. e.g. {"item.status" : "Sold Out"}
      :type update: list
      :param insert_date: Whether to include insert date as a field 'insert_date_'.
      :type insert_date: bool


   .. py:method:: update_where(self, dataset_id: str, update: dict, filters: list = [])

      Updates documents by filters. The updates to make to the documents that is returned by a filter.

      For more information about filters refer to datasets.documents.get_where.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param update: A dictionary to edit and add fields to a document. It should be specified in a format of {"field_name": "value"}. e.g. {"item.status" : "Sold Out"}
      :type update: list
      :param filters: Query for filtering the search results
      :type filters: list


   .. py:method:: bulk_update(self, dataset_id: str, updates: list, insert_date: bool = True, return_documents: bool = False)

      Edits documents by providing a key value pair of fields you are adding or changing, make sure to include the "_id" in the documents.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param updates: Updates to make to the documents. It should be specified in a format of {"field_name": "value"}. e.g. {"item.status" : "Sold Out"}
      :type updates: list
      :param insert_date: Whether to include insert date as a field 'insert_date_'.
      :type insert_date: bool
      :param include_updated_ids: Include the inserted IDs in the response
      :type include_updated_ids: bool


   .. py:method:: delete(self, dataset_id: str, id: str)

      Delete a document by ID.

      For deleting multiple documents refer to datasets.documents.bulk_delete

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param id: ID of document to delete
      :type id: string


   .. py:method:: delete_where(self, dataset_id: str, filters: list)

      Delete a document by filters.

      For more information about filters refer to datasets.documents.get_where.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param filters: Query for filtering the search results
      :type filters: list


   .. py:method:: bulk_delete(self, dataset_id: str, ids: list = [])

      Delete a list of documents by their IDs.

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param ids: IDs of documents to delete
      :type ids: list


   .. py:method:: delete_fields(self, dataset_id: str, id: str, fields: list)

      Delete fields in a document in a dataset by its id

      :param dataset_id: Unique name of dataset
      :type dataset_id: string
      :param id: ID of a document in a dataset
      :type id: string
      :param fields: List of fields to delete in a document
      :type fields: list



