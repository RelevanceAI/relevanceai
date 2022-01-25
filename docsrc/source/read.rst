Read (Dataset)
================
..
   Manually maintained. Relevant functions are copied from docsrc/source/autoapi/relevanceai/dataset_api/dataset/index.rst

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

.. py:method:: sample(self, n: int = 0, frac: float = None, filters: list = [], random_state: int = 0, select_fields: list = [])

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


.. py:method:: list_datasets(self) -> List

    .. rubric:: Example

    >>> from relevanceai import Client
    >>> client = Client()
    >>> client.list_datasets()
