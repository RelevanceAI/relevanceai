"""
All read operations for Dataset
"""
import re
import math
import warnings
import pandas as pd

from typing import Optional, Union, Dict, List
from relevanceai.client.helpers import Credentials

from relevanceai.operations.cluster.centroids import Centroids

from relevanceai.dataset.read.metadata import Metadata
from relevanceai.dataset.read.statistics import Statistics
from relevanceai.dataset.helpers import _build_filters

from relevanceai.utils.cache import lru_cache
from relevanceai.utils.decorators.analytics import track

from relevanceai.constants.constants import MAX_CACHESIZE
from relevanceai.constants.warning import Warning


class Read(Statistics):
    """

    Dataset Read
    -------------------

    A Pandas Like datatset API for interacting with the RelevanceAI python package
    """

    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
        fields: Optional[list] = None,
        image_fields: Optional[List[str]] = None,
        audio_fields: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None,
        highlight_fields: Optional[Dict[str, list]] = None,
        **kwargs,
    ):
        self.credentials = credentials
        self.fields = [] if fields is None else fields
        self.dataset_id = dataset_id
        self.centroids = Centroids(
            credentials=credentials,
            dataset_id=self.dataset_id,
        )
        self.image_fields = [] if image_fields is None else image_fields
        self.audio_fields = [] if audio_fields is None else audio_fields
        self.text_fields = [] if text_fields is None else text_fields
        self.highlight_fields = {} if highlight_fields is None else highlight_fields
        super().__init__(
            credentials=credentials,
            dataset_id=dataset_id,
            **kwargs,
        )

    @property  # type: ignore
    @track
    def shape(self):
        """
        Returns the shape (N x C) of a dataset
        N = number of samples in the Dataset
        C = number of columns in the Dataset

        Returns
        -------
        Tuple
            (N, C)

        Example
        ---------------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            length, width = df.shape
        """
        schema = self.datasets.schema(self.dataset_id)
        n_documents = self.get_number_of_documents(dataset_id=self.dataset_id)
        return (n_documents, len(schema))

    def _get_possible_dtypes(self, schema):
        possible_dtypes = []
        for v in schema.values():
            if isinstance(v, str):
                possible_dtypes.append(v)
            elif isinstance(v, dict):
                if list(v)[0] == "vector":
                    possible_dtypes.append("vector_")
        return possible_dtypes

    def _get_dtype_count(self, schema: dict):
        possible_dtypes = self._get_possible_dtypes(schema)
        dtypes = {
            dtype: list(schema.values()).count(dtype) for dtype in possible_dtypes
        }
        return dtypes

    def _get_schema(self):
        # stores schema in memory to save users API usage/reloading
        if hasattr(self, "_schema"):
            return self._schema
        self._schema = self.datasets.schema(self.dataset_id)
        return self._schema

    @track
    def info(self, dtype_count: bool = False) -> pd.DataFrame:
        """
        Return a dictionary that contains information about the Dataset
        including the index dtype and columns and non-null values.

        Parameters
        -----------
        dtype_count: bool
            If dtype_count is True, prints a value_counts of the data type


        Returns
        ---------
        pd.DataFrame
            a pandas dataframe of information

        Example
        ---------------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            df.info()
        """
        health: dict = self.datasets.monitor.health(self.dataset_id)
        schema: dict = self._get_schema()
        info_json = [
            {
                "Column": column,
                "Dtype": schema[column],
            }
            if column not in health
            else {
                "Column": column,
                "Non-Null Count": health[column]["missing"],
                "Dtype": schema[column],
            }
            for column in schema
        ]

        info_df = pd.DataFrame(info_json)
        if dtype_count:
            dtypes_info = self._get_dtype_count(schema)
            print(dtypes_info)
        return info_df

    @track
    def head(
        self, n: int = 5, raw_json: bool = False, **kw
    ) -> Union[dict, pd.DataFrame]:
        """
        Return the first `n` rows.
        returns the first `n` rows of your dataset.
        It is useful for quickly testing if your object
        has the right type of data in it.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.
        raw_json: bool
            If True, returns raw JSON and not Pandas Dataframe
        kw:
            Additional arguments to feed into show_json

        Returns
        -------
        Pandas DataFrame or Dict, depending on args
            The first 'n' rows of the caller object.

        Example
        ---------
        .. code-block::

            from relevanceai import Client

            client = Client()

            df = client.Dataset("sample_dataset_id", image_fields=["image_url])

            df.head()
        """
        print(
            f"https://cloud.relevance.ai/dataset/{self.dataset_id}/dashboard/data?page=1"
        )
        head_documents = self.get_documents(
            number_of_documents=n,
        )
        if raw_json:
            return head_documents
        else:
            try:
                return self._show_json(head_documents, **kw)
            except Exception as e:
                warnings.warn(Warning.MISSING_RELEVANCE_NOTEBOOK + str(e))
                return pd.json_normalize(head_documents).head(n=n)

    def _show_json(self, documents, **kw):
        from jsonshower import show_json

        if not self.text_fields:
            text_fields = pd.json_normalize(documents).columns.tolist()
        else:
            text_fields = self.text_fields
        return show_json(
            documents,
            image_fields=self.image_fields,
            audio_fields=self.audio_fields,
            highlight_fields=self.highlight_fields,
            text_fields=text_fields,
            **kw,
        )

    def _repr_html_(self):
        documents = self.get_documents()
        documents = [
            {
                "_id": document["_id"],
                "insert_date_": document["insert_date_"],
                **document,
            }
            for document in documents
        ]
        try:
            return self._show_json(documents, return_html=True)
        except Exception as e:
            warnings.warn(Warning.MISSING_RELEVANCE_NOTEBOOK + str(e))
            return pd.json_normalize(documents).set_index("_id")._repr_html_()

    @track
    def sample(
        self,
        n: int = 1,
        frac: float = None,
        filters: Optional[list] = None,
        random_state: int = 0,
        select_fields: Optional[list] = None,
        include_vector: bool = True,
        output_format: str = "json",
    ):
        """
        Return a random sample of items from a dataset.

        Parameters
        ----------
        n : int
            Number of items to return. Cannot be used with frac.
        frac: float
            Fraction of items to return. Cannot be used with n.
        filters: list
            Query for filtering the search results
        random_state: int
            Random Seed for retrieving random documents.
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id", image_fields=["image_url])
            df.sample()
        """
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        if not select_fields and self.fields:
            select_fields = self.fields

        if frac and n:
            raise ValueError("Only one of n or frac can be provided")

        if frac:
            if frac > 1 or frac < 0:
                raise ValueError("Fraction must be between 0 and 1")
            n = math.ceil(
                self.get_number_of_documents(self.dataset_id, filters=filters) * frac
            )

        documents = self.datasets.documents.get_where(
            dataset_id=self.dataset_id,
            filters=filters,
            page_size=n,
            random_state=random_state,
            is_random=True,
            select_fields=select_fields,
            include_vector=include_vector,
        )["documents"]
        if output_format == "json":
            return documents
        elif output_format == "pandas":
            return pd.DataFrame.from_dict(documents, orient="records")

    @lru_cache(maxsize=MAX_CACHESIZE)
    @track
    def get_all_documents(
        self,
        chunksize: int = 1000,
        filters: Optional[List] = None,
        sort: Optional[List] = None,
        select_fields: Optional[List] = None,
        include_vector: bool = True,
        show_progress_bar: bool = True,
    ):
        """
        Retrieve all documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched. For more details see documents.get_where.

        Parameters
        ------------
        chunksize: list
            Number of documents to retrieve per retrieval
        include_vector: bool
            Include vectors in the search results
        sort: list
            Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
        filters: list
            Query for filtering the search results
        select_fields : list
            Fields to include in the search results, empty array/list means all fields.

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            documents = df.get_all_documents()

        """
        filters = [] if filters is None else filters
        sort = [] if sort is None else sort
        select_fields = [] if select_fields is None else select_fields

        return self._get_all_documents(
            dataset_id=self.dataset_id,
            chunksize=chunksize,
            filters=filters,
            sort=sort,
            select_fields=select_fields,
            include_vector=include_vector,
            show_progress_bar=show_progress_bar,
        )

    @track
    def get_documents_by_ids(
        self, document_ids: Union[List, str], include_vector: bool = True
    ):
        """
        Retrieve a document by its ID ("_id" field). This will retrieve the document faster than a filter applied on the "_id" field.

        Parameters
        ----------
        document_ids: Union[list, str]
            ID of a document in a dataset.
        include_vector: bool
            Include vectors in the search results

        Example
        --------

        .. code-block::

            from relevanceai import Client, Dataset
            client = Client()
            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            df.get_documents_by_ids(["sample_id"], include_vector=False)
        """
        if isinstance(document_ids, str):
            return self.datasets.documents.get(
                self.dataset_id, id=document_ids, include_vector=include_vector
            )
        elif isinstance(document_ids, list):
            return self.datasets.documents.bulk_get(
                self.dataset_id, ids=document_ids, include_vector=include_vector
            )
        raise TypeError("Document IDs needs to be a string or a list")

    @track
    def get(self, document_ids: Union[List, str], include_vector: bool = True):
        """
        Retrieve a document by its ID ("_id" field). This will retrieve the document faster than a filter applied on the "_id" field.
        This has the same functionality as get_document_by_ids.

        Parameters
        ----------
        document_ids: Union[list, str]
            ID of a document in a dataset.
        include_vector: bool
            Include vectors in the search results

        Example
        --------

        .. code-block::

            from relevanceai import Client
            client = Client()
            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            df.get(["sample_id"], include_vector=False)
        """
        if isinstance(document_ids, str):
            return self.datasets.documents.get(
                self.dataset_id, id=document_ids, include_vector=include_vector
            )
        elif isinstance(document_ids, list):
            return self.datasets.documents.bulk_get(
                self.dataset_id, ids=document_ids, include_vector=include_vector
            )
        raise TypeError("Document IDs needs to be a string or a list")

    @property  # type: ignore
    @track
    def schema(self) -> Dict:
        """
        Returns the schema of a dataset. Refer to datasets.create for different field types available in a VecDB schema.

        Example
        -----------------

        .. code-block::

            from relevanceai import Client
            client = Client()
            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            df.schema
        """
        return self.datasets.schema(self.dataset_id)

    @property  # type: ignore
    @track
    def columns(self) -> List[str]:
        """
        Returns a list of columns

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            df.columns

        """
        return list(self.schema)

    @track
    def filter(
        self,
        index: Union[str, None] = None,
        items: Union[List, None] = None,
        like: Union[str, None] = None,
        regex: Union[str, None] = None,
        axis: Union[int, str] = 0,
    ):
        """
        Returns a subset of the dataset, filtered by the parameters given

        Parameters
        ----------
        items : str, default None
            the column on which to filter, if None then defaults to the _id column
        items : list-like
            Keep labels from axis which are in items.
        like : str
            Keep labels from axis for which "like in label == True".
        regex : str (regular expression)
            Keep labels from axis for which re.search(regex, label) == True.
        axis : {0 or `index`, 1 or `columns`},
            The axis on which to perform the search

        Returns
        ---------
        list of documents

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("ecommerce-example-encoded")
            filtered = df.filter(items=["product_title", "query", "product_price"])
            filtered = df.filter(index="query", like="routers")
            filtered = df.filter(index="product_title", regex=".*Hard.*Drive.*")

        """
        fields = []
        filters = []

        schema = list(self.schema)

        if index:
            axis = 0
        else:
            axis = 1
            index = "_id"

        rows = axis in [0, "index"]
        columns = axis in [1, "columns"]

        if items is not None:
            if columns:
                fields += items

            elif rows:
                filters += _build_filters(items, filter_type="exact_match", index=index)

        elif like:
            if columns:
                fields += [column for column in schema if like in column]

            elif rows:
                filters += _build_filters(like, filter_type="contains", index=index)

        elif regex:
            if columns:
                query = re.compile(regex)
                re_fields = list(filter(query.match, schema))
                fields += re_fields

            elif rows:
                filters += _build_filters(regex, filter_type="regexp", index=index)

        else:
            raise TypeError("Must pass either `items`, `like` or `regex`")

        filters = [{"filter_type": "or", "condition_value": filters}]

        return self.get_all_documents(select_fields=fields, filters=filters)

    @track
    def get_documents(
        self,
        number_of_documents: int = 20,
        filters: Optional[list] = None,
        cursor: str = None,
        batch_size: int = 1000,
        sort: Optional[list] = None,
        select_fields: Optional[list] = None,
        include_vector: bool = True,
        include_cursor: bool = False,
    ):
        """
        Retrieve documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched. \n
        If you are looking to combine your filters with multiple ORs, simply add the following inside the query {"strict":"must_or"}.
        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        number_of_documents: int
            Number of documents to retrieve
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.
        cursor: string
            Cursor to paginate the document retrieval
        batch_size: int
            Number of documents to retrieve per iteration
        include_vector: bool
            Include vectors in the search results
        sort: list
            Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
        filters: list
            Query for filtering the search results
        """
        filters = [] if filters is None else filters
        sort = [] if sort is None else sort
        select_fields = [] if select_fields is None else select_fields

        return self._get_documents(
            dataset_id=self.dataset_id,
            number_of_documents=number_of_documents,
            filters=filters,
            cursor=cursor,
            batch_size=batch_size,
            sort=sort,
            select_fields=select_fields,
            include_vector=include_vector,
            include_cursor=include_cursor,
        )

    def get_metadata(self):
        """
        Store Metadata
        """
        return self.datasets.metadata(self.dataset_id)["results"]

    @property
    def metadata(self):
        """Get the metadata"""
        _metadata = self.get_metadata()
        self._metadata = Metadata(_metadata, self.credentials, self.dataset_id)
        return self._metadata

    def insert_metadata(self, metadata: dict):
        """Insert metadata"""
        results = self.datasets.post_metadata(self.dataset_id, metadata)
        if results == {}:
            print("✅ You have successfully inserted data.")
        else:
            return results

    def upsert_metadata(self, metadata: dict):
        """Upsert metadata."""
        original_metadata: dict = self.get_metadata()
        original_metadata.update(metadata)
        results = self.datasets.post_metadata(self.dataset_id, metadata)
        if results == {}:
            print("✅ You have successfully inserted metadata.")
        else:
            return results
        return self.insert_metadata(metadata)

    def chunk_dataset(
        self, select_fields: List = None, chunksize: int = 100, filters: list = None
    ):
        """

        Function for chunking a dataset

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            for c in ds.chunk_dataset(
                select_fields=["sample_label"],
                chunksize=100
            ):
                # Returns a dictionary with 'cursor' and 'documents' keys
                docs = c['documents']
                cursor = c['cursor']
                for d in docs:
                    d.update({"value": 3})
                ds.upsert_documents(docs)

        """
        docs = self.get_documents(
            number_of_documents=chunksize,
            include_cursor=True,
            filters=filters,
            select_fields=select_fields,
        )
        while len(docs["documents"]) > 0:
            yield docs["documents"]
            docs = self.get_documents(
                number_of_documents=chunksize,
                include_cursor=True,
                cursor=docs["cursor"],
                filters=filters,
            )
        return

    def list_vector_fields(self):
        """
        Returns list of valid vector fields in dataset
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("_mock_dataset_")
            ds.list_vector_fields()

        """
        schema = self.datasets.schema(self.dataset_id)
        return [
            k for k in schema.keys() if k.endswith("_vector_") and "_cluster_" not in k
        ]

    def list_cluster_aliases(self):
        raise NotImplementedError()
