"""
Pandas like dataset API
"""
import warnings
import pandas as pd
from relevanceai.dataset_api.groupby import Groupby, Agg
from relevanceai.dataset_api.centroids import Centroids
from typing import List, Union, Callable
import math

from relevanceai.vector_tools.client import VectorTools
from relevanceai.api.client import BatchAPIClient


class Series(BatchAPIClient):
    """
    A wrapper class for being able to vectorize documents over field
    """

    def __init__(self, project: str, api_key: str, dataset_id: str, field) -> None:
        """
        Initialise the class
        """
        self.project = project
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.field = field
        super().__init__(project=project, api_key=api_key)

    def sample(
        self, n: int = 0, frac: float = None, filters: list = [], random_state: int = 0
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

        """
        select_fields = [self.field] if isinstance(self.field, str) else self.field
        return Dataset(self.project, self.api_key)(self.dataset_id).sample(
            n=n,
            frac=frac,
            filters=filters,
            random_state=random_state,
            select_fields=select_fields,
        )

    def all(
        self,
        chunk_size: int = 1000,
        filters: List = [],
        sort: List = [],
        include_vector: bool = True,
        show_progress_bar: bool = True,
    ):
        select_fields = [self.field] if isinstance(self.field, str) else self.field
        return Dataset(self.project, self.api_key)(self.dataset_id).all(
            chunk_size=chunk_size,
            filters=filters,
            sort=sort,
            select_fields=select_fields,
            include_vector=include_vector,
            show_progress_bar=show_progress_bar,
        )

    def vectorize(self, model) -> None:
        """
        Vectorises over a field give a model architecture

        Parameters
        ----------
        model : Machine learning model for vectorizing text`
            The dataset_id of concern
        """
        if hasattr(model, "encode_documents"):

            def encode_documents(documents):
                return model.encode_documents(self.field, documents)

        else:

            def encode_documents(documents):
                return model([self.field], documents)

        self.client.pull_update_push(self.dataset_id, encode_documents)

    def apply(
        self,
        func: Callable,
        axis: int = 0,
    ):
        """
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is either the DataFrame’s index (axis=0) or the DataFrame’s columns (axis=1). By default (result_type=None), the final return type is inferred from the return type of the applied function. Otherwise, it depends on the result_type argument.

        Parameters
        --------------
        func: function
            Function to apply to each document

        axis: int
            Axis along which the function is applied.
            - 9 or 'index': apply function to each column
            - 1 or 'columns': apply function to each row

        Example
        ---------------

        >>> df["sample_1_label"].apply(lambda x: x + 3)

        """
        if axis == 1:
            raise ValueError("We do not support column-wise operations!")

        def bulk_fn(documents):
            for d in documents:
                try:
                    if self.is_field(self.field, d):
                        self.set_field(
                            self.field, d, func(self.get_field(self.field, d))
                        )
                except Exception as e:
                    continue
            return documents

        return self.pull_update_push(
            self.dataset_id, bulk_fn, select_fields=[self.field]
        )

    def __getitem__(self, loc: Union[int, str]):
        if isinstance(loc, int):
            warnings.warn(
                "Integer selection of dataframe is not stable at the moment. Please use a string ID if possible to ensure exact selection."
            )
            return self.get_documents(
                self.dataset_id, loc + 1, select_fields=[self.field]
            )[loc][self.field]
        elif isinstance(loc, str):
            return self.datasets.documents.get(self.dataset_id, loc)[self.field]
        raise TypeError("Incorrect data type! Must be a string or an integer")


class Dataset(BatchAPIClient):
    """
    A Pandas Like datatset API for interacting with the RelevanceAI python package
    """

    def __init__(self, project: str, api_key: str) -> None:
        """
        Initialise the class
        """
        self.project = project
        self.api_key = api_key
        self.vector_tools = VectorTools(project=project, api_key=api_key)
        super().__init__(project=project, api_key=api_key)

    def __call__(
        self,
        dataset_id: str,
        image_fields: List = [],
        text_fields: List = [],
        audio_fields: List = [],
        highlight_fields: dict = {},
        output_format: str = "pandas",
    ):
        """
        Instaniates a Dataset

        Parameters
        ----------
        dataset_id : str
            The dataset_id of concern
        image_fields : str
            The image_fields within the dataset that you would like to select
        text_fields : str
            The text_fields within the dataset that you would like to select
        audio_fields : str
            The audio_fields within the dataset that you would like to select
        output_format : str
            The output format of the dataset

        Returns
        -------
        Self
        """
        self.dataset_id = dataset_id
        self.image_fields = image_fields
        self.text_fields = text_fields
        self.audio_fields = audio_fields
        self.highlight_fields = highlight_fields
        self.output_format = output_format
        self.groupby = Groupby(self.project, self.api_key, self.dataset_id)
        self.agg = Agg(self.project, self.api_key, self.dataset_id)
        self.centroids = Centroids(self.project, self.api_key, self.dataset_id)

        return self

    @property
    def shape(self):
        """
        Returns the shape (N x C) of a dataset
        N = number of samples in the Dataset
        C = number of columns in the Dataset

        Returns
        -------
        Tuple
            (N, C)
        """
        schema = self.datasets.schema(self.dataset_id)
        n_documents = self.get_number_of_documents(dataset_id=self.dataset_id)
        return (n_documents, len(schema))

    def __getitem__(self, field):
        """
        Returns a Series Object that selects a particular field within a dataset

        Parameters
        ----------
        field
            the particular field within the dataset

        Returns
        -------
        Tuple
            (N, C)
        """
        return Series(self.project, self.api_key, self.dataset_id, field)

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
        Dict
            Dictionary of information
        """
        health: dict = self.datasets.monitor.health(self.dataset_id)
        schema: dict = self._get_schema()
        info_df = pd.DataFrame()
        info_df["Non-Null Count"] = [health[key]["missing"] for key in schema]
        info_df["Dtype"] = [schema[key] for key in schema]
        if dtype_count:
            dtypes_info = self._get_dtype_count(schema)
            print(dtypes_info)
        return info_df

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
        """
        head_documents = self.get_documents(
            dataset_id=self.dataset_id,
            number_of_documents=n,
        )
        if raw_json:
            return head_documents
        else:
            try:
                return self._show_json(head_documents, **kw)
            except Exception as e:
                warnings.warn("Displaying using Pandas." + str(e))
                return pd.json_normalize(head_documents).head(n=n)

    def _show_json(self, docs, **kw):
        from jsonshower import show_json

        if not self.text_fields:
            text_fields = pd.json_normalize(docs).columns.tolist()
        else:
            text_fields = self.text_fields
        return show_json(
            docs,
            image_fields=self.image_fields,
            audio_fields=self.audio_fields,
            highlight_fields=self.highlight_fields,
            text_fields=text_fields,
        )

    def describe(self) -> dict:
        """
        Descriptive statistics include those that summarize the central tendency
        dispersion and shape of a dataset's distribution, excluding NaN values.
        """
        return self.datasets.facets(self.dataset_id)

    def vectorize(self, field, model):
        """
        Vectorizes a Particular field (text) of the dataset

        Parameters
        ----------
        field : str
            The text field to select
        model
            a Type deep learning model that vectorizes text
        """
        series = Series(self)
        series(self.dataset_id, field).vectorize(model)

    def cluster(self, field, n_clusters=10, overwrite=False):
        """
        Performs KMeans Clustering on over a vector field within the dataset.

        Parameters
        ----------
        field : str
            The text field to select
        n_cluster: int default = 10
            the number of cluster to find wihtin the vector field
        """
        centroids = self.vector_tools.cluster.kmeans_cluster(
            dataset_id=self.dataset_id,
            vector_fields=[field],
            k=n_clusters,
            alias=f"kmeans_{n_clusters}",
            overwrite=overwrite,
        )
        return centroids

    def sample(
        self,
        n: int = 0,
        frac: float = None,
        filters: list = [],
        random_state: int = 0,
        select_fields: list = [],
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

        """
        if n == 0 and frac is None:
            raise ValueError("Must provide one of n or frac")

        if frac and n:
            raise ValueError("Only one of n or frac can be provided")

        if frac:
            if frac > 1 or frac < 0:
                raise ValueError("Fraction must be between 0 and 1")
            n = math.ceil(
                self.get_number_of_documents(self.dataset_id, filters=filters) * frac
            )

        return self.datasets.documents.get_where(
            dataset_id=self.dataset_id,
            filters=filters,
            page_size=n,
            random_state=random_state,
            is_random=True,
            select_fields=select_fields,
        )["documents"]

    def apply(
        self,
        func: Callable,
        axis: int = 0,
    ):
        """
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is either the DataFrame’s index (axis=0) or the DataFrame’s columns (axis=1). By default (result_type=None), the final return type is inferred from the return type of the applied function. Otherwise, it depends on the result_type argument.

        Parameters
        --------------
        func: function
            Function to apply to each document

        axis: int
            Axis along which the function is applied.
            - 9 or 'index': apply function to each column
            - 1 or 'columns': apply function to each row

        """
        if axis == 1:
            raise ValueError("We do not support column-wise operations!")

        def bulk_fn(docs):
            for d in docs:
                func(d)
            return docs

        return self.pull_update_push(self.dataset_id, bulk_fn)

    def all(
        self,
        chunk_size: int = 1000,
        filters: List = [],
        sort: List = [],
        select_fields: List = [],
        include_vector: bool = True,
        show_progress_bar: bool = True,
    ):

        """
        Retrieve all documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched. For more details see documents.get_where.

        Parameters
        ----------
        chunk_size : list
            Number of documents to retrieve per retrieval
        include_vector: bool
            Include vectors in the search results
        sort: list
            Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
        filters: list
            Query for filtering the search results
        select_fields : list
            Fields to include in the search results, empty array/list means all fields.
        """

        return self.get_all_documents(
            dataset_id=self.dataset_id,
            chunk_size=chunk_size,
            filters=filters,
            sort=sort,
            select_fields=select_fields,
            include_vector=include_vector,
            show_progress_bar=show_progress_bar,
        )


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        super().__init__(project=project, api_key=api_key)
