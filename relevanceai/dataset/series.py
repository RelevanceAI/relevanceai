import math
import warnings
from relevanceai.constants.warning import Warning

import pandas as pd
import numpy as np

from typing import Any, Dict, List, Union, Callable, Optional

from tqdm import tqdm

from relevanceai.client.helpers import Credentials
from relevanceai.constants import MAX_CACHESIZE
from relevanceai.utils.cache import lru_cache
from relevanceai.utils.decorators.analytics import track
from relevanceai._api import APIClient
from relevanceai.utils.filters import Filter


class Series(APIClient):
    """
    Dataset Series Object
    -----------------------------

    A wrapper class for being able to vectorize documents over field

    Parameters
    ----------
    project : str
        Project name on RelevanceAI
    api_key : str
        API key for RelevanceAI
    dataset_id : str
        Data type for the output Series. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.
    field : str
        The name of the field with the Dataset.

    Examples
    --------
    Assuming the following code has been executed:

    .. code-block::

        from relevanceai import client
        relevanceai.package_utils.datasets import get_dummy_ecommerce_dataset

        documents = get_dummy_ecommerce_dataset()
        client = Client()

        df = client.Dataset('ecommerce')
        df.create()
        df.insert_documents(documents)

    Retrieve a Series from your dataset

    .. code-block::

        product_images = df['product_image'] # A Series object of every every product image url in dataset

    """

    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
        field: str,
        image_fields: Optional[List[str]] = None,
        audio_fields: Optional[List[str]] = None,
        highlight_fields: Optional[Dict[str, List]] = None,
        text_fields: Optional[List[str]] = None,
    ):
        super().__init__(credentials)
        self.dataset_id = dataset_id
        self.field = field
        self.image_fields = [] if image_fields is None else image_fields
        self.audio_fields = [] if audio_fields is None else audio_fields
        self.highlight_fields = {} if highlight_fields is None else highlight_fields
        self.text_fields = [] if text_fields is None else text_fields

    def _repr_html_(self):
        if "_vector_" in self.field:
            include_vector = True
        else:
            include_vector = False

        documents = self._get_documents(
            dataset_id=self.dataset_id,
            select_fields=[self.field],
            include_vector=include_vector,
        )
        if include_vector:
            warnings.warn(Warning.MISSING_RELEVANCE_NOTEBOOK)
            return pd.json_normalize(documents).set_index("_id")._repr_html_()

        try:
            return self._show_json(documents, return_html=True)
        except Exception as e:
            warnings.warn(Warning.MISSING_RELEVANCE_NOTEBOOK + str(e))
            return pd.json_normalize(documents).set_index("_id")._repr_html_()

    @track
    def list_aliases(self):
        fields = self._format_select_fields()
        fields = [field for field in fields if field != "_cluster_"]
        return pd.DataFrame(fields, columns=["_cluster_"])._repr_html_()

    def _format_select_fields(self):
        fields = [
            field
            for field in self.datasets.schema(self.dataset_id)
            if "_cluster_" in field
        ]
        return fields

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

    @track
    def sample(
        self,
        n: int = 1,
        frac: float = None,
        filters: Optional[list] = None,
        random_state: int = 0,
        include_vector: bool = True,
        output_format="pandas",
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

        Example
        -------
        .. code-block::

            from relevanceai import client

            client = Client()

            df = client.Dataset(dataset_id)
            df.sample(n=3)

        """
        filters = [] if filters is None else filters

        select_fields = [self.field]

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
            return pd.DataFrame.from_records(documents)
        else:
            raise ValueError("Incorrect output format")

    head = sample

    @lru_cache(maxsize=MAX_CACHESIZE)
    @track
    def all(
        self,
        chunksize: int = 1000,
        filters: Optional[List] = None,
        sort: Optional[List] = None,
        include_vector: bool = True,
        show_progress_bar: bool = True,
    ):
        filters = [] if filters is None else filters
        sort = [] if sort is None else sort

        select_fields = [self.field] if isinstance(self.field, str) else self.field
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
    def apply(
        self,
        func: Callable,
        output_field: str,
        filters: list = [],
        axis: int = 0,
        **kwargs,
    ):
        """
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is either the DataFrame’s index (axis=0) or the DataFrame’s columns (axis=1). By default (result_type=None), the final return type is inferred from the return type of the applied function. Otherwise, it depends on the result_type argument.


        .. note::
            We recommend using the bulk_apply functionality
            if you are looking to have faster processing.

        Parameters
        --------------
        func: function
            Function to apply to each document

        axis: int
            Axis along which the function is applied.
            - 9 or 'index': apply function to each column
            - 1 or 'columns': apply function to each row

        output_field: str
            The field from which to output

        Example
        ---------------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            df["sample_1_label"].apply(lambda x: x + 3, output_field="output_field")

        """
        if axis == 1:
            raise ValueError("We do not support column-wise operations!")

        def bulk_fn(documents):
            for d in tqdm(documents):
                try:
                    if self.is_field(self.field, d):
                        self.set_field(
                            output_field, d, func(self.get_field(self.field, d))
                        )
                except Exception as e:
                    continue
            return documents

        return self.pull_update_push_async(
            self.dataset_id,
            bulk_fn,
            select_fields=[self.field],
            filters=filters,
            **kwargs,
        )

    @track
    def bulk_apply(
        self,
        bulk_func: Callable,
        retrieve_chunksize: int = 100,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        show_progress_bar: bool = True,
        use_json_encoder: bool = True,
    ):
        """
        Apply a bulk function along an axis of the DataFrame.

        Parameters
        ------------
        bulk_func: function
            Function to apply to a bunch of documents at a time
        retrieve_chunksize: int
            The number of documents that are received from the original collection with each loop iteration.
        max_workers: int
            The number of processors you want to parallelize with
        max_error: int
            How many failed uploads before the function breaks
        json_encoder : bool
            Whether to automatically convert documents to json encodable format
        axis: int
            Axis along which the function is applied.
            - 9 or 'index': apply function to each column
            - 1 or 'columns': apply function to each row

        Example
        ---------
        .. code-block::

            from relevanceai import Client
            client = Client()

            df = client.Dataset("sample_dataset_id")

            def update_documents(documents):
                for d in documents:
                    d["value"] = 10
                return documents

            df.bulk_apply(update_documents)
        """
        filters = [] if filters is None else filters
        select_fields = [self.field] if select_fields is None else select_fields

        return self.pull_update_push_async(
            self.dataset_id,
            bulk_func,
            retrieve_chunk_size=retrieve_chunksize,
            filters=filters,
            select_fields=select_fields,
            show_progress_bar=show_progress_bar,
            use_json_encoder=use_json_encoder,
        )

    @track
    def numpy(self) -> np.ndarray:
        """
        Iterates over all documents in dataset and returns all numeric values in a numpy array.

        Parameters
        ---------
        None

        Returns
        -------
        vectors: np.ndarray
            an array/matrix of all numeric values selected

        Example
        ---------------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            field = "sample_field"
            arr = df[field].numpy()
        """
        documents = self._get_all_documents(self.dataset_id, select_fields=[self.field])
        vectors = self.get_field_across_documents(self.field, documents)
        vectors = np.array(vectors)
        return vectors

    @track
    def value_counts(
        self,
        normalize: bool = False,
        ascending: bool = False,
        sort: bool = False,
        bins: Optional[int] = None,
    ):
        """
        Return a Series containing counts of unique values (or values with in a range if bins is set).

        Parameters
        ----------
        normalize : bool, default False
            If True then the object returned will contain the relative frequencies of the unique values.
        ascending : bool, default False
            Sort in ascending order.
        bins : int, optional
            Groups categories into 'bins'. These bins are good for representing groups within continuous series

        Returns
        ----------
        Series

        Example
        ---------------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            field = "sample_field"
            value_counts_df = df[field].value_counts()
        """
        schema = self.datasets.schema(self.dataset_id)
        dtype = schema[self.field]

        if dtype == "numeric":
            agg_type = dtype
        else:
            agg_type = "category"

        groupby_query = [{"name": self.field, "field": self.field, "agg": agg_type}]
        aggregation = self.datasets.aggregate(
            dataset_id=self.dataset_id,
            groupby=groupby_query,
            page_size=10000,
            asc=ascending,
        )["results"]

        total = self.get_number_of_documents(dataset_id=self.dataset_id)
        aggregation = pd.DataFrame(aggregation)

        if normalize:
            aggregation["frequency"] /= total

        if bins is not None:
            vals = []
            for agg in [[agg[0]] * int(agg[1]) for agg in aggregation.values]:
                vals += agg

            vals = pd.cut(vals, bins)

            categories = [
                "({}, {}]".format(interval.left, interval.right) for interval in vals
            ]
            unique_categories = list(set(categories))

            if sort:
                categories = sorted(
                    categories, key=lambda x: float(x.split(",")[0][1:])
                )

            aggregation = pd.DataFrame(
                [categories.count(cat) for cat in unique_categories],
                index=unique_categories,
            )
            aggregation.columns = ["Frequency"]

        return aggregation

    def __getitem__(self, loc: Union[int, str]):
        """
        Indexs a value with a series, usually to get a specific sample from a column in your dataset

        Parameters
        ----------
        loc : int or str, preferably a str
            if int, this operates exactly as indexing a regular python list
            if str, this will be a string corresponding to the _id of the document

        Returns
        ----------
        A single document

        Example
        ---------------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            field = "sample_field"
            id = "sample_id"
            index = 56

            document = df[field][id]
            document = df[field][index]
        """
        if isinstance(loc, int):
            warnings.warn(Warning.INDEX_STRING)
            return self.get_documents(loc + 1, select_fields=[self.field])[loc][
                self.field
            ]
        elif isinstance(loc, str):
            return self.datasets.documents.get(self.dataset_id, loc)[self.field]
        raise TypeError("Incorrect data type! Must be a string or an integer")

    @lru_cache(maxsize=MAX_CACHESIZE)
    def _get_pandas_series(self):
        documents = self._get_all_documents(
            dataset_id=self.dataset_id,
            select_fields=[self.field],
            include_vector=False,
            show_progress_bar=True,
        )

        try:
            df = pd.DataFrame(documents)
            df.set_index("_id", inplace=True)
            return df.squeeze()
        except KeyError:
            raise Exception("No documents found")

    def __getattr__(self, attr):
        if hasattr(pd.Series, attr):
            series = self._get_pandas_series()
            try:
                return getattr(series, attr)
            except SyntaxError:
                raise AttributeError(f"'{attr}' is an invalid attribute")
        raise AttributeError(f"'{attr}' is an invalid attribute")

    def __add__(self, other):
        schema = self.datasets.schema(self.dataset_id)
        if schema[self.field] != "numeric":
            raise ValueError(f"{self.field} is not a numeric type")
        if schema[other.field] != "numeric":
            raise ValueError(f"{other.field} is not a numeric type")
        if other.field not in schema:
            raise ValueError(f"{other.field} must be an attribute of {self.dataset_id}")

        return self._get_pandas_series() + other._get_pandas_series()

    def __eq__(self, other: Union[str, float, int, bool, None]):
        if self.field == "_id":
            filter = Filter(
                field=self.field,
                dataset_id=self.dataset_id,
                filter_type="ids",
                condition="==",
                condition_value=other,
                credentials=self.credentials,
            )
        else:
            filter = Filter(
                field=self.field,
                dataset_id=self.dataset_id,
                condition="==",
                condition_value=other,
                credentials=self.credentials,
            )
        return filter.get()

    def __ne__(self, other: Union[str, float, int, bool, None]):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            condition="!=",
            condition_value=other,
            credentials=self.credentials,
        )
        return filter.get()

    def __lt__(self, other: Union[str, float, int, bool, None]):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            condition="<",
            condition_value=other,
            credentials=self.credentials,
        )
        return filter.get()

    def __gt__(self, other: Union[str, float, int, bool, None]):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            condition=">",
            condition_value=other,
            credentials=self.credentials,
        )
        return filter.get()

    def __le__(self, other: Union[str, float, int, bool, None]):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            condition="<=",
            condition_value=other,
            credentials=self.credentials,
        )
        return filter.get()

    def __ge__(self, other: Union[str, float, int, bool, None]):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            condition=">=",
            condition_value=other,
            credentials=self.credentials,
        )
        return filter.get()

    def contains(self, other: Any):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            filter_type="contains",
            condition="==",
            condition_value=other,
            credentials=self.credentials,
        )
        return filter.get()

    def exists(self):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            filter_type="exists",
            condition="==",
            condition_value=" ",
            credentials=self.credentials,
        )
        return filter.get()

    def not_exists(self):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            filter_type="exists",
            condition="!=",
            condition_value=" ",
            credentials=self.credentials,
        )
        return filter.get()

    def date(self, other: Any):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            filter_type="date",
            condition="==",
            condition_value=other,
            credentials=self.credentials,
        )
        return filter.get()

    def categories(self, other: List[Any]):
        filter = Filter(
            field=self.field,
            dataset_id=self.dataset_id,
            filter_type="categories",
            condition="==",
            condition_value=other,
            credentials=self.credentials,
        )

    def filter(self, **kwargs):
        return [kwargs]

    def set_dtype(self, dtype):
        metadata = self.datasets.metadata(self.dataset_id)["results"]
        metadata[self.field] = dtype
        self.datasets.post_metadata(
            self.dataset_id,
            metadata=metadata,
        )
