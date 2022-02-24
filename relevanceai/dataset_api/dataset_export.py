"""
Pandas like dataset API
"""
import pandas as pd

from functools import lru_cache

from relevanceai.analytics_funcs import track
from relevanceai.dataset_api.dataset_read import Read
from relevanceai.utils import introduced_in_version


class Export(Read):
    @lru_cache(maxsize=8)
    @introduced_in_version("1.1.5")
    def to_pandas_dataframe(self, **kwargs) -> pd.DataFrame:
        """
        Converts a Relevance AI Dataset to a pandas DataFrame.

        Parameters
        ----------
        kwargs: Optional
            see client.get_all_documents() for extra args

        Example
        -------
        .. code-block::
            from relevanceai import Client

            client = Client()

            relevanceai_dataset = client.Dataset("dataset_id")
            df = relevance_ai.to_pandas_dataframe()
        """
        documents = self.get_all_documents(**kwargs)

        try:
            df = pd.DataFrame(documents)
            df.set_index("_id", inplace=True)
            return df
        except KeyError:
            raise Exception("No documents found")

    def __getattr__(self, attr):
        if hasattr(pd.DataFrame, attr):
            df = self.to_pandas_dataframe(show_progress_bar=True)
            try:
                return getattr(df, attr)
            except SyntaxError:
                raise AttributeError(f"'{attr}' is an invalid attribute")
        raise AttributeError(f"'{attr}' is an invalid attribute")

    @track
    def to_csv(self, filename: str, **kwargs):
        """
        Download a dataset from Relevance AI to a local .csv file

        Parameters
        ----------
        filename: str
            path to downloaded .csv file
        kwargs: Optional
            see client.get_all_documents() for extra args

        Example
        -------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            csv_fname = "path/to/csv/file.csv"
            df.to_csv(csv_fname)
        """
        documents = self.get_all_documents(**kwargs)
        df = pd.DataFrame(documents)
        df.to_csv(filename)

    @track
    def to_dict(self, orient: str = "records"):
        """
        Returns the raw list of dicts from Relevance AI

        Parameters
        ----------
        None

        Returns
        -------
        list of documents in dictionary format

        Example
        -------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            dict = df.to_dict(orient="records")
        """
        if orient == "records":
            return self.get_all_documents()
        else:
            raise NotImplementedError
