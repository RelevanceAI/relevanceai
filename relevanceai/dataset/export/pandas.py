"""
Pandas like dataset API
"""
import pandas as pd

from relevanceai.package_utils.cache import lru_cache

from relevanceai.package_utils.analytics_funcs import track
from relevanceai.dataset.crud.dataset_read import Read
from relevanceai.package_utils.version_decorators import introduced_in_version
from relevanceai.package_utils.constants import MAX_CACHESIZE


class PandasExport(Read):
    @lru_cache(maxsize=MAX_CACHESIZE)
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
