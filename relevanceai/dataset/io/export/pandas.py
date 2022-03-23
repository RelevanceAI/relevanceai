import pandas as pd

from relevanceai.utils.cache import lru_cache

from relevanceai.dataset.read import Read
from relevanceai.utils.decorators.version import added
from relevanceai.constants.constants import MAX_CACHESIZE


class PandasExport(Read):
    @added(version="1.1.5")
    @lru_cache(maxsize=MAX_CACHESIZE)
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
