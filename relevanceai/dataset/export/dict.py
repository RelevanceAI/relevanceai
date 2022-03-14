"""
Pandas like dataset API
"""
import pandas as pd

from relevanceai.package_utils.cache import lru_cache

from relevanceai.package_utils.analytics_funcs import track
from relevanceai.dataset.crud.dataset_read import Read
from relevanceai.package_utils.version_decorators import introduced_in_version
from relevanceai.package_utils.constants import MAX_CACHESIZE


class DictExport(Read):
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
