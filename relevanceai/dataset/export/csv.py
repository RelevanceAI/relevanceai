"""
Pandas like dataset API
"""
import pandas as pd

from relevanceai.package_utils.cache import lru_cache

from relevanceai.package_utils.analytics_funcs import track
from relevanceai.dataset.crud.dataset_read import Read
from relevanceai.package_utils.version_decorators import introduced_in_version
from relevanceai.package_utils.constants import MAX_CACHESIZE


class CSVExport(Read):
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
