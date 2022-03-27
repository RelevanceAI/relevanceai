import pandas as pd

from relevanceai.dataset.read import Read
from relevanceai.utils.decorators.analytics import track


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
