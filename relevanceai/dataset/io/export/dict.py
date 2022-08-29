import pandas as pd

from relevanceai.dataset.read import Read
from relevanceai.utils.decorators.analytics import track


class DictExport(Read):
    @track
    def to_dict(self, orient: str = "records", **kwargs):
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
        documents = self.get_all_documents(**kwargs)

        if orient == "records":
            return documents
        else:
            raise NotImplementedError
            dataframe = pd.DataFrame(documents)
