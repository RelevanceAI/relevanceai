"""
Pandas like dataset API
"""
import re
import math
import warnings
import pandas as pd
import numpy as np

from doc_utils import DocUtils

from typing import Dict, List, Union, Callable, Optional

from relevanceai.dataset_api.groupby import Groupby, Agg
from relevanceai.dataset_api.centroids import Centroids
from relevanceai.dataset_api.helpers import _build_filters

from relevanceai.vector_tools.client import VectorTools
from relevanceai.api.client import BatchAPIClient
from relevanceai.dataset_api.dataset_read import Read


class Export(Read):
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

            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)

            csv_fname = "path/to/csv/file.csv"
            df.to_csv(csv_fname)
        """
        documents = self.get_all_documents(**kwargs)
        df = pd.DataFrame(documents)
        df.to_csv(filename)

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

            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)

            dict = df.to_dict(orient="records")
        """
        if orient == "records":
            return self.get_all_documents()
        else:
            raise NotImplementedError
