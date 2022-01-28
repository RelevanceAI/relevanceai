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
from relevanceai.dataset_api.dataset_series import Series

class Stats(Read):
    def value_counts(self, field: str):
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        field: str
            dataset field to which to do value counts on

        Returns
        -------
        Series

        Example
        -----------------
        .. code-block::

            from relevanceai import Client
            client = Client()
            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)
            field = "sample_field"
            value_counts_df = df.value_counts(field)

        """
        return Series(self.project, self.api_key, self.dataset_id, field).value_counts()

    def describe(self) -> dict:
        """
        Descriptive statistics include those that summarize the central tendency
        dispersion and shape of a dataset's distribution, excluding NaN values.


        Example
        -----------------
        .. code-block::

            from relevanceai import Client
            client = Client()
            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)
            field = "sample_field"
            df.describe()

        """
        return self.datasets.facets(self.dataset_id)

    @property
    def health(self) -> dict:
        """
        Gives you a summary of the health of your vectors, e.g. how many documents with vectors are missing, how many documents with zero vectors

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")
            df.health

        """
        return self.datasets.monitor.health(self.dataset_id)

