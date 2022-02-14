# -*- coding: utf-8 -*-
"""
Pandas like dataset API
"""
import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Dict
from relevanceai.analytics_funcs import track
from relevanceai.api.endpoints.services.cluster import ClusterClient
from relevanceai.dataset_api.dataset_read import Read
from relevanceai.dataset_api.dataset_series import Series


class Stats(Read):
    @track
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
            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            field = "sample_field"
            value_counts_df = df.value_counts(field)

        """
        return Series(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            firebase_uid=self.firebase_uid,
            field=field,
        ).value_counts()

    @track
    def describe(self, return_type="pandas") -> dict:
        """
        Descriptive statistics include those that summarize the central tendency
        dispersion and shape of a dataset's distribution, excluding NaN values.


        Example
        -----------------
        .. code-block::

            from relevanceai import Client
            client = Client()
            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            field = "sample_field"
            df.describe() # returns pandas dataframe of stats
            df.describe(return_type='dict') # return raw json stats

        """
        facets = self.datasets.facets(self.dataset_id)
        if return_type == "pandas":
            schema = self.schema
            dataframe = {
                col: facets["results"][col]
                for col in schema
                if col in facets["results"] and isinstance(facets["results"][col], dict)
            }
            dataframe = pd.DataFrame(dataframe)
            return dataframe
        elif return_type == "dict":
            return facets
        else:
            raise ValueError("invalid return_type, should be `dict` or `pandas`")

    @track
    def corr(self, X: str, Y: str, vector_field: str, alias: str, groupby: str = None):
        """
        Returns the Pearson correlation between two fields.

        Parameters
        ----------
        X: str
            A dataset field

        Y: str
            The other dataset field over which

        Returns
        -------
        """
        # todo: how to cover cases when fields are in schema but not "calculable" fields like clusters and deployables
        # TODO: add groupby
        cclient = ClusterClient(self.project, self.api_key, self.firebase_uid)
        res = cclient.aggregate(
            dataset_id=self.dataset_id,
            vector_fields=[vector_field],
            metrics=[{"name": "correlation", "fields": [X, Y], "agg": "correlation"}],
            alias=alias,
        )["results"]

        clusters = sorted(res.keys())

        if groupby is None:
            categories = ["cluster"]
        else:
            series = Series(
                project=self.project,
                api_key=self.api_key,
                dataset_id=self.dataset_id,
                firebase_uid=self.firebase_uid,
                field=groupby,
            ).all(show_progress_bar=False)

            categories = sorted(
                pd.Series(map(lambda _: _[groupby], series)).drop_duplicates()
            )

        data = pd.DataFrame(data=[], columns=clusters, index=categories)

        for cluster, values in res.items():
            for value in values:
                correlation_value = value["correlation"][X][Y]
                category = value.get(groupby, "cluster")
                data.at[category, cluster] = correlation_value

        ax = plt.gca()
        im = ax.imshow(data)

        # cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(ch)

    @property
    def health(self) -> dict:
        """
        Gives you a summary of the health of your vectors, e.g. how many documents with vectors are missing, how many documents with zero vectors

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id")
            df.health

        """
        return self.datasets.monitor.health(self.dataset_id)

    def __call__(
        self,
        dataset_id: str,
        image_fields: List = [],
        text_fields: List = [],
        audio_fields: List = [],
        highlight_fields: Dict[str, List] = {},
        output_format: str = "pandas",
    ):
        self.dataset_id = dataset_id
        self.image_fields = image_fields
        self.text_fields = text_fields
        self.audio_fields = audio_fields
        self.highlight_fields = highlight_fields
        self.output_format = output_format
        return self
