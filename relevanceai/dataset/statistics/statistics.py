# -*- coding: utf-8 -*-
"""
Pandas like dataset API
"""
import pandas as pd

from typing import Dict, List, Optional, Union

from relevanceai.package_utils.analytics_funcs import track
from relevanceai.api.endpoints.services.cluster import ClusterClient
from relevanceai.dataset.crud.dataset_read import Read
from relevanceai.dataset.crud.dataset_series import Series
from relevanceai.package_utils.version_decorators import introduced_in_version


class Statistics(Read):
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

    @introduced_in_version("1.2.2")
    def corr(
        self,
        X: str,
        Y: str,
        vector_field: str,
        alias: str,
        groupby: Optional[str] = None,
        fontsize: int = 16,
    ):
        """
        Returns the Pearson correlation between two fields.

        Parameters
        ----------
        X: str
            A dataset field

        Y: str
            The other dataset field

        vector_field: str
            The vector field over which the clustering has been performed

        alias: str
            The alias of the clustering

        groupby: Optional[str]
            A field to group the correlations over

        fontsize: int
            The font size of the values in the image
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # todo: how to cover cases when fields are in schema but not "calculable" fields like clusters and deployables
        cclient = ClusterClient(self.project, self.api_key, self.firebase_uid)
        groupby_agg = (
            []
            if groupby is None
            else [{"name": groupby, "field": groupby, "agg": "correlation"}]
        )
        res = cclient.aggregate(
            dataset_id=self.dataset_id,
            vector_fields=[vector_field],
            metrics=[{"name": "correlation", "fields": [X, Y], "agg": "correlation"}],
            groupby=groupby_agg,
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

        # Use a pandas DataFrame for easy indexing
        dataframe = pd.DataFrame(data=[], columns=clusters, index=categories)

        for cluster, values in res.items():
            for value in values:
                correlation_value = value["correlation"][X][Y]
                category = value.get(groupby, "cluster")
                dataframe.at[category, cluster] = correlation_value

        # Only needed pandas DataFrame for indexing, now convert to numpy
        # ndarray for convenience.
        data = dataframe.to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(2 * len(clusters), 2 * len(categories)))
        cmap = plt.get_cmap("coolwarm_r")
        cmap.set_bad("k")  # if np.nan, set imshow cell to black
        im = ax.imshow(data, norm=mpl.colors.Normalize(vmin=-1, vmax=1), cmap=cmap)

        ax.set_xticks(
            range(data.shape[1]),
            labels=clusters,
            rotation=-30,
            rotation_mode="anchor",
            ha="right",
            fontsize=fontsize + 1,
        )
        ax.set_yticks(range(data.shape[0]), labels=categories, fontsize=fontsize + 1)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        ax.spines[:].set_visible(False)
        ax.set_xticks([n - 0.5 for n in range(data.shape[1] + 1)], minor=True)
        ax.set_yticks([n - 0.5 for n in range(data.shape[0] + 1)], minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        for column, _ in enumerate(clusters):
            for row, _ in enumerate(categories):
                # Ensure that the negative sign doesn't offset the text by
                # offseting positive values.
                if data[row][column] < 0:
                    text = f"{data[row][column]:.2f}"
                else:
                    text = f" {data[row][column]:.2f}"
                im.axes.text(
                    column,
                    row,
                    text,
                    dict(horizontalalignment="center", verticalalignment="center"),
                    fontsize=fontsize,
                )

        fig.tight_layout()
        plt.show()

    def health(self, output_format="dataframe") -> Union[pd.DataFrame, dict]:
        """
        Gives you a summary of the health of your vectors, e.g. how many documents with vectors are missing, how many documents with zero vectors

        Parameters
        ----------

        output_format: str
            The format of the output. Must either be "dataframe" or "json".

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id")
            df.health

        """
        results = self.datasets.monitor.health(self.dataset_id)
        if output_format == "dataframe":
            return pd.DataFrame(results).T
        elif output_format == "json":
            return results
        else:
            raise ValueError('\'output_format\' must either be "dataframe" or "json"')

    @track
    def aggregate(
        self,
        groupby: Optional[list] = None,
        metrics: Optional[list] = None,
        filters: Optional[list] = None,
        # sort: list = [],
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
        alias: str = "default",
    ):
        return self.services.aggregate.aggregate(
            dataset_id=self.dataset_id,
            groupby=[] if groupby is None else groupby,
            metrics=[] if metrics is None else metrics,
            filters=[] if filters is None else filters,
            page_size=page_size,
            page=page,
            asc=asc,
            flatten=flatten,
            alias=alias,
            # sort=sort
        )

    def facets(
        self,
        fields: Optional[list] = None,
        date_interval: str = "monthly",
        page_size: int = 5,
        page: int = 1,
        asc: bool = False,
    ):
        """
        Get a summary of fields - such as most common, their min/max, etc.

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            from relevanceai.datasets import mock_documents
            documents = mock_documents(100)
            ds = client.Dataset("mock_documents")
            ds.upsert_documents(documents)
            ds.facets(["sample_1_value"])
        """
        return self.datasets.facets(
            dataset_id=self.dataset_id,
            fields=[] if fields is None else fields,
            date_interval=date_interval,
            page_size=page_size,
            page=page,
            asc=asc,
        )

    def __call__(
        self,
        dataset_id: str,
        image_fields: Optional[List] = None,
        text_fields: Optional[List] = None,
        audio_fields: Optional[List] = None,
        highlight_fields: Optional[Dict[str, List]] = None,
        output_format: str = "pandas",
    ):
        self.dataset_id = dataset_id
        self.image_fields = [] if image_fields is None else image_fields
        self.text_fields = [] if text_fields is None else text_fields
        self.audio_fields = [] if audio_fields is None else audio_fields
        self.highlight_fields = {} if highlight_fields is None else highlight_fields
        self.output_format = output_format
        return self
