# -*- coding: utf-8 -*-
"""
Statistics API
"""
import pandas as pd

from typing import Optional, Union

from relevanceai.client.helpers import Credentials
from relevanceai.dataset.series import Series
from relevanceai.utils.decorators.version import added
from relevanceai.utils.decorators.analytics import track
from relevanceai._api import APIClient
from relevanceai._api.endpoints.services.cluster import ClusterClient


class Statistics(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
        **kwargs,
    ):
        self.credentials = credentials
        self.dataset_id = dataset_id
        super().__init__(
            dataset_id=self.dataset_id,
            credentials=credentials,
            **kwargs,
        )

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
            credentials=self.credentials,
            dataset_id=self.dataset_id,
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
            schema = self.datasets.schema(self.dataset_id)
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

    @added(version="1.2.2")
    def corr(
        self,
        X: str,
        Y: str,
        vector_field: str,
        alias: str,
        groupby: Optional[str] = None,
        fontsize: int = 16,
        show_plot: bool = True,
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
        cclient = ClusterClient(self.credentials)
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
                credentials=self.credentials,
                dataset_id=self.dataset_id,
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

        if show_plot:
            fig.tight_layout()
            plt.show()
        else:
            return None

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
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        aggregation_query: Optional[dict] = None,
        sort: list = None,
    ):
        """
        Aggregation/Groupby of a collection using an aggregation query. The aggregation query is a json body that follows the schema of:

        Example
        ---------

        .. code-block::

            {
                "groupby" : [
                    {"name": <alias>, "field": <field in the collection>, "agg": "category"},
                    {"name": <alias>, "field": <another groupby field in the collection>, "agg": "numeric"}
                ],
                "metrics" : [
                    {"name": <alias>, "field": <numeric field in the collection>, "agg": "avg"}
                    {"name": <alias>, "field": <another numeric field in the collection>, "agg": "max"}
                ]
            }
            For example, one can use the following aggregations to group score based on region and player name.
            {
                "groupby" : [
                    {"name": "region", "field": "player_region", "agg": "category"},
                    {"name": "player_name", "field": "name", "agg": "category"}
                ],
                "metrics" : [
                    {"name": "average_score", "field": "final_score", "agg": "avg"},
                    {"name": "max_score", "field": "final_score", "agg": "max"},
                    {'name':'total_score','field':"final_score", 'agg':'sum'},
                    {'name':'average_deaths','field':"final_deaths", 'agg':'avg'},
                    {'name':'highest_deaths','field':"final_deaths", 'agg':'max'},
                ]
            }

        """
        return self.datasets.aggregate(
            dataset_id=self.dataset_id,
            groupby=groupby,
            metrics=metrics,
            sort=sort,
            asc=asc,
            filters=filters,
            page_size=page_size,
            page=page,
            aggregation_query=aggregation_query,
        )

    @track
    def facets(
        self,
        fields: Optional[list] = [],
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
            fields=fields,
            date_interval=date_interval,
            page_size=page_size,
            page=page,
            asc=asc,
        )

    @track
    def health_check(self, **kwargs):
        # This is supposed to provide utilities
        raise NotImplementedError()
        # details = self.datasets.monitor.health(self.dataset_id, **kwargs)
        # return details
