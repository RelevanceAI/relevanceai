"""
Visualisations for your clustering.
"""
import pandas as pd
from relevanceai.constants.errors import (
    MissingClusterError,
    MissingPackageError,
    SetArgumentError,
)
from tqdm.auto import tqdm
from relevanceai.utils.decorators.analytics import track
from relevanceai.operations.cluster.cluster import ClusterOps
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable


class ClusterVizOps(ClusterOps):
    """
    Cluster Visualisations. May contain additional visualisation
    dependencies.
    """

    def __init__(
        self,
        credentials,
        vector_fields: Optional[List[str]] = None,
        alias: Optional[str] = None,
        dataset_id: Optional[str] = None,
        **kwargs,
    ):
        self.vector_fields = vector_fields
        self.alias = alias  # type: ignore
        self.dataset_id = dataset_id
        super().__init__(
            credentials=credentials,
            vector_fields=vector_fields,
            alias=alias,
            dataset_id=dataset_id,
            **kwargs,
        )

    @track
    def plot_basic_distributions(
        self,
        numeric_field: str,
        top_indices: int = 10,
        dataset_id: Optional[str] = None,
    ):
        """
        Plot the sentence length distributions across each cluster

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()

            cluster_ops = client.ClusterVizOps(
                dataset_id="sample_dataset",
                vector_fields=["sample_vector_"],
                alias="kmeans-5"
            )
            cluster_ops.plot_basic_distributions()

        Parameters
        ------------
        numeric_field: str
            The numeric field to plot
        top_indices: int
            The top indices in the plotting
        dataset_id: Optional[str]
            The dataset ID

        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            raise MissingPackageError(package="seaborn")

        cluster_field = self._get_cluster_field_name()
        docs = self._get_all_documents(
            dataset_id=self.dataset_id if dataset_id is None else dataset_id,
            select_fields=[numeric_field, cluster_field],
        )
        df = pd.json_normalize(docs)
        top_comms = df[cluster_field].value_counts()
        for community in top_comms.index[:top_indices]:
            sample_comm_df = df[df[cluster_field] == community]
            sns.displot(sample_comm_df[numeric_field])
            # Get the average in the score too
            mean = sample_comm_df[numeric_field].mean()
            var = sample_comm_df[numeric_field].var()
            plt.title(
                community + str(f" - average: {round(mean, 2)}, var: {round(var, 2)}")
            )
            plt.show()

    @track
    def plot_distributions(
        self,
        numeric_field: str,
        measure_function: Callable = None,
        top_indices: int = 10,
        dataset_id: str = None,
        asc: bool = True,
        measurement_name: str = "measurement",
    ):
        """
        Plot the distributions across each cluster
        measure_function is run on each cluster and plots

        Example
        ----------
        .. code-block::

            from scipy.stats import skew
            ops.plot_distributions_measure(numeric_field, skew, dataset_id=dataset_id)

        Parameters
        -------------
        numeric_field: str
            The numeric field to plot the distribution by
        measure_function: callable
            What to measure the function
        top_indices: int
            The top indices
        dataset_id: str
            The dataset ID to use
        asc: bool
            If True, the distributions are plotted
        measurement_name: str
            The name of what should be plotted for the graphs

        """
        if measure_function is None:
            return self.plot_basic_distributions(
                numeric_field=numeric_field,
                top_indices=top_indices,
                dataset_id=dataset_id,
            )
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            raise MissingPackageError(package="seaborn")

        cluster_field = self._get_cluster_field_name()

        # use the max and min to make the x axis the same
        if dataset_id is None:
            dataset_id = self.dataset_id
        numeric_field_facet = self.datasets.facets(
            dataset_id=dataset_id, fields=[numeric_field]
        )

        facet_result = numeric_field_facet["results"][numeric_field]

        docs = self._get_all_documents(
            dataset_id=dataset_id if dataset_id is None else dataset_id,
            select_fields=[numeric_field, cluster_field],
        )
        df = pd.json_normalize(docs)
        top_comms = df[cluster_field].value_counts()
        cluster_measurements = {}
        for community in tqdm(top_comms.index):
            sample_comm_df = df[df[cluster_field] == community]
            measure_output = measure_function(
                sample_comm_df[numeric_field].dropna().to_list()
            )
            cluster_measurements[community] = measure_output

        cluster_measurements = {
            k: v
            for k, v in sorted(
                cluster_measurements.items(), key=lambda item: item[1], reverse=asc
            )
        }

        for i, (community, measurement) in enumerate(cluster_measurements.items()):
            if i == top_indices:
                return
            sample_comm_df = df[df[cluster_field] == community]
            g = sns.displot(
                sample_comm_df[numeric_field],
            )
            g.set(xlim=(facet_result["min"], facet_result["max"]))
            plt.title(community + str(f" - {measurement_name}: {measurement}"))

    def plot_most_skewed(
        self,
        numeric_field: str,
        top_indices: int = 10,
        dataset_id: str = None,
        asc: bool = True,
    ):
        """
        Plot the most skewed numeric fields
        """
        from scipy.stats import skew

        return self.plot_distributions(
            numeric_field=numeric_field,
            measure_function=skew,
            top_indices=top_indices,
            dataset_id=dataset_id,
            asc=asc,
        )

    def _check_for_dataset_id(self):
        if not hasattr(self, "dataset_id"):
            raise SetArgumentError("dataset_id")

    def _get_cluster_field_name(self, alias: str = None):
        if alias is None:
            alias = self.alias
        if isinstance(self.vector_fields, list):
            set_cluster_field = f"_cluster_.{'.'.join(self.vector_fields)}.{alias}"
        elif isinstance(self.vector_fields, str):
            set_cluster_field = f"{self.cluster_field}.{self.vector_fields}.{alias}"
        elif self.vector_fields == None:
            raise ValueError("Vector field is not set.")
        else:
            raise ValueError("Can't detect cluster field.")
        return set_cluster_field
