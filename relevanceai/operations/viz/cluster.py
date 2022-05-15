"""
Visualisations for your clustering.
"""
import pandas as pd
import numpy as np
from relevanceai.constants.errors import (
    MissingClusterError,
    MissingPackageError,
    SetArgumentError,
)
from tqdm.auto import tqdm
from relevanceai.utils.decorators.analytics import track
from relevanceai.utils import largest_indices
from relevanceai.operations.cluster.cluster import ClusterOps
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable
from relevanceai.operations.cluster.utils import ClusterUtils


class ClusterVizOps(ClusterOps, ClusterUtils):
    """
    Cluster Visualisations. May contain additional visualisation
    dependencies.
    """

    def __init__(
        self,
        credentials,
        dataset_id: str,
        vector_fields: List[str],
        alias: Optional[str] = None,
        **kwargs,
    ):
        self.vector_fields = vector_fields  # type: ignore
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

    def centroid_heatmap(
        self,
        metric: str = "cosine",
        vmin: float = 0,
        vmax: float = 1,
        print_n: int = 8,
        round_print_float: int = 2,
    ):
        """
        Heatmap visualisation of the closest clusters.
        Prints the ones ranked from top to bottom in terms of largest cosine similarity.
        """
        closest_clusters = self.closest(include_vector=True, verbose=False)
        import seaborn as sns
        from relevanceai.utils import DocUtils
        from relevanceai.utils.distances.cosine_similarity import cosine_similarity
        from sklearn.metrics import pairwise_distances

        shape = (len(closest_clusters["results"]), len(closest_clusters["results"]))
        all_vectors = []
        if self.vector_fields is not None:
            if len(self.vector_fields) == 1:  # type: ignore
                vector_field = self.vector_fields[0]
            else:
                raise NotImplementedError
        else:
            raise ValueError("Please set vector fields in the initialization.")

        for c, values in closest_clusters["results"].items():
            all_vectors.append(values["results"][0][vector_field])
        dist_out = 1 - pairwise_distances(all_vectors, metric="cosine")

        dist_df = pd.DataFrame(dist_out)
        heatmap_values = list(closest_clusters["results"].keys())
        dist_df.columns = heatmap_values
        dist_df.index = heatmap_values

        # ignore the initial set as they are the same indices
        ignore_initial = dist_out.shape[0]
        left, top = largest_indices(np.tril(dist_out), ignore_initial + print_n)

        left = left[ignore_initial:]
        top = top[ignore_initial:]

        print("Your closest centroids are:")
        for l, t in zip(left, top):
            print(
                f"{round(dist_out[l][t], round_print_float)} {heatmap_values[l]}, {heatmap_values[t]}"
            )

        return sns.heatmap(
            data=dist_df,
            vmin=vmin,
            vmax=vmax,
        ).set(title=f"{metric} plot")

    def show_closest(
        self,
        cluster_ids: Optional[List] = None,
        text_fields: Optional[List] = None,
        image_fields: Optional[List] = None,
    ):
        """
        Show the clusters with the closest.
        """
        from relevanceai import show_json

        if text_fields is None:
            text_fields = []
        if image_fields is None:
            image_fields = []
        new_closest = self.closest(cluster_ids=cluster_ids)
        closest_reformat = []
        for k, v in new_closest["results"].items():
            for r in v["results"]:
                closest_reformat.append({"cluster_id": k, **r})
        if cluster_ids is not None:
            text_fields += ["cluster_id"]
        return show_json(
            closest_reformat,
            text_fields=text_fields,
            image_fields=image_fields,
        )
