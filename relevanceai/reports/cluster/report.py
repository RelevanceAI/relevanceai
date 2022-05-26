"""
Automated Cluster Reporting

.. warning::
    This is a beta feature.

.. note::
    **Introduced in v1.0.0.**

You can run cluster reporting as a standalone module with the option
to store it in Relevance AI.

.. code-block::

    import requests
    docs = requests.get("https://raw.githubusercontent.com/fanzeyi/pokemon.json/master/pokedex.json").json()
    for d in docs:
        b = d['base']
        d.update(b)
        d['base_vector_'] = [b["Attack"], b["Defense"], b["HP"], b["Sp. Attack"], b["Sp. Defense"], b["Speed"]]

    import pandas as pd
    import numpy as np
    df = pd.DataFrame(docs)
    X = np.array(df['base_vector_'].tolist())


    from relevanceai.workflows.cluster.reports.cluster_report import ClusterReport
    from sklearn.cluster import KMeans

    N_CLUSTERS = 2
    kmeans = KMeans(n_clusters=N_CLUSTERS)
    cluster_labels = kmeans.fit_predict(X)

    report = ClusterReport(
        X=X,
        cluster_labels=cluster_labels,
        num_clusters=N_CLUSTERS,
        model=kmeans
    )

    # JSON output
    report.internal_report

    # Prettyprinted report of overall statistics
    report.internal_overall_report

    # Storing your cluster report
    from relevanceai import Client
    client = Client()
    response = client.store_cluster_report(
        report_name="kmeans",
        report=report.internal_report
    )

    # Listing all cluster reports
    client.list_cluster_reports()

    # Deleting cluster report
    client.delete_cluster_report(response['_id])


You can also insert your own centroid vectors if you want them to be represented.
For example - you may want to measure off medoids (points in your dataset) instead of centroids
(as opposed to points outside of your dataset).

In the example below, we show how you calculate centroids or medoids for HDBSCAN

.. code-block::

    from relevanceai.workflows.cluster.reports.cluster_report import ClusterReport

    import hdbscan
    clusterer = hdbscan.HDBSCAN()

    cluster_labels = clusterer.fit_predict(X)
    centroids = ClusterReport.calculate_centroids(X, cluster_labels)
    # or if you want medoids
    # medoids = ClusterReport.calculate_medoids(X, cluster_labels)

    report = ClusterReport(
        X=X,
        cluster_labels=cluster_labels,
        centroid_vectors=centroids
    )
    report.internal_overall_report

.. code-block::

    from relevanceai.workflows.cluster.reports.cluster_report import ClusterReport

    import hdbscan
    clusterer = hdbscan.HDBSCAN()

    cluster_labels = clusterer.fit_predict(X)
    medoids = ClusterReport.calculate_medoids(X, cluster_labels)

    report = ClusterReport(
        X=X,
        cluster_labels=cluster_labels,
        centroid_vectors=centroids
    )
    report.internal_overall_report

"""

import warnings
import pandas as pd
import numpy as np
import functools

from typing import Union, List, Dict, Any, Optional

from relevanceai.utils import DocUtils
from relevanceai.utils.decorators.analytics import track_event_usage
from relevanceai.utils.integration_checks import (
    is_hdbscan_available,
    is_sklearn_available,
)

from relevanceai.reports.cluster.grading import get_silhouette_grade

if is_sklearn_available():
    from sklearn.metrics import (
        davies_bouldin_score,
        calinski_harabasz_score,
        silhouette_samples,
    )
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.cluster import MiniBatchKMeans, KMeans
    from sklearn.tree import _tree, DecisionTreeClassifier
    from sklearn.neighbors import NearestNeighbors

from relevanceai.constants.warning import Warning
from relevanceai.constants.warning import Warning


DECIMAL_PLACES = 3


class ClusterReport(DocUtils):
    """
    Receive an automated cluster reprot

    .. warning::
        This is a beta feature.

    .. note::
        **Introduced in v1.0.0.**


    Parameters
    -------------

    X: np.ndarray
        The original data
    cluster_labels: List[str]
        A list of cluster labels
    model
        The model to analyze. Currently only used
    num_clusters: Optional[int]
        The number of clusters. This is required if we can't actually tell how many clusters there are
    outlier_label: Optional[str, int]
        The label if it is an outlier
    centroids: Union[list, np.ndarray]
        The centroid vectors. If supplied, will use these. Otherwise, will try to infer them
        from the model.
    """

    @track_event_usage("cluster_report")
    def __init__(
        self,
        X: Union[list, np.ndarray],
        cluster_labels: Union[List[Union[str, float]], np.ndarray],
        model: KMeans = None,
        num_clusters: int = None,
        outlier_label: Union[str, int] = -1,
        centroids: Union[list, np.ndarray] = None,
        verbose: bool = False,
        include_typecheck: bool = True,
    ):

        if include_typecheck:
            self._typecheck_model(model)

        if isinstance(X, list):
            self.X = np.array(X)
        else:
            self.X = X
        if isinstance(cluster_labels, list):
            self.cluster_labels = np.array(cluster_labels)
        else:
            self.cluster_labels = cluster_labels
        self.num_clusters = (
            len(set(cluster_labels)) if num_clusters is None else num_clusters
        )
        self.model = model
        self.outlier_label = outlier_label
        if include_typecheck:
            self._typecheck_centroid_vectors(centroids)
        self._centroids = centroids
        self.verbose = verbose

    def _typecheck_centroid_vectors(
        self, centroid_vectors: Optional[Union[list, Dict, np.ndarray]] = None
    ):
        if isinstance(centroid_vectors, (list, np.ndarray)):
            warnings.warn(Warning.CENTROID_VECTORS)

    def _typecheck_model(self, model, verbose: bool = False):
        # Warns users that their model may not be supported
        # this may not be necessary and is removable in future
        if is_hdbscan_available():
            if "hdbscan" in str(type(model)):
                return
        if is_sklearn_available():
            if "sklearn" in str(type(model)):
                return
        if verbose:
            warnings.warn(Warning.MODEL_NOT_SUPPORTED)

    @staticmethod
    def summary_statistics(array: np.ndarray, axis=0):
        """
        Basic summary statistics
        """
        if axis == 2:
            return {
                "sum": np.around(array.sum(), DECIMAL_PLACES),
                "mean": np.around(array.mean(), DECIMAL_PLACES),
                "std": np.around(array.std(), DECIMAL_PLACES),
                "variance": np.around(array.var(), DECIMAL_PLACES),
                "min": np.around(array.min(), DECIMAL_PLACES),
                "max": np.around(array.max(), DECIMAL_PLACES),
                "12_5%": np.around(np.percentile(array, 12.5), DECIMAL_PLACES),
                "25%": np.around(np.percentile(array, 25), DECIMAL_PLACES),
                "37_5%": np.around(np.percentile(array, 37.5), DECIMAL_PLACES),
                "50%": np.around(np.percentile(array, 50), DECIMAL_PLACES),
                "62_5%": np.around(np.percentile(array, 62.5), DECIMAL_PLACES),
                "75%": np.around(np.percentile(array, 75), DECIMAL_PLACES),
                "87_5%": np.around(np.percentile(array, 87.5), DECIMAL_PLACES),
            }
        else:
            return {
                "sum": np.around(array.sum(axis=axis), DECIMAL_PLACES),
                "mean": np.around(array.mean(axis=axis), DECIMAL_PLACES),
                "std": np.around(array.std(axis=axis), DECIMAL_PLACES),
                "variance": np.around(array.var(axis=axis), DECIMAL_PLACES),
                "min": np.around(array.min(axis=axis), DECIMAL_PLACES),
                "max": np.around(array.max(axis=axis), DECIMAL_PLACES),
                "12_5%": np.around(
                    np.percentile(array, 12.5, axis=axis), DECIMAL_PLACES
                ),
                "25%": np.around(np.percentile(array, 25, axis=axis), DECIMAL_PLACES),
                "37_5%": np.around(
                    np.percentile(array, 37.5, axis=axis), DECIMAL_PLACES
                ),
                "50%": np.around(np.percentile(array, 50, axis=axis), DECIMAL_PLACES),
                "62_5%": np.around(
                    np.percentile(array, 62.5, axis=axis), DECIMAL_PLACES
                ),
                "75%": np.around(np.percentile(array, 75, axis=axis), DECIMAL_PLACES),
                "87_5%": np.around(
                    np.percentile(array, 87.5, axis=axis), DECIMAL_PLACES
                ),
            }

    def get_distance_from_centroid(self, cluster_data, center_vector):
        distances_from_centroid = pairwise_distances([center_vector], cluster_data)
        return ClusterReport.summary_statistics(distances_from_centroid, axis=2)

    def get_distance_from_centroid_to_another(self, other_cluster_data, center_vector):
        """Store the distances from a centroid to another."""
        distances_from_centroid_to_another = pairwise_distances(
            [center_vector], other_cluster_data
        )
        return ClusterReport.summary_statistics(
            distances_from_centroid_to_another, axis=2
        )

    def get_distance_from_grand_centroid(self, grand_centroid, specific_cluster_data):
        distances_from_grand_centroid = pairwise_distances(
            [grand_centroid], specific_cluster_data
        )
        return ClusterReport.summary_statistics(distances_from_grand_centroid, axis=2)

    def get_distance_from_grand_centroid_to_point_in_another_cluster(
        self, grand_centroid, other_cluster_data
    ):
        distances_from_grand_centroid_to_another = pairwise_distances(
            [grand_centroid], other_cluster_data
        )
        return ClusterReport.summary_statistics(
            distances_from_grand_centroid_to_another, axis=2
        )

    @staticmethod
    def get_z_score(value, mean, std):
        return (value - mean) / std

    @functools.lru_cache(maxsize=128)
    def get_centers(self, output_format="array"):
        if hasattr(self.model, "cluster_centers_"):
            return np.around(self.model.cluster_centers_, DECIMAL_PLACES)
        elif hasattr(self.model, "get_centers"):
            return self.model.get_centers()
        elif self._centroids is not None:
            if output_format == "array":
                if isinstance(self._centroids, dict):
                    return np.around(
                        np.array(list(self._centroids.values())), DECIMAL_PLACES
                    )
                else:
                    return self._centroids
            else:
                return self._centroids
        else:
            if self.verbose:
                print(
                    "No centroids detected. We recommend including centroids to get all stats."
                )
            return

    @property  # type: ignore
    @functools.lru_cache(maxsize=128)
    def internal_report(self):
        """
        Provide the standard clustering report.
        """
        self.X_silhouette_scores = silhouette_samples(
            self.X, self.cluster_labels, metric="euclidean"
        )
        graded_score = self.X_silhouette_scores.mean()
        grade = get_silhouette_grade(graded_score)

        self._internal_report = {
            "grade": grade,
            "overall": {
                "summary": ClusterReport.summary_statistics(self.X),
                "davies_bouldin_score": round(
                    davies_bouldin_score(self.X, self.cluster_labels),
                    DECIMAL_PLACES,
                ),
                "calinski_harabasz_score": round(
                    calinski_harabasz_score(self.X, self.cluster_labels),
                    DECIMAL_PLACES,
                ),
                "silhouette_score": ClusterReport.summary_statistics(
                    self.X_silhouette_scores
                ),
            },
            "each": [
                # {
                #     "cluster_id": "cluster-1",
                #     "summary": {},
                #     "centers": {},
                #     "silhouette_score": {}
                # }
            ]
            # {
            #     "summary": {},
            #     "centers": {},
            #     "silhouette_score": {},
            # },
        }
        self._store_basic_centroid_stats(self._internal_report["overall"])

        labels, counts = np.unique(self.cluster_labels, return_counts=True)
        # if self.verbose:
        #     print("Detected the cluster labels:")
        #     print(labels)

        cluster_report = {"frequency": {"total": 0, "each": {}}}

        for i, cluster_label in enumerate(labels):
            cluster_label_doc = {
                "cluster_id": str(cluster_label),
            }
            cluster_bool = self.cluster_labels == cluster_label

            specific_cluster_data = self.X[cluster_bool]
            other_cluster_data = self.X[~cluster_bool]

            cluster_frequency = counts[i]
            cluster_report["frequency"]["total"] += cluster_frequency
            cluster_report["frequency"]["each"][cluster_label] = cluster_frequency

            cluster_label_doc["summary"] = ClusterReport.summary_statistics(self.X)

            # If each value of the vector is important
            center_stats = {"by_features": {}}

            center_stats["by_features"]["summary"] = ClusterReport.summary_statistics(
                specific_cluster_data
            )

            if self.has_centers():

                grand_centroid = np.around(
                    self.X[cluster_bool].mean(axis=0), DECIMAL_PLACES
                )

                centroid_vector = self._get_centroid_vector(
                    i, cluster_label, grand_centroid
                )

                squared_errors = np.square(
                    np.subtract(
                        [centroid_vector] * len(specific_cluster_data),
                        specific_cluster_data,
                    )
                )

                self._internal_report["overall"]["grand_centroids"].append(
                    grand_centroid
                )

                center_stats[
                    "distance_from_centroid"
                ] = self.get_distance_from_centroid(
                    specific_cluster_data, centroid_vector
                )

                center_stats[
                    "distance_from_centroid_to_point_in_another_cluster"
                ] = self.get_distance_from_centroid_to_another(
                    other_cluster_data, centroid_vector
                )

                center_stats["distances_from_grand_centroid"] = pairwise_distances(
                    [grand_centroid], specific_cluster_data
                )

                center_stats[
                    "distance_from_grand_centroid"
                ] = self.get_distance_from_grand_centroid(
                    grand_centroid, specific_cluster_data
                )

                center_stats[
                    "distance_from_grand_centroid_to_point_in_another_cluster"
                ] = self.get_distance_from_grand_centroid_to_point_in_another_cluster(
                    grand_centroid, other_cluster_data
                )

                center_stats["by_features"][
                    "overall_z_score"
                ] = ClusterReport.get_z_score(
                    centroid_vector,
                    self._internal_report["overall"]["summary"]["mean"],
                    self._internal_report["overall"]["summary"]["std"],
                )

                center_stats["by_features"]["z_score"] = ClusterReport.get_z_score(
                    centroid_vector,
                    cluster_label_doc["summary"]["mean"],
                    cluster_label_doc["summary"]["std"],
                )

                center_stats["by_features"][
                    "overall_z_score_grand_centroid"
                ] = ClusterReport.get_z_score(
                    grand_centroid,
                    self._internal_report["overall"]["summary"]["mean"],
                    self._internal_report["overall"]["summary"]["std"],
                )

                center_stats[
                    "overall_z_score_grand_centroid"
                ] = ClusterReport.summary_statistics(
                    center_stats["by_features"]["overall_z_score_grand_centroid"]
                )

                center_stats["by_features"]["z_score_grand_centroid"] = (
                    grand_centroid - cluster_label_doc["summary"]["mean"]
                ) / cluster_label_doc["summary"]["std"]

                # this might not be needed
                center_stats["overall_z_score"] = ClusterReport.summary_statistics(
                    center_stats["by_features"]["overall_z_score"]
                )

                # this might not be needed
                center_stats["z_score"] = ClusterReport.summary_statistics(
                    center_stats["by_features"]["z_score"]
                )
                center_stats[
                    "z_score_grand_centroid"
                ] = ClusterReport.summary_statistics(
                    center_stats["by_features"]["z_score_grand_centroid"]
                )

                # squared errors are calculted by the centroids
                center_stats["squared_errors"] = ClusterReport.summary_statistics(
                    squared_errors, axis=2
                )

                squared_errors_by_col = []

                for f in range(len(squared_errors[0])):
                    squared_errors_by_col.append(
                        {
                            "cluster_id": str(f),
                            "squared_errors": ClusterReport.summary_statistics(
                                squared_errors[:, f], axis=2
                            ),
                        }
                    )

                center_stats["by_features"] = squared_errors_by_col

                cluster_label_doc["centers"] = center_stats

            cluster_label_doc["silhouette_score"] = ClusterReport.summary_statistics(
                self.X_silhouette_scores[cluster_bool], axis=2
            )

            self._internal_report["each"].append(cluster_label_doc)

        if self.has_centers():

            min_centroid_distance = min(
                c["centers"]["distance_from_centroid"]["min"]
                for c in self._internal_report["each"]
            )

            max_centroid_distance = self._internal_report["overall"][
                "centroids_distance_matrix"
            ].max()

            self._internal_report["overall"]["dunn_index"] = self.dunn_index(
                min_centroid_distance, max_centroid_distance
            )

        return self._internal_report

    def _get_centroid_vector(
        self, i: int = None, cluster_label: int = None, default_vector=None
    ):
        # If this is an outlier, we will automatically default to the grand centroid
        if cluster_label == self.outlier_label:
            if self.verbose:
                print(
                    "Outlier labels detected. Using the grand centroid for the outlier label."
                )
            return default_vector
        centers = self.get_centers()
        if isinstance(centers, (list, np.ndarray)):
            centroid_vector = centers[i]  # type: ignore
        elif isinstance(centers, dict):
            try:
                centroid_vector = centers[cluster_label]
            except KeyError:
                warnings.warn(Warning.CLUSTER_LABEL_NOT_IN_CENTROIDS)
                centroid_vector = default_vector
        else:
            raise ValueError("Centroid vector needs to be a list or a dictionary.")
        return centroid_vector

    def dunn_index(self, min_distance_from_centroid, max_centroid_distance):
        return np.around(
            min_distance_from_centroid / max_centroid_distance, DECIMAL_PLACES
        )

    def has_centers(self):
        return self.get_centers() is not None

    def _store_basic_centroid_stats(self, overall_report):
        """Store"""
        if self.has_centers():
            centroids = self.get_centers()
            overall_report["centroids"] = centroids
            overall_report["centroids_distance_matrix"] = np.around(
                pairwise_distances(centroids, metric="euclidean"), DECIMAL_PLACES
            )
            overall_report["grand_centroids"] = []
            overall_report["average_distance_between_centroids"] = np.around(
                (overall_report["centroids_distance_matrix"].sum(axis=1) - 1)
                / self.num_clusters,
                3,
            )

    @property
    def davies_bouldin_score(self):
        return pd.DataFrame(
            self.subset_documents(
                ["davies_bouldin_score"], [self.internal_report["overall"]]
            )
        )

    @property
    def calinski_harabasz_score(self):
        return pd.DataFrame(
            self.subset_documents(
                ["calinski_harabasz_score"], [self.internal_report["overall"]]
            )
        )

    @property
    def internal_overall_report(self):
        """
        View the internal overall report.
        """
        # TODO: Figure out a better way to present this than having index in the middle
        if not self.has_centers():
            metrics = pd.DataFrame(
                self.subset_documents(
                    ["davies_bouldin_score", "calinski_harabasz_score"],
                    [self.internal_report["overall"]],
                )
            )
            overall_df = pd.DataFrame(self.internal_report["overall"])[
                ["summary", "silhouette_score"]
            ]
            return pd.concat([metrics, overall_df.reset_index()], axis=1).fillna(" ")
        else:
            metrics = pd.DataFrame(
                self.subset_documents(
                    ["davies_bouldin_score", "calinski_harabasz_score", "dunn_index"],
                    [self.internal_report["overall"]],
                )
            )

            overall_df = pd.DataFrame(
                self.get_fields(
                    ["overall.summary", "overall.silhouette_score"],
                    self.internal_report,
                )
            ).T
            overall_df.columns = ["summary", "silhouette_score"]
            return pd.concat([metrics, overall_df.reset_index()], axis=1).fillna(" ")

    @staticmethod
    def calculate_centroids(X, cluster_labels):
        """Calculate the centes"""
        centroid_vectors = {}
        for label in cluster_labels:
            centroid_vectors[label] = X[cluster_labels == label].mean(axis=0)
        return centroid_vectors

    @staticmethod
    def calculate_medoids(X, cluster_labels):
        centroids = ClusterReport.calculate_centroids(X, cluster_labels)
        medoids = {}
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X)
        medoid_indexes = nbrs.kneighbors(
            np.array(list(centroids.values())), n_neighbors=1, return_distance=False
        )
        medoids = X[medoid_indexes]
        return medoids

    def get_class_rules(self, tree: DecisionTreeClassifier, feature_names: list):
        self.inner_tree: _tree.Tree = tree.tree_
        self.classes = tree.classes_
        self.class_rules_dict: Dict[Any, Any] = dict()
        self.tree_dfs()

    def tree_dfs(self, node_id=0, current_rule: Optional[list] = None):
        current_rule = [] if current_rule is None else current_rule
        # if not hasattr(self, "classes"):
        #    self.get_class_rules()

        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = self.inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = self.feature_names[split_feature]
            threshold = self.inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            self.tree_dfs(self.inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            self.tree_dfs(self.inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = self.inner_tree.value[node_id][0]
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
            # register new rule to dictionary
            selected_class = self.classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = self.class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            self.class_rules_dict[selected_class] = class_rules

    def cluster_reporting(self, data: pd.DataFrame, clusters, max_depth: int = 5):
        # Create Model
        tree = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
        tree.fit(data, clusters)
        print(tree.score(data, clusters))

        # Generate Report
        self.feature_names = data.columns
        self.get_class_rules(tree, self.feature_names)

        report_class_list = []

        for class_name in self.class_rules_dict.keys():
            rule_list = self.class_rules_dict[class_name]
            combined_string = ""
            for rule in rule_list:
                combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
            report_class_list.append((class_name, combined_string))

        cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
        cluster_instance_df.columns = ["class_name", "instance_count"]

        report_df = pd.DataFrame(report_class_list, columns=["class_name", "rule_list"])
        report_df = pd.merge(
            cluster_instance_df, report_df, on="class_name", how="left"
        )
        return report_df.sort_values(by="class_name")[
            ["class_name", "instance_count", "rule_list"]
        ]

    def __repr__(self):
        return "ClusterReport"
