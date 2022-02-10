"""
Automated Cluster Reporting

.. warning::
    This is a beta feature.

.. note::
    **Introduced in v1.0.0.**


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


    from relevanceai.cluster_report.cluster_report import ClusterReport
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

    internal_report = report.get_cluster_internal_report()

"""

import pandas as pd
import numpy as np
from relevanceai.warnings import warn_function_is_work_in_progress
from typing import Union, List, Dict, Any

try:
    from sklearn.metrics import (
        davies_bouldin_score,
        calinski_harabasz_score,
        silhouette_samples,
    )
    from sklearn.metrics.pairwise import (
        pairwise_distances,
    )
    from sklearn.tree import _tree, DecisionTreeClassifier
    from sklearn.cluster import KMeans
except ModuleNotFoundError as e:
    pass


class ClusterReport:
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

    """

    def __init__(
        self,
        X: Union[list, np.ndarray],
        cluster_labels: List[Union[str, float]],
        model: KMeans = None,
        num_clusters: int = None,
    ):
        warn_function_is_work_in_progress()
        if isinstance(X, list):
            self.X = np.array(X)
        else:
            self.X = X
        if isinstance(cluster_labels, list):
            self.cluster_labels = np.array(cluster_labels)
        else:
            self.cluster_labels = cluster_labels
        self.num_clusters = (
            len(cluster_labels) if num_clusters is None else num_clusters
        )
        self.model = model

    @staticmethod
    def summary_statistics(array: np.ndarray, axis=0):
        """
        Basic summary statistics
        """
        if axis == 2:
            return {
                "sum": array.sum(),
                "mean": array.mean(),
                "std": array.std(),
                "variance": array.var(),
                "min": array.min(),
                "max": array.max(),
                "12_5%": np.percentile(array, 12.5),
                "25%": np.percentile(array, 25),
                "37_5%": np.percentile(array, 37.5),
                "50%": np.percentile(array, 50),
                "62_5%": np.percentile(array, 62.5),
                "75%": np.percentile(array, 75),
                "87_5%": np.percentile(array, 87.5),
            }
        else:
            return {
                "sum": array.sum(axis=axis),
                "mean": array.mean(axis=axis),
                "std": array.std(axis=axis),
                "variance": array.var(axis=axis),
                "min": array.min(axis=axis),
                "max": array.max(axis=axis),
                "12_5%": np.percentile(array, 12.5, axis=axis),
                "25%": np.percentile(array, 25, axis=axis),
                "37_5%": np.percentile(array, 37.5, axis=axis),
                "50%": np.percentile(array, 50, axis=axis),
                "62_5%": np.percentile(array, 62.5, axis=axis),
                "75%": np.percentile(array, 75, axis=axis),
                "87_5%": np.percentile(array, 87.5, axis=axis),
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

    # TODO: Implement caching
    def get_centers(self):
        # Here we add support for both RelevanceAI cluster models
        # but also regular sklearn cluster models
        if hasattr(self.model, "cluster_centers_"):
            return self.get_centers()
        elif hasattr(self.model, "get_centers"):
            return self.model.get_centers()
        else:
            import warnings

            warnings.warn("Missing centroids.")
            return

    def get_cluster_internal_report(self):
        """
        Provide the standard clustering report.
        """
        self.X_silouhette_scores = silhouette_samples(
            self.X, self.cluster_labels, metric="euclidean"
        )
        self.cluster_internal_report = {
            "overall": {
                "summary": ClusterReport.summary_statistics(self.X),
                "davies_bouldin_score": davies_bouldin_score(
                    self.X, self.cluster_labels
                ),
                "calinski_harabasz_score": calinski_harabasz_score(
                    self.X, self.cluster_labels
                ),
                "silouhette_score": ClusterReport.summary_statistics(
                    self.X_silouhette_scores
                ),
            },
            "each": {
                "summary": {},
                "centers": {},
                "silouhette_score": {},
            },
        }
        self._store_basic_centroid_stats(self.cluster_internal_report["overall"])

        label_value_counts = np.unique(self.cluster_labels, return_counts=True)

        cluster_report = {"frequency": {"total": 0, "each": {}}}

        for i, cluster_label in enumerate(label_value_counts[0]):
            cluster_bool = self.cluster_labels == cluster_label

            specific_cluster_data = self.X[cluster_bool]
            other_cluster_data = self.X[~cluster_bool]

            cluster_frequency = label_value_counts[1][i]
            cluster_report["frequency"]["total"] += cluster_frequency
            cluster_report["frequency"]["each"][cluster_label] = cluster_frequency

            self.cluster_internal_report["each"]["summary"][
                cluster_label
            ] = ClusterReport.summary_statistics(self.X)

            # If each value of the vector is important
            center_stats = {"by_features": {}}

            center_stats["by_features"]["summary"] = ClusterReport.summary_statistics(
                specific_cluster_data
            )

            squared_errors = np.square(
                np.subtract(
                    [self.get_centers()[i]] * len(self.X[cluster_bool]),
                    self.X[cluster_bool],
                )
            )
            if hasattr(self.model, "cluster_centers_"):

                grand_centroid = self.X[cluster_bool].mean(axis=0)
                self.cluster_internal_report["overall"]["grand_centroids"].append(
                    grand_centroid
                )

                center_stats[
                    "distance_from_centroid"
                ] = self.get_distance_from_centroid(
                    specific_cluster_data, self.get_centers()[i]
                )

                center_stats[
                    "distance_from_centroid_to_point_in_another_cluster"
                ] = self.get_distance_from_centroid_to_another(
                    other_cluster_data, self.get_centers()[i]
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
                    self.get_centers()[i],
                    self.cluster_internal_report["overall"]["summary"]["mean"],
                    self.cluster_internal_report["overall"]["summary"]["std"],
                )

                center_stats["by_features"]["z_score"] = ClusterReport.get_z_score(
                    self.get_centers()[i],
                    self.cluster_internal_report["each"]["summary"][cluster_label][
                        "mean"
                    ],
                    self.cluster_internal_report["each"]["summary"][cluster_label][
                        "std"
                    ],
                )

                center_stats["by_features"][
                    "overall_z_score_grand_centroid"
                ] = ClusterReport.get_z_score(
                    grand_centroid,
                    self.cluster_internal_report["overall"]["summary"]["mean"],
                    self.cluster_internal_report["overall"]["summary"]["std"],
                )

                center_stats[
                    "overall_z_score_grand_centroid"
                ] = ClusterReport.summary_statistics(
                    center_stats["by_features"]["overall_z_score_grand_centroid"]
                )

                center_stats["by_features"]["z_score_grand_centroid"] = (
                    grand_centroid
                    - self.cluster_internal_report["each"]["summary"][cluster_label][
                        "mean"
                    ]
                ) / self.cluster_internal_report["each"]["summary"][cluster_label][
                    "std"
                ]

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

                center_stats["squared_errors"] = ClusterReport.summary_statistics(
                    squared_errors, axis=2
                )

            squared_errors_by_col = {}

            for f in range(len(squared_errors[0])):
                squared_errors_by_col[f] = ClusterReport.summary_statistics(
                    squared_errors[:, f], axis=2
                )

            center_stats["by_features"]["squared_errors"] = squared_errors_by_col
            self.cluster_internal_report["each"]["centers"][
                cluster_label
            ] = center_stats
            self.cluster_internal_report["each"]["silouhette_score"][
                cluster_label
            ] = ClusterReport.summary_statistics(
                self.X_silouhette_scores[cluster_bool], axis=2
            )

        return self.cluster_internal_report

    def has_centers(self):
        return self.get_centers() is not None

    def _store_basic_centroid_stats(self, overall_report):
        """Store"""
        if self.has_centers():
            overall_report["centroids"] = self.get_centers()
            overall_report["centroids_distance_matrix"] = pairwise_distances(
                self.get_centers(), metric="euclidean"
            )
            overall_report["grand_centroids"] = []
            overall_report["average_distance_between_centroids"] = (
                overall_report["centroids_distance_matrix"].sum(axis=1) - 1
            ) / self.num_clusters

    def get_class_rules(self, tree: DecisionTreeClassifier, feature_names: list):
        self.inner_tree: _tree.Tree = tree.tree_
        self.classes = tree.classes_
        self.class_rules_dict: Dict[Any, Any] = dict()
        self.tree_dfs()

    def tree_dfs(self, node_id=0, current_rule=[]):
        if not hasattr(self, "classes"):
            self.get_class_rules()

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
