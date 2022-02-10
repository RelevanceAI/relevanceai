"""
Cluster Reporting Made Simple
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples,
)
from sklearn.metrics.pairwise import (
    pairwise_distances,
)
from sklearn.tree import _tree, DecisionTreeClassifier


class ClusterReport:
    def __init__(self, X, cluster_labels, num_clusters, model):
        self.X = X
        self.cluster_labels = cluster_labels
        self.num_clusters = num_clusters
        self.model = model

    @staticmethod
    def summary_statistics(array: np.ndarray, axis=0):
        if axis == 2:
            return {
                "sum": array.sum(),
                "mean": array.mean(),
                "standard_deviation": array.std(),
                "variance": array.var(),
                "min": array.min(),
                "max": array.max(),
                "12.5%": np.percentile(array, 12.5),
                "25%": np.percentile(array, 25),
                "37.5%": np.percentile(array, 37.5),
                "50%": np.percentile(array, 50),
                "62.5%": np.percentile(array, 62.5),
                "75%": np.percentile(array, 75),
                "87.5%": np.percentile(array, 87.5),
            }
        else:
            return {
                "sum": array.sum(axis=axis),
                "mean": array.mean(axis=axis),
                "standard_deviation": array.std(axis=axis),
                "variance": array.var(axis=axis),
                "min": array.min(axis=axis),
                "max": array.max(axis=axis),
                "12.5%": np.percentile(array, 12.5, axis=axis),
                "25%": np.percentile(array, 25, axis=axis),
                "37.5%": np.percentile(array, 37.5, axis=axis),
                "50%": np.percentile(array, 50, axis=axis),
                "62.5%": np.percentile(array, 62.5, axis=axis),
                "75%": np.percentile(array, 75, axis=axis),
                "87.5%": np.percentile(array, 87.5, axis=axis),
            }

    def get_cluster_internal_report(self):
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
        self.store_centroids_stats()

        label_value_counts = np.unique(self.cluster_labels, return_counts=True)

        cluster_report = {"frequency": {"total": 0, "each": {}}}

        for i, cluster_label in enumerate(label_value_counts[0]):
            cluster_bool = self.cluster_labels == cluster_label

            cluster_frequency = label_value_counts[1][i]
            cluster_report["frequency"]["total"] += cluster_frequency
            cluster_report["frequency"]["each"][cluster_label] = cluster_frequency

            self.cluster_internal_report["each"]["summary"][
                cluster_label
            ] = ClusterReport.summary_statistics(self.X)

            center_stats = {"by_features": {}}

            center_stats["by_features"]["summary"] = ClusterReport.summary_statistics(
                self.X[cluster_bool]
            )

            if hasattr(self.model, "cluster_centers_"):
                grand_centroid = self.X[cluster_bool].mean(axis=0)
                self.cluster_internal_report["overall"]["grand_centroids"].append(
                    grand_centroid
                )
                distances_from_centroid = pairwise_distances(
                    [self.model.cluster_centers_[i]], self.X[cluster_bool]
                )
                distances_from_centroid_to_another = pairwise_distances(
                    [self.model.cluster_centers_[i]], self.X[~cluster_bool]
                )

                center_stats[
                    "distance_from_centroid"
                ] = ClusterReport.summary_statistics(distances_from_centroid, axis=2)

                center_stats[
                    "distance_from_centroid_to_point_in_another_cluster"
                ] = ClusterReport.summary_statistics(
                    distances_from_centroid_to_another, axis=2
                )

                distances_from_grand_centroid = pairwise_distances(
                    [grand_centroid], self.X[cluster_bool]
                )
                distances_from_grand_centroid_to_another = pairwise_distances(
                    [grand_centroid], self.X[~cluster_bool]
                )
                center_stats[
                    "distance_from_grand_centroid"
                ] = ClusterReport.summary_statistics(
                    distances_from_grand_centroid, axis=2
                )
                center_stats[
                    "distance_from_grand_centroid_to_point_in_another_cluster"
                ] = ClusterReport.summary_statistics(
                    distances_from_grand_centroid_to_another, axis=2
                )

                center_stats["by_features"]["overall_z_score"] = (
                    self.model.cluster_centers_[i]
                    - self.cluster_internal_report["overall"]["summary"]["mean"]
                ) / self.cluster_internal_report["overall"]["summary"][
                    "standard_deviation"
                ]

                center_stats["by_features"]["z_score"] = (
                    self.model.cluster_centers_[i]
                    - self.cluster_internal_report["each"]["summary"][cluster_label][
                        "mean"
                    ]
                ) / self.cluster_internal_report["each"]["summary"][cluster_label][
                    "standard_deviation"
                ]

                squared_errors = np.square(
                    np.subtract(
                        [self.model.cluster_centers_[i]] * len(self.X[cluster_bool]),
                        self.X[cluster_bool],
                    )
                )

                center_stats["by_features"]["overall_z_score_grand_centroid"] = (
                    grand_centroid
                    - self.cluster_internal_report["overall"]["summary"]["mean"]
                ) / self.cluster_internal_report["overall"]["summary"][
                    "standard_deviation"
                ]

            center_stats[
                "overall_z_score_grand_centroid"
            ] = ClusterReport.summary_statistics(
                center_stats["by_features"]["overall_z_score_grand_centroid"]
            )

            center_stats["by_features"]["z_score_grand_centroid"] = (
                grand_centroid
                - self.cluster_internal_report["each"]["summary"][cluster_label]["mean"]
            ) / self.cluster_internal_report["each"]["summary"][cluster_label][
                "standard_deviation"
            ]

            # this might not be needed
            center_stats["overall_z_score"] = ClusterReport.summary_statistics(
                center_stats["by_features"]["overall_z_score"]
            )
            # this might not be needed
            center_stats["z_score"] = ClusterReport.summary_statistics(
                center_stats["by_features"]["z_score"]
            )
            center_stats["z_score_grand_centroid"] = ClusterReport.summary_statistics(
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

    def store_centroids_stats(self):
        if hasattr(self.model, "cluster_centers_"):
            self.cluster_internal_report["overall"][
                "centroids"
            ] = self.model.cluster_centers_
            self.cluster_internal_report["overall"][
                "centroids_distance_matrix"
            ] = pairwise_distances(self.model.cluster_centers_, metric="euclidean")
            self.cluster_internal_report["overall"]["grand_centroids"] = []
            self.cluster_internal_report["overall"][
                "average_distance_between_centroids"
            ] = (
                self.cluster_internal_report["overall"][
                    "centroids_distance_matrix"
                ].sum(axis=1)
                - 1
            ) / self.num_clusters

    def get_class_rules(self, tree: DecisionTreeClassifier, feature_names: list):
        self.inner_tree: _tree.Tree = tree.tree_
        self.classes = tree.classes_
        self.class_rules_dict = dict()
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
