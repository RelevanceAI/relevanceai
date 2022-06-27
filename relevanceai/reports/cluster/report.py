"""
Cluster Report: Evaluate Clusters
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
    report.report

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
    from scipy.cluster import hierarchy

from relevanceai.constants.warning import Warning


DECIMAL_PLACES = 3

def get_silhouette_grade(silhouette_score):
    # grades = ["F"] + [f"{grade}{sign}" for grade in ["E", "D", "C", "B", "A"] for sign in ["-", "", "+"]] + ["S"]
    grades = ["F", "E", "D", "C", "B", "A", "S"]
    scores = [(2 * i) / len(grades) for i in range(1, len(grades) + 1)]
    for score, grade in zip(scores, grades):
        if (silhouette_score + 1) < score:
            return grade
    return "N/A"

class ClusterReport:
    """
    Receive a cluster report to help you evaluate your clusters

    .. note::
        **Introduced in v1.0.0.**


    Parameters
    -------------

    X: np.ndarray
        The original data
    cluster_labels: List[str]
        A list of cluster labels
    centroids: Union[list, np.ndarray, str]
        The centroid vectors. If supplied it will use that. Otherwise, it will try to infer them
        from the model or calculate it based off the labels. To calculate mediods, let centroids = "mediods"
    model
        The model used for clustering.
    num_clusters: Optional[int]
        The number of clusters. This is required if we can't actually tell how many clusters there are
    outlier_label: Optional[str, int]
        The label if it is an outlier
    metric
        The metric to use for calculating distance, "euclidean", "cosine", ...
    verbose
        Whether to print stuff

    """

    def __init__(
        self,
        X: Union[list, np.ndarray],
        cluster_labels: Union[List[Union[str, float]], np.ndarray],
        centroids: Union[list, np.ndarray, str] = None,
        cluster_names: Union[list, dict] = None,
        feature_names: Union[list, dict] = None,
        model=None,
        outlier_label: Union[str, int] = -1,
        metric: str = "euclidean",
        top_n_features: int = 10,
        verbose: bool = False,
    ):
        if isinstance(X, list):
            self.X = np.array(X)
        else:
            self.X = X
        if isinstance(cluster_labels, list):
            self.cluster_labels = np.array(cluster_labels)
        else:
            self.cluster_labels = cluster_labels
        self.cluster_names = cluster_names
        self.feature_names = feature_names
        self.num_clusters = len(set(cluster_labels))
        self.model = model
        if centroids == "mediods":
            centroids = self.calculate_medoids(self.X, self.cluster_labels)
        if not centroids and not model:
            centroids = self.calculate_centroids(self.X, self.cluster_labels)
        elif not centroids and model:
            try:
                centroids = self.get_centroids_from_model(model)
            except:
                centroids = self.calculate_centroids(self.X, self.cluster_labels)
        self._typecheck_centroid_vectors(centroids)
        self.centroids = centroids
        self.outlier_label = outlier_label
        self.metric = metric
        self.distance_matrix = pairwise_distances(
            list(self.centroids.values()), metric=self.metric
        )
        self.linkage = hierarchy.linkage(list(self.centroids.values()), 'ward')
        self.top_n_features = top_n_features
        self.verbose = verbose

        self._report = {}
        if self.model:
            self._report['params'] = {}


    def _typecheck_centroid_vectors(
        self, centroid_vectors: Optional[Union[list, Dict, np.ndarray]] = None
    ):
        if isinstance(centroid_vectors, (list, np.ndarray)):
            warnings.warn(Warning.CENTROID_VECTORS)

    @staticmethod
    def summary_statistics(array: np.ndarray, axis=0, simple=False):
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
                # "12_5%": np.percentile(array, 12.5),
                "25%": np.percentile(array, 25),
                # "37_5%": np.percentile(array, 37.5),
                "50%": np.percentile(array, 50),
                # "62_5%": np.percentile(array, 62.5),
                "75%": np.percentile(array, 75),
                # "87_5%": np.percentile(array, 87.5),
            }
        else:
            return {
                "sum": array.sum(axis=axis),
                "mean": array.mean(axis=axis),
                "std": array.std(axis=axis),
                "variance": array.var(axis=axis),
                "min": array.min(axis=axis),
                "max": array.max(axis=axis),
                # "12_5%": np.percentile(array, 12.5, axis=axis),
                "25%": np.percentile(array, 25, axis=axis),
                # "37_5%": np.percentile(array, 37.5, axis=axis),
                "50%": np.percentile(array, 50, axis=axis),
                # "62_5%": np.percentile(array, 62.5, axis=axis),
                "75%": np.percentile(array, 75, axis=axis),
                # "87_5%": np.percentile(array, 87.5, axis=axis),
            }

    def get_distance_from_centroid(self, cluster_data, center_vector):
        distances_from_centroid = pairwise_distances([center_vector], cluster_data)
        return self.summary_statistics(distances_from_centroid, axis=2)

    def get_distance_from_centroid_to_another(self, other_cluster_data, center_vector):
        """Store the distances from a centroid to another."""
        distances_from_centroid_to_another = pairwise_distances(
            [center_vector], other_cluster_data
        )
        return self.summary_statistics(
            distances_from_centroid_to_another, axis=2
        )

    @staticmethod
    def calculate_centroids(X, cluster_labels):
        """Calculate the centroids"""
        centroid_vectors = {}
        for label in cluster_labels:
            centroid_vectors[label] = X[cluster_labels == label].mean(axis=0)
        return centroid_vectors

    def calculate_medoids(self, X, cluster_labels):
        centroids = self.calculate_centroids(X, cluster_labels)
        medoids = {}
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X)
        medoid_indexes = nbrs.kneighbors(
            np.array(list(centroids.values())), n_neighbors=1, return_distance=False
        )
        medoids = X[medoid_indexes]
        return medoids

    def get_centroids_from_model(self, model, output_format="array"):
        if hasattr(model, "cluster_centers_"):
            return model.cluster_centers_
        elif hasattr(model, "get_centers"):
            return model.get_centers()
        else:
            raise Exception

    @staticmethod
    def calculate_z_score(value, mean, std):
        return (value - mean) / std

    @staticmethod
    def calculate_dunn_index(min_distance_from_centroid, max_centroid_distance):
        return min_distance_from_centroid / max_centroid_distance

    @staticmethod
    def calculate_squared_error(X, Y):
        return np.square(
            np.subtract(
                X,
                Y,
            )
        )

    @staticmethod
    def get_top_n_index(values, n, reverse=False):
        if reverse:
            return values.argsort()[-n:][::-1]
        else:
            return values.argsort()[:n]

    def get_cluster_name(self, cluster_index):
        return self.cluster_names[cluster_index] if self.cluster_names else f"cluster-{str(cluster_index)}"

    def get_feature_name(self, feature_index):
        return self.feature_names[feature_index] if self.feature_names else f"feature-{str(feature_index)}"


    @property  # type: ignore
    def internal_report(self):
        """
        Provide the standard clustering report.
        """
        self.X_silhouette_scores = silhouette_samples(
            self.X, self.cluster_labels, metric=self.metric
        )
        silouhette_score = self.X_silhouette_scores.mean()
        grade = get_silhouette_grade(silouhette_score)
        self.X_summary = self.summary_statistics(self.X)

        self._internal_report = {
            "grade": grade,
            "overall": {
                # "summary": self.X_summary,
                "davies_bouldin_score": davies_bouldin_score(
                    self.X, self.cluster_labels
                ),
                "calinski_harabasz_score": calinski_harabasz_score(
                    self.X, self.cluster_labels
                ),
                "silouhette_score" : silouhette_score,
                "silhouette_score_summary": self.summary_statistics(
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
        }

        labels, label_counts = np.unique(self.cluster_labels, return_counts=True)

        cluster_report = {"frequency": {"total": 0, "each": {}}}

        #get stats for each cluster
        for i, cluster_label in enumerate(labels):
            if i == self.outlier_label:
                cluster_name = "outlier_cluster"
            else:
                cluster_name = self.get_cluster_name(i)
            cluster_report["frequency"]["total"] += label_counts[i]
            cluster_report["frequency"]["each"] = label_counts[i]

            cluster_bool_mask = self.cluster_labels == cluster_label
            current_cluster_data = self.X[cluster_bool_mask]
            other_cluster_data = self.X[~cluster_bool_mask]
            current_summary = self.summary_statistics(current_cluster_data, axis=2)
            centroid_vector = self.centroids[i]


            silouhette_summary =  self.summary_statistics(
                self.X_silhouette_scores[cluster_bool_mask], axis=2
            )
            cluster_label_doc = {
                "cluster_id": cluster_name,
                "cluster_index": cluster_label,
                "centroid_vector_" : centroid_vector,
                "frequency" : label_counts[i],
                "summary" : current_summary,
                "silouhette_score" : silouhette_summary["mean"],
                "silouhette_score_summary" : silouhette_summary,
                "closest_clusters" : [
                    self.get_cluster_name(c) for c in self.get_top_n_index(self.distance_matrix[i], self.top_n_features, reverse=False)
                ],
                "furthest_clusters" : [
                    self.get_cluster_name(c) for c in self.get_top_n_index(self.distance_matrix[i], self.top_n_features, reverse=True)
                ],
                "by_features" : {}
            }

            # squared errors are calculted by the centroids
            squared_errors = self.calculate_squared_error(
                [centroid_vector] * len(current_cluster_data),
                current_cluster_data,
            )
            cluster_label_doc["mean_squared_errors"] = squared_errors.mean()

            squared_errors_by_feature = []
            for f in range(len(squared_errors[0])):
                squared_errors_by_feature.append(
                    {
                        "feature_index" : f,
                        "feature_id": self.get_feature_name(f),
                        "squared_errors": squared_errors[:, f].mean(),
                    }
                )
            #filter by the lowest 10 squared errors
            cluster_label_doc["by_features"]["lowest_squared_errors"] = sorted(
                squared_errors_by_feature, key=lambda x:x['squared_errors'], 
                reverse=False)[:self.top_n_features]
            cluster_label_doc["by_features"]["highest_squared_errors"] = sorted(
                squared_errors_by_feature, key=lambda x:x['squared_errors'], 
                reverse=True)[:self.top_n_features]

            #distances
            cluster_label_doc[
                "distance_from_centroid_summary"
            ] = self.get_distance_from_centroid(
                current_cluster_data, centroid_vector
            )
            cluster_label_doc[
                "distance_from_centroid_to_another_cluster_summary"
            ] = self.get_distance_from_centroid_to_another(
                other_cluster_data, centroid_vector
            )

            #z score
            overall_z_score = self.calculate_z_score(
                centroid_vector,
                self.X_summary["mean"],
                self.X_summary["std"],
            )
            cluster_label_doc["by_features"][
                "highest_overall_z_score"
            ] = [
                {"feature_index":f, "feature_id":self.get_feature_name(f), "z_score": overall_z_score[f]}
                for f in self.get_top_n_index(overall_z_score, self.top_n_features, reverse=True)
            ]
            cluster_label_doc["by_features"][
                "lowest_overall_z_score"
            ] = [
                {"feature_index":f, "feature_id":self.get_feature_name(f), "z_score": overall_z_score[f]}
                for f in self.get_top_n_index(overall_z_score, self.top_n_features, reverse=False)
            ]

            cluster_z_score = self.calculate_z_score(
                centroid_vector,
                cluster_label_doc["summary"]["mean"],
                cluster_label_doc["summary"]["std"],
            )
            cluster_label_doc["by_features"][
                "highest_z_score"
            ] = [
                {"feature_index":f, "feature_id":self.get_feature_name(f), "z_score": cluster_z_score[f]}
                 for f in self.get_top_n_index(cluster_z_score, self.top_n_features, reverse=True)
            ]
            cluster_label_doc["by_features"][
                "lowest_z_score"
            ] = [
                {"feature_index":f, "feature_id":self.get_feature_name(f), "z_score": cluster_z_score[f]}
                 for f in self.get_top_n_index(overall_z_score, self.top_n_features, reverse=False)
            ]

            cluster_label_doc["z_score"] = cluster_z_score.mean()

            self._internal_report["each"].append(cluster_label_doc)

        #calculate dunn index
        min_centroid_distance = min(
            c["distance_from_centroid_summary"]["min"]
            for c in self._internal_report["each"]
        )
        max_centroid_distance = self.distance_matrix.max()
        self._internal_report["overall"]["dunn_index"] = self.calculate_dunn_index(
            min_centroid_distance, max_centroid_distance
        )

        return self._internal_report

    @property  # type: ignore
    def report(self):
        self._report = self.internal_report
        self._report['linkage'] = self.linkage
        self._report["centroids_distance_matrix"] = self.distance_matrix
        return self._report

    def __repr__(self):
        return "ClusterReport"