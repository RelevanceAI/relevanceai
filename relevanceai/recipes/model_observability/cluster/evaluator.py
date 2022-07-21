import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Any, Optional
from scipy.cluster import hierarchy
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples,
)
from sklearn.metrics.pairwise import (
    pairwise_distances,
)


class ClusterEvaluator:
    """
    Evaluate your clusters

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
        centroids=None,
        # centroids: Union[list, np.ndarray, str] = None,
        cluster_names: Union[list, dict] = None,
        feature_names: Union[list, dict] = None,
        model=None,
        outlier_label: Union[str, int] = -1,
        metric: str = "euclidean",
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
            centroids = self._calculate_medoids(self.X, self.cluster_labels)
        if not centroids and not model:
            centroids = self._calculate_centroids(self.X, self.cluster_labels)
        elif not centroids and model:
            try:
                centroids = self._get_centroids_from_model(model)
            except:
                centroids = self._calculate_centroids(self.X, self.cluster_labels)
        if isinstance(centroids, (list, np.ndarray)):
            centroids = {i:c for i, c in enumerate(centroids)}
        if not isinstance(centroids, (list, dict, np.ndarray)):
            raise TypeError("centroid_vectors should be of type List or Numpy array")

        self.centroids = centroids
        self.outlier_label = outlier_label
        self.metric = metric
        if isinstance(self.centroids, dict):
            self.distance_matrix = pairwise_distances(
                list(self.centroids.values()), metric=self.metric
            )
        else:
            self.distance_matrix = pairwise_distances(
                self.centroids, metric=self.metric
            )
        self.verbose = verbose

    def _get_cluster_name(self, cluster_index):
        return (
            self.cluster_names[cluster_index]
            if self.cluster_names
            else f"cluster-{str(cluster_index)}"
        )

    def _get_feature_name(self, feature_index):
        return (
            self.feature_names[feature_index]
            if self.feature_names
            else f"feature-{str(feature_index)}"
        )

    def _get_centroids_from_model(self, model, output_format="array"):
        if hasattr(model, "cluster_centers_"):
            return model.cluster_centers_
        elif hasattr(model, "get_centers"):
            return model.get_centers()
        else:
            raise Exception

    def _calculate_centroids(self, X, cluster_labels):
        """Calculate the centroids"""
        centroid_vectors = {}
        for label in np.unique(cluster_labels):
            centroid_vectors[label] = X[cluster_labels == label].mean(axis=0)
        return centroid_vectors

    def _calculate_medoids(self, X, cluster_labels):
        centroids = self._calculate_centroids(X, cluster_labels)
        medoids = {}
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X)
        medoid_indexes = nbrs.kneighbors(
            np.array(list(centroids.values())), n_neighbors=1, return_distance=False
        )
        medoids = X[medoid_indexes]
        return medoids

    def hierarchy_linkage(self, hierarchy_method: str = "ward"):
        self.linkage = hierarchy.linkage(
            list(self.centroids.values()), hierarchy_method
        )
        return self.linkage

    def pyplot_dendrogram(
        self, hierarchy_method=None, color_threshold=1.25, orientation="left", ax=None
    ):
        fig, ax = plt.subplots()
        if hierarchy_method:
            self.linkage = self.hierarchy_linkage(hierarchy_method=hierarchy_method)
        elif not hasattr(self, "linkage"):
            self.linkage = self.hierarchy_linkage(ax=ax)
        self.dendrogram = hierarchy.dendrogram(
            self.linkage,
            labels=self.cluster_names,
            orientation=orientation,
            color_threshold=color_threshold,
            ax=ax,
        )
        return fig

    def plotly_dendrogram(
        self, hierarchy_method=None, color_threshold=1.25, orientation="left"
    ):
        if hierarchy_method:
            self.linkage = self.hierarchy_linkage(hierarchy_method=hierarchy_method)
        elif not hasattr(self, "linkage"):
            self.linkage = self.hierarchy_linkage()
        try:
            import plotly.figure_factory as ff

            fig = ff.create_dendrogram(
                np.array(list(self.centroids.values())),
                labels=self.cluster_names,
                orientation=orientation,
                color_threshold=color_threshold,
                linkagefun=lambda x: self.linkage,
            )
            return fig
        except ImportError:
            raise ImportError(
                "This requires plotly installed, install with `pip install -U plotly`"
            )
        except TypeError as e:
            raise TypeError(
                e
                + " This is a common error that can be fixed with `pip install pyyaml==5.4.1`"
            )

    def plot_dendrogram(
        self,
        plot_method=None,
        hierarchy_method=None,
        color_threshold: float = 1,
        orientation: str = "left",
    ):
        if plot_method == "pyplot":
            return (
                self.pyplot_dendrogram(
                    hierarchy_method=hierarchy_method,
                    color_threshold=color_threshold,
                    orientation=orientation,
                ),
                "pyplot",
            )
        elif plot_method == "plotly":
            return (
                self.plotly_dendrogram(
                    hierarchy_method=hierarchy_method,
                    color_threshold=color_threshold,
                    orientation=orientation,
                ),
                "plotly",
            )
        else:
            try:
                return (
                    self.plotly_dendrogram(
                        hierarchy_method=hierarchy_method,
                        color_threshold=color_threshold,
                        orientation=orientation,
                    ),
                    "plotly",
                )
            except:
                return (
                    self.pyplot_dendrogram(
                        hierarchy_method=hierarchy_method,
                        color_threshold=color_threshold,
                        orientation=orientation,
                    ),
                    "pyplot",
                )

    def plot_distance_matrix(self, metric="euclidean", decimals=4):
        try:
            import plotly.express as px

            if metric == "euclidean":
                distance_matrix = self.distance_matrix
            else:
                if isinstance(self.centroids, dict):
                    distance_matrix = pairwise_distances(
                        list(self.centroids.values()), metric=metric
                    )
                else:
                    distance_matrix = pairwise_distances(
                        self.centroids, metric=metric
                    )
            
            if self.cluster_names:
                return (
                    px.imshow(
                        pd.DataFrame(
                            np.round(distance_matrix, 4),
                            columns=self.cluster_names,
                            index=self.cluster_names,
                        ),
                        text_auto=True,
                    ),
                    "plotly",
                )
            else:
                return (
                    px.imshow(np.round(distance_matrix, 4), text_auto=True),
                    "plotly",
                )
        except ImportError:
            raise ImportError(
                "This requires plotly installed, install with `pip install -U plotly`"
            )
        except TypeError as e:
            raise TypeError(
                e
                + " This is a common error that can be fixed with `pip install pyyaml==5.4.1`"
            )

    def plot_boxplot(self, summary_stats, name=""):
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(
                go.Box(
                    y=[
                        summary_stats['min'], 
                        summary_stats['25%'], 
                        summary_stats['50%'], 
                        summary_stats['75%'], 
                        summary_stats['max']
                    ], 
                    name=name
                )
            )
            return fig, "plotly"

        except ImportError:
            raise ImportError(
                "This requires plotly installed, install with `pip install -U plotly`"
            )

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
                "25%": np.percentile(array, 25),
                "50%": np.percentile(array, 50),
                "75%": np.percentile(array, 75),
            }
        else:
            return {
                "sum": array.sum(axis=axis),
                "mean": array.mean(axis=axis),
                "std": array.std(axis=axis),
                "variance": array.var(axis=axis),
                "min": array.min(axis=axis),
                "max": array.max(axis=axis),
                "25%": np.percentile(array, 25, axis=axis),
                "50%": np.percentile(array, 50, axis=axis),
                "75%": np.percentile(array, 75, axis=axis),
            }

    def silhouette_samples(self):
        if not hasattr(self, "X_silhouette_samples"):
            self.X_silhouette_samples = silhouette_samples(
                self.X, self.cluster_labels, metric=self.metric
            )
        return self.X_silhouette_samples

    def silhouette_score(self):
        if not hasattr(self, "X_silhouette_samples"):
            self.silhouette_samples()
        return self.X_silhouette_samples.mean()

    def calinski_harabasz_score(self):
        return calinski_harabasz_score(self.X, self.cluster_labels)

    def davies_bouldin_score(self):
        return davies_bouldin_score(self.X, self.cluster_labels)

    @staticmethod
    def dunn_index(min_distance_from_centroid, max_centroid_distance):
        return min_distance_from_centroid / max_centroid_distance

    def distance_from_centroid(self, cluster_data, centroid):
        distances_from_centroid = pairwise_distances([centroid], cluster_data)
        return self.summary_statistics(distances_from_centroid, axis=2)

    def distance_from_centroid_to_another(self, other_cluster_data, centroid):
        """Store the distances from a centroid to another."""
        distances_from_centroid_to_another = pairwise_distances(
            [centroid], other_cluster_data
        )
        return self.summary_statistics(distances_from_centroid_to_another, axis=2)

    def squared_error_samples(self):
        if not hasattr(self, "X_squared_error_samples"):
            self.X_squared_error_samples = np.square(
                np.subtract(
                    [self.centroids[c] for c in self.cluster_labels],
                    self.X,
                )
            )
        return self.X_squared_error_samples

    def squared_error_score(self):
        if not hasattr(self, "X_squared_error_samples"):
            self.squared_error_samples()
        return self.X_squared_error_samples.sum()

    def mean_squared_error_score(self):
        if not hasattr(self, "X_silhouette_samples"):
            self.squared_error_samples()
        return self.X_squared_error_samples.mean()

    def squared_error_features_from_samples(self, squared_error_samples):
        squared_errors_by_feature = []
        for f in range(len(squared_error_samples[0])):
            squared_errors_by_feature.append(
                {
                    "feature_index": f,
                    "feature_id": self._get_feature_name(f),
                    "squared_errors": squared_error_samples[:, f].mean(),
                }
            )
        return squared_errors_by_feature

    @staticmethod
    def z_score(value, mean, std):
        return (value - mean) / std

    def closest_clusters(self, cluster_index, n_clusters):
        return [
            self._get_cluster_name(c)
            for c in self.distance_matrix[cluster_index].argsort()[:n_clusters]
        ]

    def furthest_clusters(self, cluster_index, n_clusters):
        return [
            self._get_cluster_name(c)
            for c in self.distance_matrix[cluster_index].argsort()[-n_clusters:][::-1]
        ]

    def internal_overview_report(
        self,
        store_centroids: bool = True,
        store_distance_matrix: bool = True,
        save=True,
    ):
        report = {}
        report["silhouette_score_summary"] = self.summary_statistics(
            self.silhouette_samples()
        )
        report["squared_error_summary"] = self.summary_statistics(
            self.squared_error_samples(), axis=2
        )
        report["calinski_harabasz_score"] = self.calinski_harabasz_score()
        report["davies_bouldin_score"] = self.davies_bouldin_score()
        report["silhouette_score"] = self.silhouette_score()
        report["total_squared_error_score"] = self.squared_error_score()
        report["mean_squared_error_score"] = self.mean_squared_error_score()
        # if self.model:
        #     report["_model_params"] = self.model.__dict__
        report["metric"] = self.metric
        report["num_clusters"] = self.num_clusters
        report["total_frequency"] = len(self.X)
        if self.cluster_names:
            report["cluster_names"] = self.cluster_names
        if self.feature_names:
            report["feature_names"] = self.feature_names
        if store_centroids:
            report["centroids"] = self.centroids
        if store_distance_matrix:
            report["distance_matrix"] = self.distance_matrix
        if save:
            self.report = report
        return report

    def internal_report(
        self,
        top_n_clusters: int = 5,
        top_n_features: int = 5,
        store_squared_errors: bool = True,
        store_distances: bool = True,
        store_centroids: bool = False,
        store_distance_matrix: bool = False,
        save=True,
    ):
        report = {}
        report.update(
            self.internal_overview_report(
                store_centroids=store_centroids,
                store_distance_matrix=store_distance_matrix,
            )
        )
        report["each_cluster_frequency"] = []
        report["each_cluster"] = []
        labels, label_counts = np.unique(self.cluster_labels, return_counts=True)
        for i, cluster_label in enumerate(labels):
            if cluster_label == self.outlier_label:
                cluster_name = "outlier_cluster"
            else:
                cluster_name = self._get_cluster_name(i)
            cluster_bool = self.cluster_labels == cluster_label
            current_cluster_data = self.X[cluster_bool]
            other_cluster_data = self.X[~cluster_bool]
            centroid_vector = self.centroids[cluster_label]
            current_summary = self.summary_statistics(current_cluster_data, axis=2)
            silouhette_summary = self.summary_statistics(
                self.X_silhouette_samples[cluster_bool], axis=2
            )
            cluster_document = {
                "cluster_id": cluster_name,
                "cluster_index": cluster_label,
                "centroid_vector_": centroid_vector,
                "frequency": label_counts[i],
                "summary": current_summary,
                "silouhette_score": silouhette_summary["mean"],
                "silouhette_score_summary": silouhette_summary,
                "closest_clusters": self.closest_clusters(i, top_n_clusters),
                "furthest_clusters": self.furthest_clusters(i, top_n_clusters),
                "features": {},
            }
            # squared errors
            if store_squared_errors:
                cluster_document["squared_error_summary"] = self.summary_statistics(
                    self.X_squared_error_samples[cluster_bool], axis=2
                )
                squared_error_by_features = self.squared_error_features_from_samples(
                    self.X_squared_error_samples[cluster_bool]
                )
                cluster_document["features"]["lowest_squared_errors"] = sorted(
                    squared_error_by_features,
                    key=lambda x: x["squared_errors"],
                    reverse=False,
                )[:top_n_features]
                cluster_document["features"]["highest_squared_errors"] = sorted(
                    squared_error_by_features,
                    key=lambda x: x["squared_errors"],
                    reverse=True,
                )[:top_n_features]
            # distances
            if store_distances:
                cluster_document[
                    "distance_from_centroid_summary"
                ] = self.distance_from_centroid(current_cluster_data, centroid_vector)
                cluster_document[
                    "distance_from_centroid_to_another_cluster_summary"
                ] = self.distance_from_centroid_to_another(
                    other_cluster_data, centroid_vector
                )
            report["each_cluster"].append(cluster_document)

        if store_distances:
            report["dunn_index_score"] = self.dunn_index(
                min_distance_from_centroid=min(
                    c["distance_from_centroid_summary"]["min"]
                    for c in report["each_cluster"]
                ),
                max_centroid_distance=self.distance_matrix.max(),
            )
        if save:
            self.report = report
        return report
