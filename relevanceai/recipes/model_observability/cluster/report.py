import numpy as np
from relevanceai._api.endpoints.datasets import cluster
from relevanceai.apps import ReportApp
from relevanceai.recipes.model_observability.cluster.evaluator import ClusterEvaluator
from typing import Union, List, Dict, Any, Optional


class ClusterReport(ReportApp):
    def __init__(self, name: str, dataset, deployable_id: str = None, **kwargs):
        self.started_cluster_evaluator = False
        super().__init__(
            dataset=dataset,
            name=name,
            deployable_id=deployable_id,
            **kwargs,
        )

    def start_cluster_evaluator(
        self,
        X: Union[list, np.ndarray],
        cluster_labels: Union[List[Union[str, float]], np.ndarray],
        centroids: Union[list, np.ndarray, str] = None,
        cluster_names: Union[list, dict] = None,
        feature_names: Union[list, dict] = None,
        model=None,
        outlier_label: Union[str, int] = -1,
        metric: str = "euclidean",
        verbose: bool = False,
    ):
        self.evaluator = ClusterEvaluator(
            X=X,
            cluster_labels=cluster_labels,
            centroids=centroids,
            cluster_names=cluster_names,
            feature_names=feature_names,
            model=model,
            outlier_label=outlier_label,
            metric=metric,
            verbose=verbose,
        )
        self.started_cluster_evaluator = True
        self.evaluator.internal_overview_report()

    def start_cluster_evaluator_from_dataset(
        self, 
        vector_fields:list, 
        alias:str,
        feature_names: Union[list, dict] = None,
        metric: str = "euclidean",
        verbose: bool = False,
        show_progress_bar: bool = False,
    ):
        cluster_field = f"_cluster_.{'.'.join(vector_fields)}.{alias}"
        documents = self.dataset.get_all_documents(
            filters=[
                {
                    "field": field,
                    "filter_type": "exists",
                    "condition": ">=",
                    "condition_value": " ",
                } for field in [cluster_field] + vector_fields
            ],
            select_fields=[cluster_field] + vector_fields,
            show_progress_bar=show_progress_bar,
        )
        self.X = self.dataset.get_field_across_documents(vector_fields[0], documents)
        self.cluster_labels = self.dataset.get_field_across_documents(cluster_field, documents)
        self.centroids = {
            d["_id"] : self.dataset.get_field(vector_fields[0], d)
            for d in self.dataset.datasets.cluster.centroids.documents(
                dataset_id= self.dataset.dataset_id,
                vector_fields=vector_fields,
                alias=alias,
                page_size=9999,
                include_vector=True,
            )["results"]
        }
        self.start_cluster_evaluator(
            self.X, 
            self.cluster_labels, 
            self.centroids,
            # cluster_names=cluster_names,
            feature_names=feature_names,
            # model=model,
            # outlier_label=outlier_label,
            metric=metric,
            verbose=verbose,
        )

    # create wrapper to make sure cluster_evaluator is started

    def section_cluster_report(
        self,
        hierarchy_methods=["ward"],
        color_threshold: float = 1.25,
        plot_method=None,
        add=True,
    ):
        self.h1("Cluster report")
        self.section_cluster_overview_metrics()
        self.section_cluster_dendrogram(
            hierarchy_methods=hierarchy_methods,
            color_threshold=color_threshold,
            plot_method=plot_method,
        )
        self.section_cluster_distance_matrix()
        # self.section_cluster_indepth()

    def section_cluster_overview_metrics(self, add=True):
        self.h2("Overview of cluster metrics")
        self.paragraph(
            "When measuring the performance of a cluster there are 2 main things to look at"
        )
        self.bullet_list(
            [
                "Compactness (intra-cluster distance): Measures the variation within each cluster - how close observations are within each cluster.",
                "Separation (inter-cluster distance): How well separated are the clusters from each other.",
            ]
        )
        for metric, explanation in {
            "davies_bouldin_score" : "      [Compactness, Separation] (0 to infinity, lower is better) This calculates the ratio between each cluster's squared error to the distance between cluster centroids.",
            "calinski_harabasz_score" : "      [Compactness, Separation] (-infinity to infinity, higher is better) Similar to Davies Bouldin score, but also considers the 'group dispersion matrix' that considers the cluster size. Its equivalent to the Variance Ratio Criterion",
            "silhouette_score" : "      [Compactness, Separation] (-1 to 1, higher is better) This is the distance between a sample and all other points in the same cluster, and the same sample to the closest other clusters. This silhouette score is the average of every point’s silhouette score.",
            "total_squared_error_score" : "      [Compactness] (0 to infinity, lower is better) The average squared error between each point of a cluster to its centroid. Its equivalent to inertia.",
        }.items():
            metric_name = " ".join(metric.split("_")).title()
            self.paragraph(
                [self.bold(f"{metric_name}: "), self.evaluator.report[metric]], add=add
            )
            self.paragraph([self.italic(explanation)])
        plot, plotted_method = self.evaluator.plot_boxplot(self.evaluator.report["silhouette_score_summary"], name="Silhouette Score")
        self.plot_by_method(plot, title="Silhouette Score Box Plot", plot_method=plotted_method, add=add)
        plot, plotted_method = self.evaluator.plot_boxplot(self.evaluator.report["squared_error_summary"], name="Squared Error")
        self.plot_by_method(plot, title="Squared Error Box Plot", plot_method=plotted_method, add=add)

    def section_cluster_dendrogram(
        self,
        hierarchy_methods=["ward"],
        color_threshold: float = 1,
        orientation: str = "left",
        plot_method=None,
        add=True,
    ):
        self.h2("Overview of cluster dendrogram")
        self.paragraph(
            "A dendrogram shows the hierarchical relationship between cluster. This can be especially useful to determine which clusters to combined with hierarchical linkage."
        )
        for method in hierarchy_methods:
            height = max(15*self.evaluator.num_clusters, 300)
            plot, plotted_method = self.evaluator.plot_dendrogram(
                hierarchy_method=method,
                plot_method=plot_method,
                color_threshold=color_threshold,
                orientation=orientation,
            )
            self.plot_by_method(
                plot, 
                title=f"{method.title()} linkage dendrogram", 
                plot_method=plotted_method,
                height=height,
                add=add
            )

    def section_cluster_distance_matrix(self, metrics=["cosine", "euclidean"], decimals: int = 4, add=True):
        self.h2("Overview of cluster similarity matrix")
        self.paragraph(
            "Shows a heatmap of the similarity scores between different clusters. This can be especially useful to determine which clusters to combined."
        )
        for metric in metrics:
            plot, plotted_method = self.evaluator.plot_distance_matrix(metric=metric, decimals=decimals)
            chart_title = f"{metric.title()} similarity matrix"
            if metric in ["cosine"]:
                chart_title += " (higher is more similar)"
            else:
                chart_title += " (lower is more similar)"
            self.plot_by_method(plot, title=chart_title, plot_method=plotted_method, add=add)
