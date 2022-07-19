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
    ):
        return 

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
                "Compactness (intra-cluster distance): Measures the variation within each cluster, how close observations are within each cluster.",
                "Separation (inter-cluster distance): How well separated are the clusters from each other.",
            ]
        )
        for metric in [
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "silhouette_score",
        ]:
            metric_name = " ".join(metric.split("_")).title()
            self.paragraph(
                [self.bold(f"{metric_name}: "), self.evaluator.report[metric]], add=add
            )

    def section_cluster_dendrogram(
        self,
        hierarchy_methods=["ward"],
        color_threshold: float = 1,
        orientation: str = "left",
        plot_method=None,
        add=True,
    ):
        self.h2("Overview of cluster dendrogram")
        for method in hierarchy_methods:
            plot, plotted_method = self.evaluator.plot_dendrogram(
                hierarchy_method=method,
                plot_method=plot_method,
                color_threshold=color_threshold,
                orientation=orientation,
            )
            self.plot_by_method(plot, plot_method=plotted_method, add=add)

    def section_cluster_distance_matrix(self, decimals: int = 4, add=True):
        self.h2("Overview of cluster similarity matrix")
        plot, plotted_method = self.evaluator.plot_distance_matrix(decimals=decimals)
        self.plot_by_method(plot, plot_method=plotted_method, add=add)
