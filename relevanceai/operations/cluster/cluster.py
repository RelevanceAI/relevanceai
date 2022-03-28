import numpy as np
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from relevanceai.client.helpers import Credentials
from relevanceai.constants import CLUSTER_APP_LINK, Warning
from relevanceai._api import APIClient


class ClusterOps(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        model: Union[str, Any],
        vector_fields: Optional[List[str]] = None,
        alias: Optional[str] = None,
        n_clusters: Optional[int] = None,
        cluster_config: Optional[Dict[str, Any]] = None,
        outlier_value: int = -1,
        outlier_label: str = "outlier",
        **kwargs,
    ):
        """
        ClusterOps object

        Parameters
        -------------

        model: Union[str, Any]
            The string of clustering algorithm, class of clustering algorithm or custom clustering class.
            If custom, the model must contain the method for fit_predict and must output a numpy array for labels

        vector_fields: Optional[List[str]] = None
            A list of vector_fields to cluster over

        alias: Optional[str] = None
            An alias to be given for clustering.
            If no alias is provided, alias becomes the cluster model name hypen n_clusters e.g. (kmeans-5, birch-8)

        n_clusters: Optional[int] = None
            Number of clusters to find, should the argument be required of the algorithm.
            If None, n_clusters defaults to 8 for sklearn models ONLY.

        cluster_config: Optional[Dict[str, Any]] = None
            Hyperparameters for the clustering model.
            See the documentation for supported clustering models for list of hyperparameters

        outlier_value: int = -1
            For unsupervised clustering models like OPTICS etc. outliers are given the integer label -1

        outlier_label: str = "outlier"
            When viewing the cluster app dashboard, outliers will be prefixed with outlier_label

        """
        self.vector_field = None if vector_fields is None else vector_fields[0]

        self.config = {} if cluster_config is None else cluster_config  # type: ignore
        if n_clusters is not None:
            self.config["n_clusters"] = n_clusters  # type: ignore

        self.model_name = None
        self.model = self._get_model(model)

        if "package" not in self.__dict__:
            self.package = self._get_package(self.model)

        if hasattr(self.model, "n_clusters"):
            self.n_clusters = self.model.n_clusters
        else:
            self.n_clusters = n_clusters

        self.alias = self._get_alias(alias)
        self.outlier_value = outlier_value
        self.outlier_label = outlier_label

        super().__init__(credentials, **kwargs)

    def __call__(self, dataset_id: str, vector_fields: List[str]) -> None:
        self.forward(dataset_id=dataset_id, vector_fields=vector_fields)

    def _get_alias(self, alias: Any) -> str:
        # Auto-generates alias here
        if alias is None:
            if hasattr(self.model, "n_clusters"):
                n_clusters = (
                    self.n_clusters
                    if self.n_clusters is not None
                    else self.model.n_clusters
                )
                alias = f"{self.model_name}-{n_clusters}"

            elif hasattr(self.model, "k"):
                n_clusters = (
                    self.n_clusters if self.n_clusters is not None else self.model.k
                )
                alias = f"{self.model_name}-{n_clusters}"

            else:
                alias = self.model_name

            Warning.MISSING_ALIAS.format(alias=alias)
        return alias.lower()

    def _get_package(self, model):
        model_name = str(model.__class__)
        if "function" in model_name:
            model_name = str(model.__name__)

        if "sklearn" in model_name:
            package = "sklearn"

        elif "faiss" in model_name:
            package = "faiss"

        elif "hdbscan" in model_name:
            package = "hdbscan"

        elif "community_detection" in model_name:
            package = "sentence-transformers"

        return package

    def _get_model(self, model):
        if isinstance(model, str):
            model = model.lower().replace(" ", "").replace("_", "")
            self.model_name = model
        else:
            self.model_name = str(model.__class__).split(".")[-1].split("'>")[0]
            self.package = "custom"
            return model

        if isinstance(model, str):
            if model == "affinitypropagation":
                from sklearn.cluster import AffinityPropagation

                model = AffinityPropagation(**self.config)

            elif model == "agglomerativeclustering":
                from sklearn.cluster import AgglomerativeClustering

                model = AgglomerativeClustering(**self.config)

            elif model == "birch":
                from sklearn.cluster import Birch

                model = Birch(**self.config)

            elif model == "dbscan":
                from sklearn.cluster import DBSCAN

                model = DBSCAN(**self.config)

            elif model == "optics":
                from sklearn.cluster import OPTICS

                model = OPTICS(**self.config)

            elif model == "kmeans":
                from sklearn.cluster import KMeans

                model = KMeans(**self.config)

            elif model == "featureagglomeration":
                from sklearn.cluster import FeatureAgglomeration

                model = FeatureAgglomeration(**self.config)

            elif model == "meanshift":
                from sklearn.cluster import MeanShift

                model = MeanShift(**self.config)

            elif model == "minibatchkmeans":
                from sklearn.cluster import MiniBatchKMeans

                model = MiniBatchKMeans(**self.config)

            elif model == "spectralclustering":
                from sklearn.cluster import SpectralClustering

                model = SpectralClustering(**self.config)

            elif model == "spectralbiclustering":
                from sklearn.cluster import SpectralBiclustering

                model = SpectralBiclustering(**self.config)

            elif model == "spectralcoclustering":
                from sklearn.cluster import SpectralCoclustering

                model = SpectralCoclustering(**self.config)

            elif model == "hdbscan":
                from hdbscan import HDBSCAN

                model = HDBSCAN(**self.config)

            elif model == "community_detection":
                # TODO: this is a callable (?)
                from sentence_transformers.util import community_detection

                model = community_detection

            elif "faiss" in model:
                from faiss import Kmeans

                model = Kmeans(**self.config)

        else:
            # TODO: this needs to be referenced from relevance.constants.errors
            raise ValueError("ModelNotSupported")

        return model

    def _format_labels(self, labels: np.ndarray) -> List[str]:
        labels = labels.flatten().tolist()
        cluster_labels = [
            f"cluster-{str(label)}"
            if label == self.outlier_value
            else self.outlier_label
            for label in labels
        ]
        return cluster_labels

    def _get_centroid_documents(
        self, vectors: np.ndarray, labels: List[str]
    ) -> List[Dict[str, Any]]:
        centroid_documents = []

        centroids: Dict[str, Any] = {}
        for label, vector in zip(labels, vectors.tolist()):
            if label not in centroids:
                centroids[label] = []
            centroids[label].append(vector)

        for centroid, vectors in centroids.items():
            centroid_vector = np.array(vectors).mean(0).tolist()
            centroid_document = dict(
                _id=centroid,
                centroid_vector=centroid_vector,
            )
            centroid_documents.append(centroid_document)

        return centroid_documents

    def _insert_centroids(
        self,
        dataset_id: str,
        vector_field: str,
        centroid_documents: List[Dict[str, Any]],
    ) -> None:
        self.services.cluster.centroids.insert(
            dataset_id=dataset_id,
            cluster_centers=centroid_documents,
            vector_fields=[vector_field],
            alias=self.alias,
        )

    def _fit_predict(
        self, documents: List[Dict[str, Any]], vector_field: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        vectors = np.array(
            [
                self.get_field(vector_field, document)
                for document in documents
                if self.is_field(vector_field, document)
            ]
        )

        if self.package == "sklearn":
            labels = self.model.fit_predict(vectors)

        elif self.package == "hdbscan":
            self.model.fit(vectors)
            labels = self.model.labels_.reshape(-1, 1)

        elif self.package == "faiss":
            vectors = vectors.astype("float32")
            self.model.train(vectors)
            labels = self.model.assign(vectors)[1]

        elif self.package == "sentencetransformers":
            # TODO: make teh config here better
            labels = self.model(vectors)
            labels = np.array(labels)

        elif self.package == "custom":
            labels = self.model.fit_predict(vectors)
            if not isinstance(labels, np.ndarray):
                raise ValueError("Custom model must output np.ndarray for labels")

            if labels.shape[0] != vectors.shape[0]:
                raise ValueError("incorrect shape")

        labels = self._format_labels(labels)

        cluster_field = f"_cluster_.{vector_field}.{self.alias}"
        self.set_field_across_documents(
            field=cluster_field, values=labels, docs=documents
        )

        centroid_documents = self._get_centroid_documents(vectors, labels)

        return centroid_documents, documents

    def _print_app_link(self):
        link = CLUSTER_APP_LINK.format(self.dataset_id)
        print(link)

    def forward(
        self,
        dataset_id: Union[str, Any],
        vector_fields: List[str],
        show_progress_bar: bool = True,
    ) -> None:
        if not isinstance(dataset_id, str):
            if hasattr(dataset_id, "dataset_id"):
                dataset_id = dataset_id.dataset_id

        self.dataset_id = dataset_id
        vector_field = vector_fields[0]
        self.vector_field = vector_field

        # get all documents
        documents = self._get_all_documents(
            dataset_id=dataset_id,
            select_fields=vector_fields,
            show_progress_bar=show_progress_bar,
            include_vector=True,
        )

        # fit model, predict and label all documents
        centroid_documents, labelled_documents = self._fit_predict(
            documents=documents,
            vector_field=vector_field,
        )

        # update all documents
        self._update_documents(
            dataset_id=dataset_id,
            documents=labelled_documents,
            show_progress_bar=show_progress_bar,
        )

        # insert centroids
        self._insert_centroids(
            dataset_id=dataset_id,
            vector_field=vector_field,
            centroid_documents=centroid_documents,
        )

        # link back to dashboard
        self._print_app_link()

    def closest(
        self,
        dataset_id: Optional[str] = None,
        vector_field: Optional[str] = None,
        alias: Optional[str] = None,
    ):
        dataset_id = self.dataset_id if dataset_id is None else dataset_id
        vector_field = self.vector_field if vector_field is None else vector_field
        alias = self.alias if alias is None else alias

        return self.services.cluster.centroids.list_closest_to_center(
            dataset_id=dataset_id,
            vector_fields=[vector_field],
            alias=alias,
        )

    def furthest(
        self,
        dataset_id: Optional[str] = None,
        vector_field: Optional[str] = None,
        alias: Optional[str] = None,
    ):
        """
        List furthest from cluster centers
        """
        dataset_id_ = self.dataset_id if dataset_id is None else dataset_id
        vector_field_ = self.vector_field if vector_field is None else vector_field
        alias_ = self.alias if alias is None else alias

        return self.services.cluster.centroids.list_furthest_from_center(
            dataset_id=dataset_id_,
            vector_fields=[vector_field_],  # type: ignore
            alias=alias_,
        )

    # Convenience functions
    list_closest = closest
    list_furthest = furthest
