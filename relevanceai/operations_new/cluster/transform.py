"""
Base class for clustering
"""
import warnings
from typing import List, Dict, Any, Optional
from relevanceai.operations_new.cluster.models.base import ClusterModelBase
from relevanceai.operations_new.transform_base import TransformBase
from relevanceai.operations_new.cluster.alias import ClusterAlias


class ClusterTransform(TransformBase, ClusterAlias):

    model: ClusterModelBase

    def __init__(
        self,
        vector_fields: List[str],
        alias: str,
        model: Any,
        model_kwargs: Optional[dict] = None,
        cluster_field: str = "_cluster_",
        silhouette_score:bool = False,
        **kwargs,
    ):

        self.vector_fields = vector_fields
        self.cluster_field = cluster_field

        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        self.alias = self._get_alias(alias)
        self.model = self._get_model(model=model, model_kwargs=self.model_kwargs)

        self.silhouette_score = silhouette_score
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._check_vector_fields()

    @property
    def name(self):
        return "cluster"

    # def centroids(self):
    #     if hasattr(self.model, "_centroids") and self.model._centroids is not None:
    #         centroid_vectors = self.model._centroids
    #         # get the cluster label function
    #         labels = range(len(centroid_vectors))
    #         cluster_ids = self.format_cluster_labels(labels)
    #         if len(self.vector_fields) > 1:
    #             warnings.warn(
    #                 "Currently do not support inserting centroids with multiple vector fields"
    #             )
    #         centroids = [
    #             {"_id": k, self.vector_fields[0]: v}
    #             for k, v in zip(cluster_ids, centroid_vectors)
    #         ]
    #         return centroids
    #     else:
    #         return []

    def fit_predict_documents(self, documents, warm_start=False):
        """
        If warm_start=True, copies the values from the previous fit.
        Only works for cluster models that use centroids. You should
        not have to use this parameter.
        """
        # run fit predict on documetns
        if hasattr(self.model, "fit_predict_documents"):
            return self.model.fit_predict_documents(
                documents=documents,
                vector_fields=self.vector_fields,
                warm_start=warm_start,
            )
        elif hasattr(self.model, "fit_predict"):
            if len(self.vector_fields) == 1:
                vectors = self.get_field_across_documents(
                    self.vector_fields[0],
                    documents,
                )
                cluster_labels = self.model.fit_predict(
                    vectors,
                    # warm_start=warm_start,
                )

                return self.format_cluster_labels(cluster_labels)
        raise AttributeError("Model is missing a `fit_predict` method.")

    def transform(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """It takes a list of documents, and for each document, it runs the document through each of the
        models in the pipeline, and returns the updated documents.

        Parameters
        ----------
        documents : List[Dict[str, Any]]
            List[Dict[str, Any]]

        Returns
        -------
            A list of dictionaries.

        """

        if not self.is_field_across_documents(self.vector_fields[0], documents):
            raise ValueError(
                "You have missing vectors in your document. You need to filter out documents that don't contain the vector field."
            )
        labels = self.fit_predict_documents(
            documents=documents,
        )
        # Get the cluster field name
        cluster_field_name = self._get_cluster_field_name()

        documents_to_upsert = [{"_id": d["_id"]} for d in documents]
        self.set_field_across_documents(cluster_field_name, labels, documents_to_upsert)
        if self.silhouette_score:
            try:
                from sklearn.metrics import silhouette_samples
                vectors = self.get_field_across_documents(
                    self.vector_fields[0],
                    documents,
                )
                silhouettes = silhouette_samples(vectors, labels)
                self.set_field_across_documents(f'_silhouette_score_{self.alias}', silhouettes, documents_to_upsert)
            except ImportError:
                raise ImportError("sklearn missing")
            except:
                print("Couldn't calculate silhouette scores")
        return documents_to_upsert

    def _get_cluster_field_name(self):
        alias = self.alias
        if isinstance(self.vector_fields, list):
            if hasattr(self, "cluster_field"):
                set_cluster_field = (
                    f"{self.cluster_field}.{'.'.join(self.vector_fields)}.{alias}"
                )
            else:
                set_cluster_field = f"_cluster_.{'.'.join(self.vector_fields)}.{alias}"

        elif isinstance(self.vector_fields, str):
            set_cluster_field = f"{self.cluster_field}.{self.vector_fields}.{alias}"

        return set_cluster_field

    def format_cluster_label(self, label):
        """> If the label is an integer, return a string that says "cluster-" and the integer. If the label is
        a string, return the string. If the label is neither, raise an error

        Parameters
        ----------
        label
            the label of the cluster. This can be a string or an integer.

        Returns
        -------
            A list of lists.

        """
        if isinstance(label, str):
            return label
        return "cluster_" + str(label)

    def format_cluster_labels(self, labels):
        return [self.format_cluster_label(label) for label in labels]
