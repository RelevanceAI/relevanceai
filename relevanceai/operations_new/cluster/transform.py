"""
Base class for clustering
"""
import numpy as np
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
        include_cluster_report: bool = False,
        **kwargs,
    ):

        self.vector_fields = vector_fields
        self.cluster_field = cluster_field

        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        self.alias = self._get_alias(alias)
        self.model = self._get_model(model=model, model_kwargs=self.model_kwargs)

        self.include_cluster_report = include_cluster_report

        for k, v in kwargs.items():
            setattr(self, k, v)
        self._check_vector_fields()

    @property
    def name(self):
        return "cluster"

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
        if hasattr(self, "include_cluster_report") and self.include_cluster_report:
            vectors = self.get_field_across_documents(
                self.vector_fields[0],
                documents,
            )
            try:
                self.set_field_across_documents(
                    self._silhouette_score_field_name(),
                    self.calculate_silhouette(vectors, labels),
                    documents_to_upsert,
                )
            except:
                pass
            try:
                self.set_field_across_documents(
                    self._squared_error_field_name(),
                    self.calculate_squared_error(
                        vectors, labels, self.model._centroids
                    ),
                    documents_to_upsert,
                )
            except:
                import traceback

                traceback.print_exc()
                pass
        return documents_to_upsert

    @staticmethod
    def calculate_silhouette(vectors, labels):
        try:
            from sklearn.metrics import silhouette_samples

            return silhouette_samples(vectors, labels)
        except ImportError:
            raise ImportError("sklearn missing")
        except:
            raise Exception("Couldn't calculate silhouette scores")

    def _silhouette_score_field_name(self):
        return f"_silhouette_score_{self.alias}"

    @staticmethod
    def calculate_squared_error(vectors, labels, centroids):
        try:
            label_to_centroid_index = {
                label: i for i, label in enumerate(sorted(np.unique(labels).tolist()))
            }
            return np.square(
                np.subtract(
                    [centroids[label_to_centroid_index[l]] for l in labels],
                    vectors,
                )
            )
        except:
            raise Exception("Couldn't calculate squared errors")

    def _squared_error_field_name(self):
        return f"_squared_error_{self.alias}"

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
