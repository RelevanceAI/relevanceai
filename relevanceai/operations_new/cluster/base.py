"""
Base class for clustering
"""
from typing import List, Dict, Any, Optional
from relevanceai.operations_new.cluster.models.base import ModelBase
from relevanceai.operations_new.run import OperationRun


class ClusterBase(OperationRun):

    model: ModelBase

    def __init__(
        self,
        vector_fields: List[str],
        alias: str,
        model: Any,
        cluster_field: str = "_cluster_",
        model_kwargs: Optional[dict] = None,
        **kwargs,
    ):

        self.vector_fields = vector_fields

        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        self.alias = self._get_alias(alias)
        self.model = self._get_model(model=model, model_kwargs=self.model_kwargs)

        self.cluster_field = cluster_field

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._check_vector_fields()

    def _get_model(self, model: Any, model_kwargs: dict) -> Any:
        if isinstance(model, str):
            model = self._get_model_from_string(model, model_kwargs)

        elif "sklearn" in model.__module__:
            model = self._get_sklearn_model_from_class(model)

        elif "faiss" in model.__module__:
            model = self._get_faiss_model_from_class(model)

        return model

    def _get_sklearn_model_from_class(self, model):
        from relevanceai.operations_new.cluster.models.sklearn.base import (
            SklearnModelBase,
        )

        model_kwargs = model.__dict__
        model = SklearnModelBase(model=model, model_kwargs=model_kwargs)
        return model

    def _get_faiss_model_from_class(self, model):
        raise NotImplementedError

    def normalize_model_name(self, model):
        if isinstance(model, str):
            return model.lower().replace("-", "").replace("_", "")
        return model

    def _get_model_from_string(self, model: str, model_kwargs: dict = None):
        if model_kwargs is None:
            model_kwargs = {}

        model = self.normalize_model_name(model)
        model_kwargs = {} if model_kwargs is None else model_kwargs

        from relevanceai.operations_new.cluster.models.sklearn import sklearn_models

        if model in sklearn_models:
            from relevanceai.operations_new.cluster.models.sklearn.base import (
                SklearnModelBase,
            )

            model = SklearnModelBase(
                model=model,
                model_kwargs=model_kwargs,
            )
            return model

        elif model == "communitydetection":
            from relevanceai.operations_new.cluster.models.sentence_transformers.community_detection import (
                CommunityDetection,
            )

            model = CommunityDetection(**model_kwargs)
            return model

        raise ValueError("Model not supported.")

    @property
    def name(self):
        return "cluster"

    def format_cluster_label(self, label):
        """> If the label is an integer, return a string that says "cluster_" and the integer. If the label is
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
                    warm_start=warm_start,
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

        # TODO: add support for sklearn kmeans
        if not self.is_field_across_documents(self.vector_fields[0], documents):
            raise ValueError(
                "You have missing vectors in your document. You will want to apply a filter for vector fields. See here for a page of filter options: https://relevanceai.readthedocs.io/en/development/core/filters/exists.html#exists."
            )
        labels = self.fit_predict_documents(
            documents=documents,
        )
        # Get the cluster field name
        cluster_field_name = self._get_cluster_field_name()

        documents_to_upsert = [{"_id": d["_id"]} for d in documents]

        self.set_field_across_documents(cluster_field_name, labels, documents_to_upsert)
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
