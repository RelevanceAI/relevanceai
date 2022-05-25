"""
OperationBase
"""
from typing import Any, Dict, Optional, List
from relevanceai.operations_new.cluster.alias import ClusterAlias
from relevanceai.operations_new.cluster.batch.models.base import BatchClusterModelBase
from relevanceai.operations_new.base import OperationBase
from relevanceai.operations_new.cluster.base import ClusterBase


class BatchClusterBase(ClusterBase, ClusterAlias):
    def __init__(
        self,
        vector_fields: list,
        model: Any,
        model_kwargs: dict,
        *args,
        **kwargs,
    ):
        self.vector_fields = vector_fields
        self._check_vector_fields()
        self.model = self._get_model(model, model_kwargs)
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs: dict = model_kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

    def partial_fit(self, documents: List, model_kwargs=None):
        """Run partial fitting on a list of documents"""
        if isinstance(self.model, str):
            model_kwargs = {} if model_kwargs is None else model_kwargs
            self.model = self._get_model(self.model, model_kwargs)
        return self.model.partial_fit(
            self.get_field_across_documents(self.vector_fields, documents)
        )

    @property
    def name(self):
        return type(self.model).__name__.lower()

    @property
    def full_cluster_field(self):
        if not hasattr(self, "_cluster_field"):
            self._cluster_field = self._get_cluster_field_name()
            print("Saving cluster labels to: ")
            print(self._cluster_field)
        return self._cluster_field

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
        labels = self.fit_predict_documents(
            documents=documents,
        )
        # Get the cluster field name
        cluster_field_name = self._get_cluster_field_name()

        documents_to_upsert = [{"_id": d["_id"]} for d in documents]

        self.set_field_across_documents(
            cluster_field_name,
            labels,
            documents_to_upsert,
        )
        return documents_to_upsert

    def _get_model(self, model: Any, model_kwargs: dict) -> Any:
        if isinstance(model, str):
            model = self._get_model_from_string(model, model_kwargs)

        elif "sklearn" in model.__module__:
            model = self._get_sklearn_model_from_class(model)

        elif "faiss" in model.__module__:
            model = self._get_faiss_model_from_class(model)

        return model
