"""
Base class for clustering
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Dict, Any
from relevanceai.operations_new.cluster.models.base import ModelBase
from relevanceai.operations_new.base import OperationBase


class ClusterBase(OperationBase, ABC):

    model: ModelBase

    def __init__(
        self,
        vector_fields: List[str],
        alias: str,
        model: Any,
        model_kwargs,
        cluster_field: str = "_cluster_",
        **kwargs,
    ):

        self.vector_fields = vector_fields
        self.alias = alias

        if model_kwargs is None:
            model_kwargs = {}
        self.model = self._get_model(
            model=model,
        )
        self.cluster_field = cluster_field
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._check_vector_fields()

    def _get_model(self, model):
        # TODO: change this from abstract to an actual get_model
        from relevanceai.operations_new.cluster.models.base import ModelBase

        if isinstance(model, str):
            return self._get_model_from_string(model)

        elif isinstance(model, ModelBase):
            return model
        elif model is None:
            return model
        raise NotImplementedError

    def normalize_model_name(self, model):
        if isinstance(model, str):
            return model.lower().replace("-", "").replace("_", "")
        return model

    def _get_model_from_string(self, model: str, *args, **kwargs):
        model = self.normalize_model_name(model)
        if model == "kmeans":
            from relevanceai.operations_new.cluster.models.sklearn.kmeans import (
                KMeansModel,
            )

            model = KMeansModel(*args, **kwargs)
            return model
        elif model == "communitydetection":
            raise NotImplementedError("Community detection not supported yet")
        raise ValueError("Model not supported.")

    @property
    def name(self):
        return "cluster"

    def run(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        updated_documents = deepcopy(documents)

        labels = self.model.fit_predict_documents(
            vector_fields=self.vector_fields,
            documents=updated_documents,
        )
        # Get the cluster field name
        cluster_field_name = self._get_cluster_field_name()

        self.set_field_across_documents(cluster_field_name, labels, updated_documents)

        # removes unnecessary info for updated_where
        updated_documents = [
            {
                key: value
                for key, value in document.items()
                if key not in self.vector_fields or key == "_id"
            }
            for document in updated_documents
        ]

        return updated_documents

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
