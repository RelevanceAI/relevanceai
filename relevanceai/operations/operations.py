from typing import Optional, Any, List

from relevanceai.client.helpers import Credentials

from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.vector import VectorizeOps, Search
from relevanceai.operations.dr import ReduceDimensionsOps

from relevanceai._api import APIClient


class Operations(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
    ):
        self.credentials = credentials
        self.dataset_id = dataset_id
        super().__init__(self.credentials)

    def cluster(
        self,
        vector_fields: List[str],
        alias: Optional[str] = None,
        model: Any = "community_detection",
        **kwargs,
    ):
        ops = ClusterOps(
            credentials=self.credentials,
            model=model,
            alias=alias,
            vector_fields=vector_fields,
            dataset_id=self.dataset_id,
            **kwargs,
        )
        return ops(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
        )

    def dr(
        self,
        alias: str,
        vector_fields: List[str],
        model: Any = "umap",
        n_components: int = 3,
        **kwargs,
    ):
        """
        Reduce dimensions

        Parameters
        --------------

        model: Callable
            model to reduce dimensions
        n_components: int
            The number of components
        alias: str
            The alias of the model
        vector_fields: List[str]
            The list of vector fields to support

        """
        ops = ReduceDimensionsOps(
            credentials=self.credentials,
            model=model,
            n_components=n_components,
            **kwargs,
        )
        return ops.fit(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
            alias=alias,
        )

    def vectorize(
        self,
        text_fields=None,
        image_fields=None,
        **kwargs,
    ):
        """
        Vectorize the model
        """
        ops = VectorizeOps(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
            **kwargs,
        )
        return ops(
            text_fields=text_fields,
            image_fields=image_fields,
        )

    def vector_search(self, **kwargs):
        ops = Search(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
        )

        return ops.vector_search(**kwargs)

    def hybrid_search(self, **kwargs):
        ops = Search(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
        )

        return ops.hybrid_search(**kwargs)

    def chunk_search(self, **kwargs):
        ops = Search(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
        )

        return ops.chunk_search(**kwargs)

    def multistep_chunk_search(self, **kwargs):
        ops = Search(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
        )

        return ops.multistep_chunk_search(**kwargs)
