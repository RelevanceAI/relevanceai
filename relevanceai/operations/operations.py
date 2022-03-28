from typing import List, Dict, Optional

from relevanceai.client.helpers import Credentials
from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.vector import Vectorize
from relevanceai.operations.vector import Search
from relevanceai.operations.dr import ReduceDimensionsOps
from relevanceai.utils.decorators import deprecated
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
        model: str,
        vector_fields: List[str],
        alias: Optional[str] = None,
        **kwargs,
    ):
        ops = ClusterOps(
            credentials=self.credentials,
            model=model,
            alias=alias,
            **kwargs,
        )
        return ops(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
        )

    @deprecated(version="2.0", message="auto_cluster does not work as intended")
    def auto_cluster(
        self,
        alias: str,
        vector_fields: List[str],
        **kwargs,
    ):
        model = alias.split("-")[0]
        ops = ClusterOps(
            credentials=self.credentials,
            model=model,
            alias=alias,
            **kwargs,
        )
        return ops(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
        )

    def dr(
        self,
        model,
        n_components,
        alias,
        vector_fields,
        **kwargs,
    ):
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
        ops = Vectorize(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
            **kwargs,
        )
        return ops.vectorize(
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
