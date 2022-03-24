from relevanceai._api import APIClient

from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.vector import Vectorize
from relevanceai.operations.dr import ReduceDimensionsOps


class Operations(APIClient):
    def __init__(
        self,
        project: str,
        api_key: str,
        firebase_uid: str,
        dataset_id: str,
    ):
        self.dataset_id = dataset_id
        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)

    def cluster(self, model, alias, vector_fields, **kwargs):
        ops = ClusterOps(
            project=self.project,
            api_key=self.api_key,
            firebase_uid=self.firebase_uid,
            model=model,
            **kwargs,
        )
        return ops.fit(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
            alias=alias,
        )

    def dr(self, model, alias, vector_fields, **kwargs):
        ops = ReduceDimensionsOps(
            project=self.project,
            api_key=self.api_key,
            firebase_uid=self.firebase_uid,
            model=model,
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
            project=self.project,
            api_key=self.api_key,
            firebase_uid=self.firebase_uid,
            dataset_id=self.dataset_id,
            **kwargs,
        )
        return ops.vectorize(
            text_fields=text_fields,
            image_fields=image_fields,
        )