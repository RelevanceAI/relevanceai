from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.dr import ReduceDimensionsOps


class Operations:
    project: str
    api_key: str
    firebase_uid: str
    dataset_id: str

    def cluster(self, model, alias, vector_fields, **kwargs):
        ops = ClusterOps(
            project=self.project,
            api_key=self.api_key,
            firebase_uid=self.firebase_uid,
            model=model,
            **kwargs
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
            **kwargs
        )
        return ops.fit(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
            alias=alias,
        )
