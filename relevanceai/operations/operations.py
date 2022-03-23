from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.dr import ReduceDimensionsOps


class Operations:
    project: str
    api_key: str
    firebase_uid: str

    def cluster(self, model, alias, vector_fields, **kwargs):
        ops = ClusterOps(
            self.project,
            self.api_key,
            self.firebase_uid,
            model=model,
            alias=alias,
            vector_fields=vector_fields,
            **kwargs
        )
        return ops.fit()
