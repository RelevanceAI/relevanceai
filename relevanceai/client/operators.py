from relevanceai.client.helpers import Credentials
from relevanceai.utils.decorators.analytics import track


class Operators:
    credentials: Credentials

    @track
    def ClusterOps(
        self,
        dataset_id: str,
        vector_fields: list,
        alias: str,
        model=None,
        **kwargs,
    ):
        from relevanceai.operations_new.cluster.ops import ClusterOps

        return ClusterOps(
            credentials=self.credentials,
            dataset_id=dataset_id,
            vector_fields=vector_fields,
            alias=alias,
            model=model,
            **kwargs,
        )

    @track
    def SubClusterOps(
        self,
        credentials,
        alias,
        dataset,
        model,
        vector_fields: list,
        parent_field: str,
    ):
        """
        Sub Cluster Ops.
        """
        from relevanceai.operations.cluster.sub import SubClusterOps

        return SubClusterOps(
            credentials=self.credentials,
            alias=alias,
            dataset=dataset,
            model=model,
            vector_fields=vector_fields,
            parent_field=parent_field,
        )

    @track
    def LabelOps(
        self,
        **kwargs,
    ):
        from relevanceai.operations_new.label.ops import LabelOps

        return LabelOps(
            credentials=self.credentials,
            **kwargs,
        )

    @track
    def VectorizeOps(
        self,
        **kwargs,
    ):
        from relevanceai.operations_new.vectorize.ops import VectorizeOps

        return VectorizeOps(
            credentials=self.credentials,
            **kwargs,
        )

    @track
    def DimReductionOps(
        self,
        **kwargs,
    ):
        from relevanceai.operations_new.vectorize.ops import VectorizeOps

        return VectorizeOps(
            credentials=self.credentials,
            **kwargs,
        )
