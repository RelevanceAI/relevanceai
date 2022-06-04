from relevanceai.client.helpers import Credentials
from relevanceai.utils.decorators.analytics import track


class Operators:
    credentials: Credentials

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
