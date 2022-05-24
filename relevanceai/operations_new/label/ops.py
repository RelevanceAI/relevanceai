"""
Labelling with API-related functions
"""
from relevanceai.dataset import Dataset
from relevanceai.operations_new.label.base import LabelBase
from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.context import Upload


class LabelOps(LabelBase, OperationAPIBase):  # type: ignore
    """
    Label Operations
    """

    def run(
        self,
        dataset: Dataset,
        vector_fields: list = None,
        filters: list = None,
        *args,
        **kwargs,
    ):

        with Upload(
            dataset=dataset,
            operation=self,
            metadata=kwargs,
        ) as dataset:

            documents = dataset.get_all_documents(
                filters=filters,
                select_fields=vector_fields,
            )

            res = self.transform(
                documents,
                *args,
                **kwargs,
            )

        return
