"""Base Workflow
"""
import traceback
from typing import Callable, Optional, Union
from doc_utils import DocUtils
from uuid import uuid4

from relevanceai.dataset import Dataset


class Workflow(DocUtils):
    """
    Base Workflow. A workflow is useful for measuring what you did with a dataset.
    By adding an alias, you allow it to continue running even when it errors.
    """

    def __init__(
        self, func: Callable, workflow_alias: str, notes: Optional[str] = None
    ):
        """ """
        self.func = func
        self.workflow_alias = workflow_alias
        self.notes = notes

    def fit_dataset(
        self,
        dataset: Dataset,
        input_field: str,
        output_field: str,
        filters: Optional[list] = None,
        log_to_file: bool = True,
        chunksize: int = 20,
        chunk_field: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        """
        Fit on dataset
        """
        filters = [] if filters is None else filters

        self.dataset = dataset
        exist_filters = [
            {
                "field": input_field,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " ",
            },
            {
                "field": output_field,
                "filter_type": "exists",
                "condition": "!=",
                "condition_value": " ",
            },
        ]
        filters += exist_filters

        if chunk_field is not None:
            if chunk_field not in output_field:
                output_field = chunk_field + "." + output_field

            if chunk_field not in input_field:
                input_field = chunk_field + "." + input_field

        def update_func(doc):
            if chunk_field:
                try:
                    self.run_function_across_chunks(
                        function=self.func,
                        chunk_field=chunk_field,
                        field=input_field,
                        output_field=output_field,
                        doc=doc,
                    )
                except Exception as e:
                    traceback.print_exc()
            else:
                try:
                    value = self.get_field(input_field, doc)
                    self.set_field(output_field, doc, self.func(value))
                except Exception as e:
                    traceback.print_exc()
            return doc

        # Store this workflow inside the dataset's metadata
        results = self.dataset.apply(
            update_func,
            select_fields=[input_field],
            filters=filters,
            log_to_file=log_to_file,
            retrieve_chunksize=chunksize,
            log_file=log_file,
        )
        self._store_workflow_to_metadata(input_field, output_field)
        return results

    def _store_workflow_to_metadata(
        self, input_field, output_field, run_id: str = None
    ):
        if run_id is None:
            run_id = uuid4().__str__()
        workflow_metadata = {
            "input_field": input_field,
            "output_field": output_field,
            "workflow_alias": self.workflow_alias,
            "run_id": run_id,
        }
        if self.notes is not None:
            workflow_metadata["notes"] = self.notes
        metadata = self.dataset.metadata
        if "workflows" not in metadata:
            metadata["workflows"] = []
        workflows = metadata["workflows"]
        workflows.append(workflow_metadata)
        self.dataset.metadata["workflows"] = workflows
