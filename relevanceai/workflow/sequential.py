"""
Sequential Workflows allow users to quickly swap in and out 
new operations to test new things all while having different
operations logged.

Example
---------



"""
from doc_utils import DocUtils
from relevanceai.utils.logger import FileLogger
from relevanceai.dataset.dataset import Dataset
from abc import abstractmethod, ABC


class Ops(DocUtils):
    """Base class for operations"""

    # @abstractmethod
    # def __call__(self, data):
    #     # Passes through documents or a dataset object
    #     raise NotImplementedError


class UpdateOp(DocUtils):
    """Inherit this class if you want documents to update
    at the end of this call
    """

    _update_: bool = True


class Input(Ops):
    # How do I link the original inputs to the outputs
    def __init__(self, input_fields: list, chunksize: int = 20):
        self.input_fields = input_fields
        self.chunksize = chunksize

    def __call__(self, dataset):
        return self.__iter__(dataset)

    def __iter__(self, dataset: Dataset):
        for c in dataset.chunk_dataset(
            chunksize=self.chunksize, select_fields=self.input_fields
        ):
            yield c
        return None


class Output(Ops):
    """Update documents"""

    _update_: bool = True

    def __init__(self, output_field: list):
        self.output_field = output_field

    def __call__(self, values, documents, dataset: Dataset):
        self.set_field_across_documents(self.output_field, values, documents)
        upsert_results = dataset.upsert_documents(documents)
        return upsert_results, documents


class SequentialWorkflow(DocUtils):
    """
    Defining a workflow with input and output fields and
    linking them up accordingly.

    Example
    ----------

    .. code-block::

        import random
        from relevanceai import Client
        from relevanceai.workflow.sequential import SequentialWorkflow

        client = Client()

        def vectorize(docs, *args, **kw):
            return [[random.randint(0, 100) for _ in range(3)] for _ in range(len(docs))]

        workflow = SequentialWorkflow(
            [
                Input(["sample_1_label"], chunksize=50),
                vectorize,
                Output("simple")
            ],
            log_filename="logs"
        )

        ds = client.Dataset("_mock_dataset_")
        workflow.run(ds, verbose=True)

    """

    def __init__(
        self, list_of_operations: list, log_filename: str = "workflow-run.txt"
    ):
        # TODO: make log filename generated otf
        self.list_of_operations = list_of_operations
        self.log_filename = log_filename

    def __call__(self, dataset):
        if isinstance(dataset, Dataset):
            # run on dataset
            return self.run(dataset)
        elif isinstance(dataset, list):
            return self.run_documents(dataset)

        raise ValueError(
            "Data type not supported. Please ensure it is a list of dicts of a Dataset object."
        )

    def run(self, dataset: Dataset, verbose: bool = True, log_to_file: bool = True):
        """
        Run the sequential workflow
        """
        if len(self.list_of_operations) == 0:
            return

        with FileLogger(self.log_filename, verbose=verbose, log_to_file=log_to_file):
            input_operator: Input = self.list_of_operations[0]

            if not hasattr(input_operator, "input_fields"):
                raise ValueError("The first operation needs to be Input.")

            # TODO: add progress bar
            for i, documents in enumerate(input_operator(dataset)):
                if documents is None:
                    return
                if len(input_operator.input_fields) == 1:
                    values = self.get_field_across_documents(
                        input_operator.input_fields[0], documents
                    )
                elif len(input_operator.input_fields) > 1:
                    values = self.get_fields_across_documents(
                        input_operator.input_fields[0], documents
                    )
                else:
                    raise ValueError("Input Operator")
                if verbose:
                    print(values)
                for j, op in enumerate(self.list_of_operations[1:]):
                    if hasattr(op, "operate"):
                        values = op.operate(values)
                        if verbose:
                            print(values)
                    elif hasattr(op, "_update_") and op._update_:
                        if verbose:
                            print("updating")
                            print(values)
                            print(documents)
                        self.set_field_across_documents(
                            op.output_field, values, documents
                        )
                        if verbose:
                            print(documents)
                        upsert_results = dataset.upsert_documents(documents)
                        if verbose:
                            print(upsert_results)
                    else:
                        values = op(values)
                        if verbose:
                            print(values)
        return
