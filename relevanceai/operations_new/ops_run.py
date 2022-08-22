"""
Base class for base.py to inherit.
All functions related to running operations on datasets.
"""
import time
import psutil
import threading
import multiprocessing as mp
import warnings

from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

from relevanceai.dataset import Dataset
from relevanceai.operations_new.transform_base import TransformBase

from tqdm.auto import tqdm


class PullUpdatePush:
    def __init__(
        self,
        dataset: Dataset,
        func: Callable,
        func_args: Optional[Tuple[Any, ...]] = None,
        func_kwargs: Optional[Dict[str, Any]] = None,
        multithreaded_update: bool = True,
        pull_batch_size: Optional[int] = 128,
        update_batch_size: Optional[int] = 128,
        push_batch_size: Optional[int] = 128,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        update_workers: int = 1,
        push_workers: int = 1,
        buffer_size: int = 0,
        show_progress_bar: bool = True,
        timeout: Optional[int] = None,
        ingest_in_background: bool = False,
        block_return: bool = False,
        ram_ratio: float = 0.25,
    ):
        """
        Buffer size:
            number of documents to be held in limbo by both queues at any one time

        """
        super().__init__()

        self.dataset = dataset
        self.dataset_id = dataset.dataset_id

        ndocs = self.dataset.get_number_of_documents(
            dataset_id=self.dataset_id,
            filters=filters,
        )
        self.ndocs = ndocs

        self.pull_batch_size = min(pull_batch_size, ndocs)
        self.update_batch_size = min(update_batch_size, ndocs)
        self.push_batch_size = min(push_batch_size, ndocs)

        self.timeout = timeout
        self.ingest_in_background = ingest_in_background

        self.filters = [] if filters is None else filters
        self.select_fields = [] if select_fields is None else select_fields

        self.lock = threading.Lock()
        self.func_lock: Union[threading.Lock, None]

        if not multithreaded_update:
            self.func_lock = threading.Lock()
            self.update_workers = 1
        else:
            self.func_lock = None
            self.update_workers = update_workers

        self.push_workers = push_workers

        self.func_args = () if func_args is None else func_args
        self.func_kwargs = {} if func_kwargs is None else func_kwargs

        if buffer_size == 0:
            ram_size = psutil.virtual_memory().total  # in bytes
            average_document_size_inmem = 2**17
            self.total_buffer_size = int(
                ram_size * ram_ratio / average_document_size_inmem
            )
        else:
            self.total_buffer_size = buffer_size

        self.single_buffer_size = int(self.total_buffer_size / 2)
        tqdm.write(f"Setting single queue size to be: {self.single_buffer_size:,}")

        self.tq: mp.Queue = mp.Queue(maxsize=self.single_buffer_size)
        self.pq: mp.Queue = mp.Queue(maxsize=self.single_buffer_size)
        self.func = func

        self.tqdm_kwargs = dict(leave=False, disable=(not show_progress_bar))
        self.block_return = block_return

    def pull(self):

        documents: List[Dict[str, Any]] = [{"placeholder": "placeholder"}]
        after_id: Union[str, None] = None
        overflow: List[Dict[str, Any]] = []

        while documents:
            res = self.dataset.datasets.documents.get_where(
                dataset_id=self.dataset_id,
                page_size=self.pull_batch_size,
                filters=self.filters,
                select_fields=self.select_fields,
                sort=[],
                after_id=after_id,
            )
            documents = res["documents"]
            after_id = res["after_id"]

            pool = documents + overflow
            if len(pool) >= self.update_batch_size:
                n_batches = int(len(pool) / self.update_batch_size)

                for batch_idx in range(n_batches):
                    batch_start = batch_idx * self.update_batch_size
                    batch_end = (batch_idx + 1) * self.update_batch_size
                    batch = pool[batch_start:batch_end]

                    self.tq.put(batch)

                    with self.lock:
                        self.pull_bar.update(len(batch))

                overflow = pool[n_batches * self.update_batch_size :]
            else:
                overflow = pool

        if overflow:
            self.tq.put(overflow)
            with self.lock:
                self.pull_bar.update(len(overflow))

    def update(self):
        while self.update_bar.n < self.ndocs:
            try:
                batch = self.tq.get(timeout=self.timeout)
            except:
                break

            if self.func_lock is not None:
                with self.func_lock:
                    new_batch = self.func(batch, *self.func_args, **self.func_kwargs)
            else:
                new_batch = self.func(batch, **self.func_kwargs)
            batch = PullUpdatePush._postprocess(new_batch, batch)

            for document in batch:
                self.pq.put(document)

            with self.lock:
                self.update_bar.update(len(batch))

    @staticmethod
    def _postprocess(
        new_batch: List[Dict[str, Any]],
        old_batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        batch = [
            {
                key: value
                for key, value in new_batch[idx].items()
                if key not in old_batch[idx].keys() or key == "_id"
            }
            for idx in range(len(new_batch))
        ]
        return batch

    def _handle_failed_documents(
        self,
        res: Dict[str, Any],
        batch: List[Dict[str, Any]],
    ) -> None:

        # check if there is any failed documents...
        failed_documents = res["response_json"]["failed_documents"]

        if failed_documents:
            with self.lock:
                self.ndocs += len(failed_documents)
                desc = f"push - failed_documents = {self.ndocs - self.ndocs}"
                self.push_bar.set_description(desc)

            # ...find these failed documents within the batch...
            failed_ids = set(map(lambda x: x["_id"], failed_documents))
            failed_documents = [
                document for document in batch if document["_id"] in failed_ids
            ]

            # ...and re add them to the push queue
            for failed_document in failed_documents:
                self.pq.put(failed_document)

        return failed_documents

    def push(self):

        batch = [self.pq.get(timeout=self.timeout)]

        while self.push_bar.n < self.ndocs:
            while len(batch) < self.push_batch_size:
                try:
                    document = self.pq.get(timeout=1)
                except:
                    break
                batch.append(document)

            batch = self.dataset.json_encoder(batch)
            res = self.dataset.datasets.documents.bulk_update(
                self.dataset_id,
                batch,
                return_documents=True,
                ingest_in_background=self.ingest_in_background,
            )

            failed_documents = self._handle_failed_documents(res, batch)

            with self.lock:
                self.push_bar.update(len(batch) - len(failed_documents))
                batch = []

    def _init_worker_pool(self):
        self.pull_bar = tqdm(
            desc="pull",
            position=0,
            total=self.ndocs,
            **self.tqdm_kwargs,
        )
        self.update_bar = tqdm(
            range(self.ndocs),
            desc="update",
            position=1,
            **self.tqdm_kwargs,
        )
        self.push_bar = tqdm(
            range(self.ndocs),
            desc="push",
            position=2,
            **self.tqdm_kwargs,
        )
        threads = [
            threading.Thread(
                target=self.pull,
            )
        ]
        threads += [
            threading.Thread(
                target=self.update,
            )
            for _ in range(self.update_workers)
        ]
        threads += [
            threading.Thread(
                target=self.push,
            )
            for _ in range(self.push_workers)
        ]
        for thread in threads:
            thread.start()

        # for thread in threads:
        #     thread.join()

    def run(self):
        """
        Do the pulling, the updating, and of course, the pushing.
        """
        if self.ndocs <= 0:
            return

        self._init_worker_pool()


class OperationRun(TransformBase):
    """
    All functions related to running transforms as an operation on datasets
    """

    def is_chunk_valid(self, chunk):
        return chunk is not None and len(chunk) > 0

    # @abstractmethod
    def post_run(self, dataset, documents, updated_documents):
        pass

    def run(
        self,
        dataset: Dataset,
        batched: Optional[bool] = False,
        chunksize: Optional[int] = 100,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        output_fields: Optional[list] = None,
        refresh: bool = False,
        *args,
        **kwargs,
    ):
        """It takes a dataset, and then it gets all the documents from that dataset. Then it transforms the
        documents and then it upserts the documents.

        Parameters
        ----------
        dataset : Dataset
            Dataset,
        select_fields : list
            Used to determine which fields to retrieve for filters
        output_fields: list
            Used to determine which output fields are missing to continue running operation

        filters : list
            list = None,

        """

        if filters is None:
            filters = []
        if select_fields is None:
            select_fields = []

        # store this
        if hasattr(dataset, "dataset_id"):
            self.dataset_id = dataset.dataset_id

        schema = dataset.schema

        self._check_fields_in_schema(select_fields)

        filters += [
            {
                "filter_type": "or",
                "condition_value": [
                    {
                        "field": field,
                        "filter_type": "exists",
                        "condition": "==",
                        "condition_value": " ",
                    }
                    for field in select_fields
                ],
            }
        ]

        # add a checkmark for output fields
        if not refresh and output_fields is not None and len(output_fields) > 0:
            filters += [
                {
                    "field": output_fields[0],
                    "filter_type": "exists",
                    "condition": "!=",
                    "condition_value": " ",
                }
            ]

        # needs to be here due to circular imports
        from relevanceai.operations_new.ops_manager import OperationManager

        with OperationManager(
            dataset=dataset,
            operation=self,
        ) as dataset:

            if batched:
                self.batch_transform_upsert(
                    dataset=dataset,
                    select_fields=select_fields,
                    filters=filters,
                    chunksize=chunksize,
                    **kwargs,
                )
            else:
                documents = dataset.get_all_documents(
                    select_fields=select_fields,
                    filters=filters,
                )
                updated_documents = self.transform(
                    documents,
                    *args,
                    **kwargs,
                )  # Should be in the transform.py
                dataset.upsert_documents(updated_documents)
                self.post_run(
                    dataset=dataset,
                    documents=documents,
                    updated_documents=updated_documents,
                )  # Should be in the ops.py
        return

    def batch_transform_upsert(
        self,
        dataset: Dataset,
        select_fields: list = None,
        filters: list = None,
        chunksize: int = None,
        max_active_threads: int = 1,
        timeout: int = 30,
        buffer_size: int = 0,
        show_progress_bar: bool = True,
        update_batch_size: int = 32,
        multithreaded_update: bool = False,
        *args,
        **kwargs,
    ):
        if multithreaded_update:
            warnings.warn(
                "Multithreaded-update should be False for vectorizing with 1 GPU only. Could hang if True. Works fine on CPU."
            )
        pup = PullUpdatePush(
            dataset=dataset,
            func=self.transform,
            func_args=args,
            func_kwargs=kwargs,
            multithreaded_update=multithreaded_update,
            pull_batch_size=chunksize,
            update_batch_size=update_batch_size,
            push_batch_size=chunksize,
            filters=filters,
            select_fields=select_fields,
            update_workers=max_active_threads,
            push_workers=max_active_threads,
            buffer_size=buffer_size,
            show_progress_bar=show_progress_bar,
            timeout=timeout,
        )
        pup.run()

    def store_operation_metadata(
        self,
        dataset: Dataset,
        values: Optional[Dict[str, Any]] = None,
    ):
        """This function stores metadata about operators

        Parameters
        ----------
        dataset : Dataset
            Dataset,
        values : Optional[Dict[str, Any]]
            Optional[Dict[str, Any]] = None,

        Returns
        -------
            The dataset object with the metadata appended to it.

        .. code-block::

            {
                "_operationhistory_": {
                    "1-1-1-17-2-3": {
                        "operation": "vector",
                        "model_name": "miniLm"
                    },
                }
            }

        """
        if values is None:
            values = self.get_operation_metadata()

        tqdm.write("Storing operation metadata...")
        timestamp = str(datetime.now().timestamp()).replace(".", "-")
        metadata = dataset.metadata.to_dict()
        if "_operationhistory_" not in metadata:
            metadata["_operationhistory_"] = {}
        metadata["_operationhistory_"].update(
            {
                timestamp: {
                    "operation": self.name,
                    "parameters": str(values),
                }
            }
        )
        # Gets metadata and appends to the operation history
        return dataset.upsert_metadata(metadata)
